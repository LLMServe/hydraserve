"""Utilities for downloading and initializing model weights."""
import fnmatch
import glob
import hashlib
import json
import os
import time
import socket
from multiprocessing import shared_memory
import asyncio
import tempfile
from collections import defaultdict
from typing import Any, Generator, Iterable, List, Optional, Tuple

import filelock
import huggingface_hub.constants
import numpy as np
import torch
from huggingface_hub import HfFileSystem, snapshot_download
from safetensors.torch import load_file, safe_open, save_file
from tqdm.auto import tqdm

from vllm.config import LoadConfig, ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (QuantizationConfig,
                                                     get_quantization_config)
from vllm.model_executor.layers.quantization.schema import QuantParamSchema

logger = init_logger(__name__)

# use system-level temp directory for file locks, so that multiple users
# can share the same lock without error.
# lock files in the temp directory will be automatically deleted when the
# system reboots, so users will not complain about annoying lock files
temp_dir = tempfile.gettempdir()


def enable_hf_transfer():
    """automatically activates hf_transfer
    """
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            # enable hf hub transfer if available
            import hf_transfer  # type: ignore # noqa
            huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
        except ImportError:
            pass


enable_hf_transfer()


class DisabledTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir or temp_dir
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name),
                             mode=0o666)
    return lock


def _shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def convert_bin_to_safetensor_file(
    pt_filename: str,
    sf_filename: str,
) -> None:
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = _shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})

    # check file size
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """)

    # check if the tensors are the same
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


# TODO(woosuk): Move this to other place.
def get_quant_config(model_config: ModelConfig,
                     load_config: LoadConfig) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)
    # Read the quantization config from the HF model config, if available.
    hf_quant_config = getattr(model_config.hf_config, "quantization_config",
                              None)
    if hf_quant_config is not None:
        return quant_cls.from_config(hf_quant_config)
    model_name_or_path = model_config.model
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_name_or_path, load_config.download_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    possible_config_filenames = quant_cls.get_config_filenames()

    # If the quantization config is not found, use the default config.
    if not possible_config_filenames:
        return quant_cls()

    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(
            f.endswith(x) for x in possible_config_filenames)
    ]
    if len(quant_config_files) == 0:
        raise ValueError(
            f"Cannot find the config file for {model_config.quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}")

    quant_config_file = quant_config_files[0]
    with open(quant_config_file, "r") as f:
        config = json.load(f)
    return quant_cls.from_config(config)


def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: Optional[str],
    allow_patterns: List[str],
    revision: Optional[str] = None,
) -> str:
    """Download model weights from Hugging Face Hub.

    Args:
        model_name_or_path (str): The model name or path.
        cache_dir (Optional[str]): The cache directory to store the model
            weights. If None, will use HF defaults.
        allow_patterns (List[str]): The allowed patterns for the
            weight files. Files matched by any of the patterns will be
            downloaded.
        revision (Optional[str]): The revision of the model.

    Returns:
        str: The path to the downloaded model weights.
    """
    if not huggingface_hub.constants.HF_HUB_OFFLINE:
        # Before we download we look at that is available:
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

        # depending on what is available we download different things
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break

    logger.info("Using model weights format %s", allow_patterns)
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        hf_folder = snapshot_download(
            model_name_or_path,
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
            tqdm_class=DisabledTqdm,
            revision=revision,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )
    return hf_folder


def filter_files_not_needed_for_inference(
        hf_weights_files: List[str]) -> List[str]:
    """
    Exclude files that are not needed for inference.

    See https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
    """
    blacklist = [
        "training_args.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    ]
    hf_weights_files = [
        f for f in hf_weights_files
        if not any(f.endswith(x) for x in blacklist)
    ]
    return hf_weights_files


def np_cache_weights_iterator(
    model_name_or_path: str, cache_dir: Optional[str], hf_folder: str,
    hf_weights_files: List[str]
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model np files.

    Will dump the model weights to numpy files if they are not already dumped.
    """
    # Convert the model weights from torch tensors to numpy arrays for
    # faster loading.
    np_folder = os.path.join(hf_folder, "np")
    os.makedirs(np_folder, exist_ok=True)
    weight_names_file = os.path.join(np_folder, "weight_names.json")
    # Use file lock to prevent multiple processes from
    # dumping the same model weights to numpy at the same time.
    with get_lock(model_name_or_path, cache_dir):
        if not os.path.exists(weight_names_file):
            weight_names = []
            for bin_file in hf_weights_files:
                state = torch.load(bin_file, map_location="cpu")
                for name, param in state.items():
                    param_path = os.path.join(np_folder, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.cpu().detach().numpy())
                    weight_names.append(name)
            with open(weight_names_file, "w") as f:
                json.dump(weight_names, f)

    with open(weight_names_file, "r") as f:
        weight_names = json.load(f)

    for name in weight_names:
        param_path = os.path.join(np_folder, name)
        with open(param_path, "rb") as f:
            param = np.load(f)
        yield name, torch.from_numpy(param)

def getTorchType(dtype_str : str) -> (torch.dtype, int):
    match dtype_str:
        case 'F16' :
            return (torch.float16, 2)
        case 'F32' :
            return (torch.float32, 4)
        case 'F64' :
            return (torch.float64, 8)
        case 'BF16' :
            return (torch.bfloat16, 2)
        case _ :
            print("Error converting dtype {}, use BF16".format(dtype_str))
            return (torch.bfloat16, 2)

model_id = ''
server_ip = ''
pp_rank = -1
pp_size = -1
shm = None

def safetensors_weights_iterator(
    hf_weights_files: List[str]
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""

    enable_para = int(os.getenv('ENABLE_PARA', '1'))
    use_local_server = int(os.getenv('LOCAL_SERVER', '0'))
    if enable_para == 1:
        stime = time.time()

        while True:
            state_dict = torch.ops.download_model.get_state_dict()
            name = state_dict[0]
            param = state_dict[1]
            if len(name) == 1:
                if name == 'e':
                    # empty
                    time.sleep(0.01)
                    continue
                elif name == 's':
                    # stop
                    break
                else:
                    print(f"name = {name}")
                    raise ValueError("state_dict name error")
            else:
                yield name, param

        etime = time.time()
        print(f"initialize model: model download & dispatch elapsed {etime - stime} seconds")
    elif use_local_server == 1:
        stime = time.time()

        global model_id
        global server_ip
        global pp_rank
        global pp_size
        global shm

        if pp_size == -1:
            model_id = os.getenv('REMOTE_MODEL_ID')
            server_ip = os.getenv('STORAGE_IP')
            pp_rank = int(os.getenv('PP_RANK', ''))
            pp_size = int(os.getenv('PP_SIZE', ''))
            is_dest = False
        else:
            # the second load model request must target dest model
            is_dest = True
        
        fast_coldstart = int(os.getenv("FAST_COLDSTART", "1"))

        if fast_coldstart == 1:
            # directly obtain state dict from libtorch (we have started the loading process in app.py)
            while True:
                name, param = torch.ops.download_model.get_state_dict_shm()
                if len(name) == 1:
                    if name == 'e':
                        # empty
                        time.sleep(0.01)
                        continue
                    elif name == 's':
                        # stop
                        break
                    else:
                        print(f"name = {name}")
                        raise ValueError("state_dict name error")
                else:
                    yield name, param
        else:
            client = socket.socket(socket.AF_INET)
            client.connect((server_ip, 6666))

            req_header = {"model": model_id, "pp_rank": pp_rank, "pp_size": pp_size, "is_dest": is_dest, "pre_load": "no"}
            req_header_bytes = json.dumps(req_header).encode('utf-8')
            req_header_size = len(req_header_bytes)
            req_header_size_bytes = req_header_size.to_bytes(length=8, byteorder='little', signed=False)

            client.send(req_header_size_bytes + req_header_bytes)

            shm_addr_bytes = client.recv(8)
            shm_addr = int.from_bytes(shm_addr_bytes, byteorder='little', signed=True)
            shm_size_bytes = client.recv(8)
            shm_size = int.from_bytes(shm_size_bytes, byteorder='little', signed=True)
            shm_name = client.recv(50).decode('utf-8')
            shm = shared_memory.SharedMemory(shm_name)

            def read_from_shm(begin: int, end: int):
                while True:
                    cur_len = int.from_bytes(shm.buf[shm_addr:shm_addr+8], byteorder='little', signed=True)
                    if cur_len >= end:
                        break
                return shm.buf[shm_addr+begin:shm_addr+end].tobytes()

            header_size_bytes = read_from_shm(8, 16)
            header_size = int.from_bytes(header_size_bytes, byteorder='little', signed=False)
            header_data = read_from_shm(16, 16 + header_size)
            header = json.loads(header_data.decode("UTF-8"))

            print(f"load model: read header elapsed {time.time() - stime} seconds")

            # receive all tensors
            data_begin_pos = 8 + 8 + header_size
            for i, (tensor_name, info) in enumerate(header.items()):
                start_pos = info["data_offsets"][0]
                end_pos = info["data_offsets"][1]
                data = read_from_shm(data_begin_pos + start_pos, data_begin_pos + end_pos)
                dtype, sz = getTorchType(info["dtype"])
                tensor = torch.frombuffer(data, dtype=dtype, count=(end_pos - start_pos) // sz)
                tensor = torch.reshape(tensor, info["shape"])
                yield tensor_name, tensor
            
            shm.close()

            client.send(b'ok')
            client.close()
            
            print(f"load model: download model elapsed {time.time() - stime} seconds")
    else:
        stime = time.time()
        model_path = os.getenv("MODEL_PATH")
        with safe_open(model_path, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                param = f.get_tensor(name)
                yield name, param
        print(f"load model: download model elapsed {time.time() - stime} seconds")

def pt_weights_iterator(
    hf_weights_files: List[str]
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model bin/pt files."""
    for bin_file in hf_weights_files:
        state = torch.load(bin_file, map_location="cpu")
        for name, param in state.items():
            yield name, param
        del state
        torch.cuda.empty_cache()


def kv_cache_scales_loader(
        filename: str, tp_rank: int, tp_size: int, num_hidden_layers: int,
        model_type: Optional[str]) -> Iterable[Tuple[int, float]]:
    """
    A simple utility to read in KV cache scaling factors that have been
    previously serialized to disk. Used by the model to populate the appropriate
    KV cache scaling factors. The serialization should represent a dictionary
    whose keys are the TP ranks and values are another dictionary mapping layers
    to their KV cache scaling factors.
    Keep this function in sync with the output of examples/fp8/extract_scales.py
    """
    try:
        with open(filename) as f:
            context = {
                "model_type": model_type,
                "num_hidden_layers": num_hidden_layers,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
            }
            schema_dct = json.load(f)
            schema = QuantParamSchema.model_validate(schema_dct,
                                                     context=context)
            layer_scales_map = schema.kv_cache.scaling_factor[tp_rank]
            return layer_scales_map.items()

    except FileNotFoundError:
        logger.error("File or directory '%s' not found.", filename)
    except json.JSONDecodeError:
        logger.error("Error decoding JSON in file '%s'.", filename)
    except Exception as e:
        logger.error("An error occurred while reading '%s': %s", filename, e)
    # This section is reached if and only if any of the excepts are hit
    # Return an empty iterable (list) => no KV cache scales are loaded
    # which ultimately defaults to 1.0 scales
    logger.warning(
        "Defaulting to KV cache scaling factors = 1.0 for all "
        "layers in TP rank %d as an error occurred during loading.", tp_rank)
    return []


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:
    """convert PySafeSlice object from safetensors to torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    """
    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x


def default_weight_loader(param: torch.Tensor,
                          loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    assert param.size() == loaded_weight.size()
    param.data = loaded_weight


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
) -> None:
    """Initialize model weights with random values.

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.
    """
    for param in model.state_dict().values():
        if torch.is_floating_point(param):
            param.data.uniform_(low, high)
