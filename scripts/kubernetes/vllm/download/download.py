import os 
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download
from modelscope import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import re
import torch
import sys
import shutil
import psutil
import gc
import json
import time
from safetensors.torch import save_file

def get_current_memory_gb():
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024. / 1024.

def get_size(sz):
    mul = 1
    for dim in sz:
        mul = mul * dim
    return mul

def getType(key : str, _tied_weights_keys: list[str]):
    """
    Get download type for a layer name to ensure equally layer partition
    0: embed tokens
    1: decoder layer 0
    2: lm_head, etc.
    3: none
    """
    if "lm_head" in key:
        # final lm head
        if _tied_weights_keys is not None:
            for tied_key in _tied_weights_keys:
                if "lm_head" in tied_key:
                    return 3
        return 2
    if "rotary" in key:
        return 3
    if "emb" in key or "wte" in key:
        # embedding
        # Check if lm_head is a copy of embed_token
        if _tied_weights_keys is not None:
            for tied_key in _tied_weights_keys:
                if "lm_head" in tied_key:
                    return [0, 2]
        return 0
    pattern = re.compile(r'\d+')
    substrs = pattern.findall(key)
    if len(substrs) > 0:
        # decoding layers
        id = int(substrs[0])    # get layer id
        if id == 0:
            return 1
        return 3
    else:
        # final layer norm
        return 2

def get_layer_distribution(tot_layer, pp_size, state_dict, _tied_weights_keys):
    layers = []
    embed_weights = 0
    decoder_weights = 0
    final_weights = 0
    for key, value in state_dict.items():
        tensor_type_list = getType(key, _tied_weights_keys)
        tensor_size = value.nelement() * value.element_size()
        if isinstance(tensor_type_list, int):
            tensor_type_list = [tensor_type_list]
        for tensor_type in tensor_type_list:
            if tensor_type == 0:
                embed_weights += tensor_size
            elif tensor_type == 1:
                decoder_weights += tensor_size
            elif tensor_type == 2:
                final_weights += tensor_size
    if embed_weights == 0 and final_weights == 0:
        # equally distribution
        num_layers_per_node = int(tot_layer / pp_size)
        sum = 0
        for i in range(pp_size):
            sum += num_layers_per_node 
            layers.append(sum)
        if sum < tot_layer:
            for i in range(0, tot_layer - sum):
                layers[-i-1] += 1
    else:
        # consider embedding and lm_head
        layers = [0] * pp_size
        weights = [0] * pp_size
        weights[0] = embed_weights
        weights[-1] = final_weights
        for i in range(tot_layer):
            # find the stage that holds minimum weights
            mn_id = 0
            mn_value = weights[0]
            for j in range(1, pp_size):
                if weights[j] <= mn_value:
                    mn_value = weights[j]
                    mn_id = j
            weights[mn_id] += decoder_weights
            layers[mn_id] += 1
        for i in range(1, pp_size):
            layers[i] += layers[i-1]
    return layers

def transform(key : str, layers : list[int], _tied_weights_keys: list[str]):
    """
    Get host id for a layer name.
    """
    num_hosts = len(layers)
    if "lm_head" in key:
        # final lm head
        if _tied_weights_keys is not None:
            for tied_key in _tied_weights_keys:
                if "lm_head" in tied_key:
                    return [], key
        return num_hosts - 1, key
    if "rotary" in key:
        return range(0, num_hosts), key
    if "emb" in key or "wte" in key:
        # embedding
        # Check if lm_head is a copy of embed_token
        if _tied_weights_keys is not None:
            for tied_key in _tied_weights_keys:
                if "lm_head" in tied_key:
                    return [0, num_hosts - 1], key
        return 0, key
    pattern = re.compile(r'\d+')
    substrs = pattern.findall(key)
    if len(substrs) > 0:
        # decoding layers
        id = int(substrs[0])    # get layer id
        last_sum = 0
        for i, sum in enumerate(layers):
            if sum > id:
                key = key.replace(str(id), str(int(id - last_sum)), 1)
                return i, key
            last_sum = sum
        raise ValueError("model layer id is too large")
    else:
        # final layer norm
        return num_hosts - 1, key

    raise ValueError("do not recognize key")

def partition(model_path: str, pp_size: int, save_partitioned_state_dict: int):
    if model_path.endswith("/"):
        model_path = model_path[:-1]

    print(f"partition: model_path = {model_path}, pp_size = {pp_size}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype = config.torch_dtype,
        trust_remote_code = True
    )
    print(f"torch dtype = {model.config.torch_dtype}")

    if pp_size == 1:
        config.save_pretrained(model_path)
        # delete large files
        items = os.listdir(model_path)
        for item in items:
            if item.endswith(".bin") or item.endswith(".pth") or item.endswith(".pt") or item.endswith(".gguf") or item.endswith(".safetensors") or item.endswith(".msgpack") or item.endswith(".h5") or item.endswith(".index.json"):
                os.remove(model_path + "/" + item)
        # save state_dict as a single file
        state_dict = model.state_dict()
        if config.tie_word_embeddings and hasattr(model, "_tied_weights_keys"):
            _tied_weights_keys = model._tied_weights_keys
            for key in _tied_weights_keys:
                if key in state_dict.keys():
                    print(f"Delete tied_key {key} from state_dict")
                    del state_dict[key]
        save_file(state_dict, model_path + "/model.safetensors", metadata = {'format': 'pt'})
        return

    if hasattr(model.config, "num_hidden_layers"):
        tot_layer = model.config.num_hidden_layers
    elif hasattr(model.config, "num_layers"):
        tot_layer = model.config.num_layers
    else:
        raise ValueError("Cannot obtain layer num.")

    # Obtain layer count for each pp stage
    state_dict = model.state_dict()
    _tied_weights_keys = None
    if config.tie_word_embeddings and hasattr(model, "_tied_weights_keys"):
        _tied_weights_keys = model._tied_weights_keys
    layers = get_layer_distribution(tot_layer, pp_size, state_dict, _tied_weights_keys)       # prefix sum of number of decoder layers of each pp stage
    dest_layers = []
    for dest_pp_size in range(1, pp_size):
        dest_layers.append(get_layer_distribution(tot_layer, dest_pp_size, state_dict, _tied_weights_keys))       # prefix sum of number of decoder layers of each pp stage
    state_dict = None
    _tied_weights_keys = None

    print(f"Layer number = {tot_layer}, layers: ")
    for num in layers:
        print(num)

    items = os.listdir(model_path)
    items_needed = []
    for item in items:
        if item.endswith(".safetensors"):
            continue
        if item.endswith(".bin") or item.endswith(".pth") or item.endswith(".pt") or item.endswith(".gguf") or item.endswith(".msgpack") or item.endswith(".h5"):
            continue
        if os.path.isfile(model_path + "/" + item):
            items_needed.append(item)

    for id in range(0, pp_size):
        model_path_ = model_path + "-" + str(id) + "-" + str(pp_size)
        # copy all needed files
        os.makedirs(model_path_, exist_ok=True)
        for file_name in items_needed:
            shutil.copy(model_path + "/" + file_name, model_path_)
        print(f"file copied to {model_path_}!")
        # save config
        num_layers = layers[id] - (0 if id == 0 else layers[id-1])
        config_ = config
        if hasattr(config_, "num_hidden_layers"):
            config_.num_hidden_layers = num_layers
        elif hasattr(config_, "num_layers"):
            config_.num_layers = num_layers
        config_.save_pretrained(model_path_)
        # save dest info
        for dest_pp_size in range(1, pp_size):
            for dest_pp_rank in range(dest_pp_size):
                pp_tail = "-" + str(dest_pp_rank) + "-" + str(dest_pp_size)
                dest_num_hidden_layers = dest_layers[dest_pp_size-1][dest_pp_rank] - (0 if dest_pp_rank == 0 else dest_layers[dest_pp_size-1][dest_pp_rank-1])
                with open(model_path_+ "/dest_info" + pp_tail, "w") as f:
                    f.write(str(dest_num_hidden_layers))

    # save parameters
    for id in range(0, pp_size):
        model_path_ = model_path + "-" + str(id) + "-" + str(pp_size)
        # obtain split state dict
        _tied_weights_keys = None
        if config.tie_word_embeddings and hasattr(model, "_tied_weights_keys"):
            _tied_weights_keys = model._tied_weights_keys
        split_state_dict = {}
        state_dict = model.state_dict()
        duplicate_state_dict = [[{} for _ in range(dest_pp_size)] for dest_pp_size in range(1, pp_size)]
        dest_split_state_dict = [[{} for _ in range(dest_pp_size)] for dest_pp_size in range(1, pp_size)]
        for key, value in state_dict.items():
            # get normal split state dict
            new_id, new_key = transform(key, layers, _tied_weights_keys)
            if isinstance(new_id, int):
                new_id = [new_id]
            split_state_dict_key = None
            for id_ in new_id:
                if id_ == id:
                    print("{}: {}, use new key = {}".format(new_id, key, new_key))
                    split_state_dict[new_key] = value
                    split_state_dict_key = new_key
                    break
            for dest_pp_size in range(1, pp_size):
                # get dest split state dict for each possible dest_pp_rank
                dest_new_id, dest_new_key = transform(key, dest_layers[dest_pp_size-1], _tied_weights_keys)
                for dest_pp_rank in range(dest_pp_size):
                    if isinstance(dest_new_id, int):
                        dest_new_id = [dest_new_id]
                    for id_ in dest_new_id:
                        if id_ == dest_pp_rank:
                            print("For dest_pp_rank [{}] - {}: {}, use new key = {}".format(dest_pp_rank, dest_new_id, key, dest_new_key))
                            if split_state_dict_key is not None:
                                duplicate_state_dict[dest_pp_size-1][dest_pp_rank][split_state_dict_key] = dest_new_key
                            else:
                                dest_split_state_dict[dest_pp_size-1][dest_pp_rank][dest_new_key] = value
                            break
        
        if save_partitioned_state_dict == 1:
            # save split_state_dict
            save_file(split_state_dict, model_path_ + "/model.safetensors", metadata = {'format': 'pt'})
            print(f"model saved to {model_path_}!")

        # duplicate_state_dict for dest model
        for dest_pp_size in range(1, pp_size):
            for dest_pp_rank in range(dest_pp_size):
                pp_tail = "-" + str(dest_pp_rank) + "-" + str(dest_pp_size)
                if save_partitioned_state_dict == 1:
                    save_file(dest_split_state_dict[dest_pp_size-1][dest_pp_rank], model_path_ + "/dest_model" + pp_tail + ".safetensors", metadata = {'format': 'pt'})
                with open(model_path_ + "/duplicate_state_dict" + pp_tail + ".json", "w") as f:
                    json.dump(duplicate_state_dict[dest_pp_size-1][dest_pp_rank], f)
                print(f"dest model [{dest_pp_rank}-{dest_pp_size}]'s duplicate_state_dict saved to {model_path_}!")

def download_model():
    model_id = os.getenv('MODEL_ID', '')
    revision = os.getenv('MODEL_VERSION', '')
    cache_dir = os.getenv('MODELSCOPE_CACHE', '')
    sdk_token = os.getenv('MODELSCOPE_TOKEN', '')
    max_pp_size = int(os.getenv('MAX_PP_SIZE', '1'))
    save_partitioned_state_dict = int(os.getenv('SAVE_PART', '0'))
    # download all possible model parts for max_pp_size

    if revision == "latest":
        revision = ""
    
    # login first.
    HubApi().login(sdk_token)
    print(f"cache_dir = {cache_dir}")
    if "Llama-3-8B" in model_id:
        # LLM-Research/Meta-Llama-3-8B-Instruct only has safetensors format
        ignore_patterns = ["*.pth"]
    else:
        ignore_patterns = ["*.safetensors", "*.pth", "*.pt", "*.msgpack", "*.h5", "*.safetensors.index.json"]
    if len(revision) > 0:
        model_dir = snapshot_download (model_id =model_id, 
                           revision = revision,
                           cache_dir = cache_dir,
                           ignore_patterns = ignore_patterns)
    else:
        model_dir = snapshot_download (model_id =model_id, 
                            cache_dir = cache_dir,
                            ignore_patterns = ignore_patterns)
    model_dir = os.path.join(cache_dir, model_id.replace('.', '___'))
    print("download model success, execute partition...")
    for pp_size in range(1, max_pp_size + 1):
        partition(model_dir, pp_size, save_partitioned_state_dict)

if __name__ == '__main__':
    download_model()