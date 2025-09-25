import os
import re
import sys
import time
import torch
import json
import traceback
import ctypes
import math
import socketserver
from modelscope.hub.snapshot_download import snapshot_download
from modelscope import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import vllm
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.usage.usage_lib import UsageContext

serverless_llm = 1 if os.getenv('CACHE_TYPE', 'vllm') == 'serverlessllm' else 0
serverless_pilot = int(os.getenv("SERVERLESS_PILOT", "0"))
chunk_size = 32 * 1024 * 1024      # 32MB

def transferTorchType(dtype) -> str:
    if dtype == torch.float16 or dtype == torch.half:
        return "F16"
    if dtype == torch.float32 or dtype == torch.float:
        return "F32"
    if dtype == torch.float64:
        return "F64"
    if dtype == torch.bfloat16:
        return "BF16"
    print(f"Error: converting dtype {dtype}, use F16")
    return "F16"

def getType(key : str, tie_word_embeddings: bool):
    """
    Get download type for a layer name to ensure equally layer partition
    0: embed tokens
    1: decoder layer 0
    2: lm_head, etc.
    3: none
    """
    if "lm_head" in key:
        # final lm head
        if tie_word_embeddings:
            return 3
        return 2
    if "rotary" in key:
        return 3
    if "emb" in key or "wte" in key:
        # embedding
        # Check if lm_head is a copy of embed_token
        if tie_word_embeddings:
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

def get_layer_distribution(tot_layer, pp_size, state_dict, tie_word_embeddings):
    layers = []
    embed_weights = 0
    decoder_weights = 0
    final_weights = 0
    for key, value in state_dict.items():
        tensor_type_list = getType(key, tie_word_embeddings)
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

def transform(key : str, layers : list[int], tie_word_embeddings: bool):
    """
    Get host id for a layer name.
    """
    num_hosts = len(layers)
    if "lm_head" in key:
        # final lm head
        if tie_word_embeddings:
            return [], key
        return num_hosts - 1, key
    if "rotary" in key:
        return range(0, num_hosts), key
    if "emb" in key or "wte" in key:
        # embedding
        # Check if lm_head is a copy of embed_token
        if tie_word_embeddings:
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

class Model:
    def __init__(self, model, model_path, max_pp_size = 4):
        self.state_dict_org = model.state_dict()
        self.state_dict = self.transfer_state_dict()
        # for each pp_size, partition format is dict{pp_rank: state_dict]}
        self.partition = {}
        self.duplicate_state_dict = {}
        self.get_partition(model, model_path, max_pp_size)
        del self.state_dict_org
    
    def transfer_state_dict(self):
        # transfer state_dict to bytes
        state_dict = {}
        for key, tensor in self.state_dict_org.items():
            tensor_size = tensor.numel() * tensor.element_size()
            tensor_dtype = transferTorchType(tensor.dtype)
            tensor_shape = tensor.shape
            byte_string = ctypes.string_at(tensor.data_ptr(), tensor_size)
            state_dict[key] = (tensor_dtype, tensor_shape, byte_string)
        return state_dict
    
    def get_partition(self, model, model_path, max_pp_size):
        tie_word_embeddings = model.config.tie_word_embeddings
        for pp_size in range(1, max_pp_size + 1):
            cur_num_layers = 0
            layers = []
            for pp_rank in range(pp_size):
                self.partition[(pp_size, pp_rank)] = {}
                self.duplicate_state_dict[(pp_size, pp_rank)] = {}
                model_path_ = model_path if pp_size == 1 else model_path + "-" + str(pp_rank) + "-" + str(pp_size)
                config = AutoConfig.from_pretrained(model_path_, trust_remote_code=True)
                if hasattr(config, "num_hidden_layers"):
                    cur_num_layers += config.num_hidden_layers
                elif hasattr(config, "num_layers"):
                    cur_num_layers += config.num_layers
                else:
                    raise ValueError("Cannot obtain layer num.")
                layers.append(cur_num_layers)
            
            # get layer distribution
            # layers = get_layer_distribution(tot_layer, pp_size, self.state_dict_org, tie_word_embeddings)
            
            for key, value in self.state_dict.items():
                pp_ranks, new_key = transform(key, layers, tie_word_embeddings)
                if isinstance(pp_ranks, int):
                    pp_ranks = [pp_ranks]
                for pp_rank in pp_ranks:
                    self.partition[(pp_size, pp_rank)][new_key] = value
                    self.duplicate_state_dict[(pp_size, pp_rank)][new_key] = key

    def get_state_dict(self, pp_rank, pp_size, is_dest):
        if not is_dest:
            return self.partition[(pp_size, pp_rank)]
        state_dict = {}
        for key, value in self.state_dict.items():
            state_dict[key] = value
        for new_key, origin_key in self.duplicate_state_dict[(pp_size, pp_rank)].items():
            del state_dict[origin_key]
        return state_dict

num_threads = 4
chunk_size = 32 * 1024 * 1024 #32MB

class SllmModel:
    def __init__(self, model_dir):
        tensor_files = os.listdir(model_dir)
        # print(f"model_dir = {model_dir}, tensor_files = {tensor_files}")
        contents = bytes()
        for file in tensor_files:
            if file.startswith('tensor.data'):
                file_path = os.path.join(model_dir, file)
                with open(file_path, "rb") as f:
                    content = f.read()
                    contents = contents + content
        
        num_bytes = len(contents)
        print(f"Model {model_dir} get tensor size: {num_bytes}")

        num_chunks = math.ceil(num_bytes / chunk_size)
        chunk_per_thread = math.ceil(num_chunks / num_threads)

        print(f"Model num_chunks = {num_chunks}, chunk_per_thread = {chunk_per_thread}")

        self.thread_content = []
        for thread_id in range(num_threads):
            start_chunk_id = thread_id * chunk_per_thread
            end_chunk_id = min((thread_id + 1) * chunk_per_thread, num_chunks)
            start_offset = start_chunk_id * chunk_size
            end_offset = min(end_chunk_id * chunk_size, num_bytes)
            self.thread_content.append(contents[start_offset:end_offset])
    
    def send_state_dict(self, socket, thread_id, num_thread):
        if num_thread != num_threads:
            print(f"Error: use {num_thread} threads, but the initialized one is {num_threads}")
        socket.sendall(self.thread_content[thread_id])

model_set = int(os.getenv("MODEL_SET", "3"))

ModelSet = {}

ModelSet[0] = [
    "modelscope/Llama-2-7b-chat-ms",
    "LLM-Research/Meta-Llama-3-8B-Instruct",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "AI-ModelScope/falcon-7b",
]

ModelSet[1] = [
    "modelscope/Llama-2-13b-chat-ms",
    "facebook/opt-13b",
]

ModelSet[3] = [
    "modelscope/Llama-2-7b-chat-ms",
    "modelscope/Llama-2-13b-chat-ms",
]

model_list = ModelSet[model_set]

if model_set != 3:
    for additional_model in ModelSet[3]:
        if additional_model not in model_list:
            model_list.append(additional_model)

# model_list = [
#     "modelscope/Llama-2-7b-chat-ms",
#     "modelscope/Llama-2-13b-chat-ms",
# ]

# For Serverless Pilot
if serverless_pilot == 1:
    model_list = [
        "ZhipuAI/chatglm2-6b",
    ]

models = {}

def send_state_dict(state_dict, socket, thread_id, num_thread, is_query):
    stime = time.time()
    header = {}
    cur_pos = 0
    sorted_tensors = sorted(state_dict.items())
    for tensor_name, tensor_info in sorted_tensors:
        info = {}
        tensor_size = len(tensor_info[2])
        info["data_offsets"] = (cur_pos, cur_pos + tensor_size)
        cur_pos += tensor_size
        info["dtype"] = tensor_info[0]
        info["shape"] = tensor_info[1]
        header[tensor_name] = info
    
    header_bytes = json.dumps(header).encode('utf-8')
    header_length = len(header_bytes)
    safetensors_bytes_length = 8 + header_length + cur_pos

    if is_query:
        socket.sendall(safetensors_bytes_length.to_bytes(length=8, byteorder='little', signed=False))
        elapsed = time.time() - stime
        print(f"send_query time cost = {elapsed} seconds")
        return

    if thread_id == 0:
        # send header
        pkt_head_bytes = header_length.to_bytes(length=8, byteorder='little', signed=False) + header_bytes 
        socket.sendall(pkt_head_bytes)
    
    bytes_per_thread = safetensors_bytes_length // num_thread
    num_bytes_start = thread_id * bytes_per_thread
    num_bytes_end = (thread_id + 1) * bytes_per_thread
    if thread_id == num_thread - 1:
        num_bytes_end += safetensors_bytes_length % num_thread
    cur_pos = 8 + header_length
    for tensor_name, tensor_info in sorted_tensors:
        tensor_len = len(tensor_info[2])
        if cur_pos + tensor_len <= num_bytes_start:
            cur_pos += tensor_len
            continue
        tensor_start_pos = max(num_bytes_start - cur_pos, 0)
        tensor_end_pos = min(num_bytes_end - cur_pos, tensor_len)
        socket.sendall(tensor_info[2][tensor_start_pos:tensor_end_pos])
        cur_pos += tensor_len
        if cur_pos >= num_bytes_end:
            break

    size = (num_bytes_end - num_bytes_start) / 1024 / 1024 / 1024
    elapsed = time.time() - stime
    print(f"send_state_dict [{thread_id}-{num_thread}] time cost = {elapsed} seconds, speed = {size / elapsed} GB/s")

class MyTCPHandler(socketserver.BaseRequestHandler):
    """
    The request handler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.
    """

    def handle(self):
        if serverless_llm == 1:
            thread_idx_bytes = self.request.recv(4)
            thread_idx = int.from_bytes(thread_idx_bytes, byteorder='little', signed=True)
            if thread_idx == -1:
                # initialization query
                self.request.sendall(bytes('initialization finished', 'utf-8'))

                # sleep 1 second to wait for peer read data
                time.sleep(1)
                self.request.close()
                return
            num_thread_bytes = self.request.recv(4)
            num_thread = int.from_bytes(num_thread_bytes, byteorder='little', signed=True)
            model_path = self.request.recv(100).decode('utf-8')
            # Check whether model_path ends with /0, /1, etc. If true, get rid of them
            pattern = re.compile(r'/\d+')
            model_path = re.sub(pattern, '', model_path)
            print(f"Received request for model {model_path} [{thread_idx}-{num_thread}]")

            if model_path not in models:
                print(f"Error: we only have models {models.keys()}")
                self.request.close()
                return

            stime = time.time()
            models[model_path].send_state_dict(self.request, thread_idx, num_thread)

            print(f"Model {model_path}[{thread_idx}-{num_thread}] send_state_dict time cost = {time.time() - stime} seconds")
            # sleep 10 second to wait for peer read data
            time.sleep(10)
            self.request.close()
            return

        req_header_size_bytes = self.request.recv(8)
        req_header_size = int.from_bytes(req_header_size_bytes, byteorder='little', signed=False)
        if req_header_size == 0:
            # initialization query
            self.request.sendall(bytes('initialization finished', 'utf-8'))

            # sleep 1 second to wait for peer read data
            time.sleep(1)
            self.request.close()
            return
        
        req_header_bytes = self.request.recv(req_header_size)
        req_header = json.loads(req_header_bytes.decode("UTF-8"))
        model_id = req_header["model"]
        pp_rank = int(req_header["pp_rank"])
        pp_size = int(req_header["pp_size"])
        is_dest = bool(req_header["is_dest"])
        thread_id = int(req_header["thread_id"]) if "thread_id" in req_header else 0
        num_thread = int(req_header["num_thread"]) if "num_thread" in req_header else 1
        is_query = bool(req_header["is_query"]) if "is_query" in req_header else False
        if is_query:
            print(f"Received query request for {model_id}-{pp_rank}-{pp_size} [{thread_id}-{num_thread}], is_dest = {is_dest}")
        else:
            print(f"Received model request for {model_id}-{pp_rank}-{pp_size} [{thread_id}-{num_thread}], is_dest = {is_dest}")

        if serverless_pilot == 0:
            # Get rid of instance type id
            origin_model_id = model_id
            pos = model_id.rfind('/')
            model_id = model_id[:pos]

        state_dict = models[model_id].get_state_dict(pp_rank, pp_size, is_dest)
        send_state_dict(state_dict, self.request, thread_id, num_thread, is_query)

        if is_query:
            print(f"Finished query request for {origin_model_id}-{pp_rank}-{pp_size} [{thread_id}-{num_thread}], is_dest = {is_dest}")
        else:
            print(f"Finished model request for {origin_model_id}-{pp_rank}-{pp_size} [{thread_id}-{num_thread}], is_dest = {is_dest}")
        
        # sleep 1 second to wait for peer read data
        time.sleep(1)
        self.request.close()

if __name__ == "__main__":
    cache_dir = os.getenv('MODELSCOPE_CACHE', '')
    if serverless_llm == 0:
        # Initialize models
        os.environ["CPU_ENV"] = "1"
        os.environ["ENABLE_PARA"] = "0"
        for model_name in model_list:
            print(f"Start init Model {model_name}...")
            stime = time.time()
            model_path = os.path.join(cache_dir, model_name.replace('.', '___'))
            os.environ["MODEL_PATH"] = os.path.join(model_path, "model.safetensors")
            engine_args = EngineArgs(
                model=model_path,
                enforce_eager=True,
                trust_remote_code=True,
                dtype="float16"
            )
            engine = LLMEngine.from_engine_args(
                engine_args, usage_context=UsageContext.OPENAI_API_SERVER)
            model = engine.model_executor.driver_worker.model_runner.model
            if serverless_pilot == 1:
                models[model_name] = Model(model, model_path, 6)
            else:
                models[model_name] = Model(model, model_path)
            print(f"Model {model_name} inited, time cost = {time.time() - stime} seconds.")
        print(f"All models have initialized.")
    else:
        for model_name in model_list:
            model_path = os.path.join(cache_dir, model_name)
            rank_dir = os.path.join(model_path, "rank_0")
            models[os.path.join("vllm", model_name, "rank_0")] = SllmModel(rank_dir)
        
        print(f"All models have initialized.")
    
    # Start TCP Server
    with socketserver.ForkingTCPServer(('0.0.0.0', 8888), MyTCPHandler) as server:
        server.serve_forever()

