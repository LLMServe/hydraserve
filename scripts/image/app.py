import time
runtime_start_time = time.time()
print("runtime started")

import sys
import os
import asyncio
import re
import json
import threading
from multiprocessing import shared_memory

etime = time.time()
print(f"import initial libraries time cost = {etime - runtime_start_time} seconds")

import torch

cur_time = time.time()
print(f"import torch time cost = {cur_time - etime} seconds")

# Inform libtorch to load model in another thread
writer = None
async def start_load_model_shm(model_id: str, pp_rank: int, pp_size: int, is_dest: bool):
    stime = time.time()
    global writer
    server_ip = os.getenv('STORAGE_IP')
    req_header = {"model": model_id, "pp_rank": pp_rank, "pp_size": pp_size, "is_dest": is_dest, "pre_load": "no"}
    req_header_bytes = json.dumps(req_header).encode('utf-8')
    req_header_size = len(req_header_bytes)
    req_header_size_bytes = req_header_size.to_bytes(length=8, byteorder='little', signed=False)

    reader, writer = await asyncio.open_connection(server_ip, 6666)
    
    writer.write(req_header_size_bytes + req_header_bytes)
    await writer.drain()

    shm_addr_bytes = await reader.read(8)
    shm_addr = int.from_bytes(shm_addr_bytes, byteorder='little', signed=True)
    shm_size_bytes = await reader.read(8)
    shm_size = int.from_bytes(shm_size_bytes, byteorder='little', signed=True)
    shm_name_bytes = await reader.read(50)
    shm_name = shm_name_bytes.decode('utf-8')

    torch.ops.download_model.start_load_shm(shm_name, shm_addr, shm_size)
    print(f"start_load_model_shm: start_load_shm time cost = {time.time() - stime} seconds")

async def end_load_model_shm():
    stime = time.time()
    global writer
    try:
        writer.close()
    except Exception as e:
        print(f"Close writer meets exception: {e}")
    # await writer.wait_closed()
    print(f"end_load_model_shm: close connection time cost = {time.time() - stime} seconds")

engine = None
model_executor = None
model_path = None
load_thread = None

# get model-info from env 

FC_MODEL_CACHE_DIR = os.getenv('MODELSCOPE_CACHE')
pp_rank = int(os.getenv('PP_RANK', ''))
pp_size = int(os.getenv('PP_SIZE', ''))
dest_pp_size = int(os.getenv('DEST_PP_SIZE', ''))
model_id = os.getenv('MODEL_ID', '')
enable_para = int(os.getenv('ENABLE_PARA', '1'))        #  if enable_para==1, the C++ library will start to load model from 'model.safetensors' first
use_local_server = int(os.getenv('LOCAL_SERVER', '0'))  #  else if use_local_server==1, we load model from local storage server
no_pre_init = int(os.getenv("NO_PRE_INIT", "0"))        #    if no_pre_init==1, we should not do any initialization before post "/init"
fast_coldstart = int(os.getenv("FAST_COLDSTART", "1"))  #    if fast_coldstart==1, we disable profile run and use libtorch to load models 
gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95"))
model_id = model_id.replace('.', '___')
pp_tail = ""
dest_init_started = False
conn_inited = False
model_kwargs = {}
if pp_size > 1:
    pp_tail = '-' + str(pp_rank) + '-' + str(pp_size)
model_path = FC_MODEL_CACHE_DIR + '/' + model_id + pp_tail

chat_template = None
if "Llama-2" in model_id:
    chat_template = "chat_template/llama2.jinja"
elif "chatglm3" in model_id:
    chat_template = "chat_template/chatglm3.jinja"
elif "chatglm2" in model_id:
    chat_template = "chat_template/chatglm2.jinja"
elif "baichuan" in model_id:
    chat_template = "chat_template/baichuan.jinja"

if not os.path.exists(model_path):
    print(f"model path = {model_path}")
    raise ValueError('[ERROR] model not found in cache')

print(f"model_id = {model_id}")
print(f"pp_rank = {pp_rank}, pp_size = {pp_size}, dest_pp_size = {dest_pp_size}")

# set MODEL_PATH
os.environ["MODEL_PATH"] = os.path.join(model_path, "model.safetensors")

if enable_para == 1 or use_local_server == 1:
    LIB_DIR = "/vllm-workspace/libst_pybinding.so" 
    if not os.path.exists(LIB_DIR):
        raise RuntimeError(
            f"Could not find the C++ library libst_pybinding.so at {LIB_DIR}. "
            "Please build the C++ library first or put it at the right place."
        )
    torch.ops.load_library(LIB_DIR)
    if use_local_server == 1 and no_pre_init == 0 and fast_coldstart == 1:
        stime = time.time()
        remote_model_id = os.getenv('REMOTE_MODEL_ID')
        asyncio.run(start_load_model_shm(remote_model_id, pp_rank, pp_size, False))
        print(f"app.py: start load_model_shm time cost = {time.time() - stime} seconds")

end_time = time.time()
print(f" load library time cost = {end_time - cur_time} seconds")
start_time = end_time

from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Set, Dict, List
import fastapi
import uvicorn
from fastapi import Request, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import make_asgi_app
from starlette.routing import Mount

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.executor.gpu_executor import GPUExecutorAsync
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              ChatCompletionResponse,
                                              CompletionRequest, ErrorResponse)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.utils import socket_send

end_time = time.time()
print(f"import vllm libs completed, time cost = {end_time - start_time} seconds")

TIMEOUT_KEEP_ALIVE = 5  # seconds

openai_serving_chat = None
openai_serving_completion = None
logger = init_logger(__name__)

_running_tasks: Set[asyncio.Task[Any]] = set()


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    # if not engine_args.disable_log_stats:
    #     task = asyncio.create_task(_force_log())
    #     _running_tasks.add(task)
    #     task.add_done_callback(_running_tasks.remove)

    yield


app = fastapi.FastAPI(lifespan=lifespan)


# Add prometheus asgi middleware to route /metrics requests
route = Mount("/metrics", make_asgi_app())
# Workaround for 307 Redirect for /metrics
route.path_regex = re.compile('^/metrics(?P<path>.*)$')
app.routes.append(route)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    await openai_serving_chat.engine.check_health()
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    models = await openai_serving_chat.show_available_models()
    return JSONResponse(content=models.model_dump())


@app.get("/version")
async def show_version():
    ver = {"version": vllm.__version__}
    return JSONResponse(content=ver)

async def run_worker_loop():
    # the background loop is running and this request keeps the whole instance alive
    global model_executor

    if model_executor is not None:
        model_executor.start_running()
        model_executor.set_new_timer()
        while model_executor is not None and model_executor.has_requests_in_progress():
            await asyncio.sleep(0)
    response = ""
    elapsed_time = time.time() - runtime_start_time
    message = { "elapsed_time": elapsed_time }
    result = { "message": { "content":  response } }
    ret = { 'Message':  message,
        'Data': result,
        'Success': True,
        'Code': 200}
    return JSONResponse(ret)

@app.post("/utils/warmup")
async def do_warmup():
    if pp_rank == 0:
        raise RuntimeError("received warmup request for first stage")
    response = await run_worker_loop()
    return response

@app.post("/utils/stop_origin_serve")
async def stop_origin_serve():
    global model_executor
    global engine
    if engine is None:
        raise RuntimeError("Cannot stop origin serving engine as new engine has not initialized.")
    if model_executor is None:
        return JSONResponse({'Message': "Origin serving engine has already stopped.", 'Code': 500})
    model_executor.quit = True
    return "Origin serving engine stopped."

# For dest_pp_rank > 0: Server will soon return response.
# For dest_pp_rank = 0 and pp_rank = 0: Server will soon return response.
# For dest_pp_rank = 0 and pp_rank > 0: Server will return response util the new engine has inited. All requests after that point should go to this function. 
@app.post("/utils/dest")
async def start_load_dest(request: Dict):
    global load_thread
    global model_executor
    global engine
    global openai_serving_chat
    global openai_serving_completion
    global dest_init_started

    func_stime = time.time()

    print("Start to initialize dest model.")

    if dest_init_started:
        message = { "elapsed_time": 0 }
        return JSONResponse({'Message': message})

    dest_init_started = True

    if use_local_server == 1:
        # Start load dest model from local server
        remote_model_id = os.getenv('REMOTE_MODEL_ID')
        await start_load_model_shm(remote_model_id, pp_rank, pp_size, True)

    dest_pp_rank = int(request['dest_pp_rank'])
    ip_list = request['ip_list']
    origin_ranks = request['origin_ranks']

    print(f"start_load_dest: my dest_pp_rank = {dest_pp_rank}")

    with open(model_path + "/dest_info-" + str(dest_pp_rank) + "-" + str(dest_pp_size), "r") as f:
        res = f.read()
        res = res.split()
        dest_num_hidden_layers = int(res[0])

    os.environ["DEST_PP_RANK"] = str(dest_pp_rank)
    os.environ["DEST_NUM_LAYERS"] = str(dest_num_hidden_layers)
    dest_model_file_name = "dest_model-" + str(dest_pp_rank) + "-" + str(dest_pp_size) + ".safetensors"
    os.environ["DEST_MODEL_PATH"] = os.path.join(model_path, dest_model_file_name)

    # change environ pp_rank and pp_size to dest_pp_rank and dest_pp_size to initialize a new worker for dest model
    os.environ["PP_RANK"] = str(dest_pp_rank)
    os.environ["PP_SIZE"] = str(dest_pp_size)

    etime_1 = time.time()
    # wait for loading from shared memory finish
    custom_op = torch.ops.download_model.check_load_fin_shm
    while custom_op() != 1:
        await asyncio.sleep(0.1)
    print(f"start_load_dest: wait for loading from shm finish time cost = {time.time() - etime_1} seconds")

    print("start_load_dest: init dest model worker started")
    if pp_rank > 0:
        model_executor.init_dest_model_worker(ip_list, origin_ranks)
        # load_thread = threading.Thread(target=model_executor.init_dest_model_worker, args=(ip_list,origin_ranks))
    else:
        engine.engine.model_executor.init_dest_model_worker(ip_list, origin_ranks)
        # load_thread = threading.Thread(target=engine.engine.model_executor.init_dest_model_worker, args=(ip_list,origin_ranks))
    # load_thread.start()

    if use_local_server == 1:
        await end_load_model_shm()

    if dest_pp_rank == 0:
        if pp_rank > 0:
            # Initalize a new LLMEngine
            if dest_pp_size > 1:
                raise ValueError("We do not support transfer from ranki to rank0 when dest_pp_size>1 because we do not know when to change invoke_socket")
            served_model_names = [model_id]
            engine_args = AsyncEngineArgs(
                model=model_path,
                enforce_eager=True,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            stime = time.time()
            engine = AsyncLLMEngine.from_engine_args(
                engine_args, usage_context=UsageContext.OPENAI_API_SERVER, model_executor=model_executor)
            time_1 = time.time()
            print(f"app.py: init engine time cost = {time_1 - stime} seconds")
            openai_serving_chat = OpenAIServingChat(engine, served_model_names, "assistant", chat_template=chat_template)
            time_2 = time.time()
            print(f"app.py: init serving chat time cost = {time_2 - time_1} seconds")
            openai_serving_completion = OpenAIServingCompletion(engine, served_model_names)
            time_3 = time.time()
            print(f"app.py: init serving completion time cost = {time_3 - time_2} seconds")
        
        # Init KV cache
        engine.engine.extend_gpu_blocks()
    engine.engine.dest_inited = True
            
    cur_time = time.time()
    print(f"app.py: load_dest time cost = {cur_time - func_stime} seconds")
    elapsed_time = cur_time - func_stime
    message = { "elapsed_time": elapsed_time }
    return JSONResponse({'Message': message})

@app.post("/utils/migration")
async def start_migration(request: Dict):
    num_coldstart_reqs = int(request['num_coldstart_reqs'])
    if num_coldstart_reqs > 0:
        print("start to perform kv cache migration")

        engine.engine.start_request_migration = True
        stime = time.time()

        while not engine.engine.fin_request_migration:
            await asyncio.sleep(0.5)
        
        print(f"migrate request time cost = {(time.time() - stime) * 1000} ms")

    print(f"app.py: dest model inited!")

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    stime = time.time()

    global engine
    global model_executor
    global openai_serving_chat

    print(f"create_chat_completion: received a request for {request.model}")

    try:
        generator = await openai_serving_chat.create_chat_completion(
            request, raw_request)
    except Exception as e:
        logger.error("Error in create chat completion from request: %s", e)
        return openai_serving_chat.create_error_response(str(e))
    
    cur_time = time.time()
    print(f" generate time cost = {cur_time - stime} seconds")

    elapsed_time = cur_time - runtime_start_time
    print(f" total elapsed_time = {elapsed_time} seconds")

    if isinstance(generator, ErrorResponse):
        response = generator.model_dump()
        return JSONResponse(content=response,
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        assert isinstance(generator, ChatCompletionResponse)
        response = generator.model_dump()
        return JSONResponse(content=response)

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    stime = time.time()

    global engine
    global model_executor
    global openai_serving_completion

    generator = await openai_serving_completion.create_completion(
        request, raw_request)

    cur_time = time.time()
    print(f" generate time cost = {cur_time - stime} seconds")

    elapsed_time = cur_time - runtime_start_time
    print(f" total elapsed_time = {elapsed_time} seconds")

    if isinstance(generator, ErrorResponse):
        response = generator.model_dump()
        return JSONResponse(content=response,
                            status_code=generator.code)
    if request.stream:
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        response = generator.model_dump()
        return JSONResponse(content=response)

async def init_model(ip_list: List[str]):
    global engine
    global model_executor
    global openai_serving_chat
    global openai_serving_completion

    stime = time.time()
    print(f"Start Initialize Model at {stime - runtime_start_time} seconds")

    if use_local_server == 1 and no_pre_init == 1 and fast_coldstart == 1:
        remote_model_id = os.getenv('REMOTE_MODEL_ID')
        await start_load_model_shm(remote_model_id, pp_rank, pp_size, False)
        print(f"app.py: start load_model_shm time cost = {time.time() - stime} seconds")
    
    if pp_rank == 0:
        served_model_names = [model_id]
        engine_args = AsyncEngineArgs(
            model=model_path,
            enforce_eager=True,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        stime = time.time()
        engine = AsyncLLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.OPENAI_API_SERVER, ip_list=ip_list)
        time_1 = time.time()
        print(f"app.py: init engine time cost = {time_1 - stime} seconds")
        openai_serving_chat = OpenAIServingChat(engine, served_model_names, "assistant", chat_template=chat_template)
        time_2 = time.time()
        print(f"app.py: init serving chat time cost = {time_2 - time_1} seconds")
        openai_serving_completion = OpenAIServingCompletion(engine, served_model_names)
        time_3 = time.time()
        print(f"app.py: init serving completion time cost = {time_3 - time_2} seconds")
    else:
        # only initialize a GPU executor and start running
        engine_args = EngineArgs(
            model=model_path,
            enforce_eager=True,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        engine_config = engine_args.create_engine_config()
        max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))
        print(f"Warning: limit max_model_len to {max_model_len} due to limited KV cache")
        engine_config.model_config.max_model_len = max_model_len
        model_executor = GPUExecutorAsync(
            model_config=engine_config.model_config,
            cache_config=engine_config.cache_config,
            parallel_config=engine_config.parallel_config,
            scheduler_config=engine_config.scheduler_config,
            device_config=engine_config.device_config,
            lora_config=engine_config.lora_config,
            vision_language_config=engine_config.vision_language_config,
            speculative_config=engine_config.speculative_config,
            load_config=engine_config.load_config,
        )
        model_executor.init_connection(ip_list)
        num_gpu_blocks, num_cpu_blocks = (
            model_executor.determine_num_available_blocks())
        model_executor.initialize_cache(num_gpu_blocks, num_cpu_blocks)
    
    if use_local_server == 1 and fast_coldstart == 1:
        await end_load_model_shm()

async def background_loop():
    # This function keeps the inference loop running
    global model_executor

    if model_executor is not None:
        model_executor.start_running()

    while True:
        await asyncio.sleep(0)

    # TODO(chiheng): if we want to change this stage to engine, we should stop the background loop

@app.post("/init")
async def init_model_handler(request: Dict, raw_request: Request, background_task: BackgroundTasks):
    global openai_serving_chat

    if openai_serving_chat is not None:
        print(f"model inited, quit")
        return

    stime = time.time()
    ip_list = request["ip_list"]
    await init_model(ip_list)
    etime = time.time()
    print(f" initialize model elapsed: {etime - stime} seconds")
    background_task.add_task(background_loop)
    return "model inited"

if __name__ == "__main__":
    print(f"Start uvicorn at {time.time() - runtime_start_time} seconds")
    uvicorn.run(app,
                host='0.0.0.0',
                port=8080,
                log_level='debug',
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
