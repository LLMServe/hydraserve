"""
[This file has been deprecated.]
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import time
runtime_start_time = time.time()
print("runtime started")

import sys
import os
import threading
import asyncio
from torch import ops

cur_time = time.time()
print(f"import initial libraries time cost = {cur_time - runtime_start_time} seconds")

engine = None
model_path = None
background_loop_unsheilded = None
background_loop = None

# get model-info from env 

FC_MODEL_CACHE_DIR = os.getenv('MODELSCOPE_CACHE')
task = os.getenv('TASK', '')
pp_rank = int(os.getenv('PP_RANK', ''))
pp_size = int(os.getenv('PP_SIZE', ''))
model_id = os.getenv('MODEL_ID', '')
enable_para = int(os.getenv('ENABLE_PARA', '1'))
model_id = model_id.replace('.', '___')
pp_tail = ""
counter = 0
seq_len = 0
model_kwargs = {}
if pp_size > 1:
    pp_tail = '-' + str(pp_rank) + '-' + str(pp_size)
model_path = FC_MODEL_CACHE_DIR + '/' + model_id + pp_tail

if not os.path.exists(model_path):
    print(f"model path = {model_path}")
    raise ValueError('[ERROR] model not found in cache')

os.environ["MODEL_PATH"] = os.path.join(model_path, "model.safetensors")

if enable_para == 1:
    LIB_DIR = "/vllm-workspace/libst_pybinding.so" 
    if not os.path.exists(LIB_DIR):
        raise RuntimeError(
            f"Could not find the C++ library libst_pybinding.so at {LIB_DIR}. "
            "Please build the C++ library first or put it at the right place."
        )
    ops.load_library(LIB_DIR)

    end_time = time.time()
    print(f" load library time cost = {end_time - cur_time} seconds")
    start_time = end_time
else:
    start_time = time.time()

import argparse
import json
import ssl
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.executor.gpu_executor import GPUExecutor
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid

end_time = time.time()
print(f"import vllm libs completed, time cost = {end_time - start_time} seconds")

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_id = request.headers.get("x-fc-request-id")
    request_dict = await request.json()
    
    # process prompt
    prompt = ""
    if "input" in request_dict:
        input = request_dict["input"]
        messages = input['messages']
        print(f"messages = {messages}")

        # use chat template to tokenize
        prompt = engine.engine.tokenizer.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    print(f"Prompt: {prompt}")
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)

    assert engine is not None
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    response = final_output.outputs[0].text
    
    eplased_time = time.time() - runtime_start_time
    message = { "eplased_time": eplased_time }
    result = { "message": { "content":  response } }
    
    ret = { 'Message':  message,
            'Data': result,
            'RequestId': request_id,
            'Success': True }
    return JSONResponse(ret)


if __name__ == "__main__":
    stime = time.time()
    print(f"Start Initialize Model at {stime - runtime_start_time} seconds")
    if pp_rank == 0:
        engine_args = AsyncEngineArgs(
            model=model_path,
            enforce_eager=True,
            trust_remote_code=True
        )
        engine = AsyncLLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.API_SERVER)
    else:
        # only initialize a GPU executor and start running
        engine_args = EngineArgs(
            model=model_path,
            enforce_eager=True,
            trust_remote_code=True
        )
        engine_config = engine_args.create_engine_config()
        engine = GPUExecutor(
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
        num_gpu_blocks, num_cpu_blocks = (
            engine.determine_num_available_blocks())
        engine.initialize_cache(num_gpu_blocks, num_cpu_blocks)
        # start background loop
        background_loop_unsheilded = asyncio.get_event_loop().create_task(engine.driver_worker.run())
        background_loop = asyncio.shield(background_loop_unsheilded)
        print("background loop started")
    etime = time.time()
    elapsed = etime - stime
    print(f" initialize model elapsed: {elapsed:.1f}")
    uvicorn.run(app,
                host='0.0.0.0',
                port=9000,
                log_level='debug',
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
