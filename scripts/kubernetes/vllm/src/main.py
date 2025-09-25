import os
import sys
import time
import json
import random
from kubernetes import client, config
from openai import OpenAI
from typing import List, Dict, Optional
import traceback
import threading

import fastapi
import uvicorn
from fastapi import Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import asyncio

from utils import create_deployment, delete_deployment, post_request, post_request_util_succ
from engine import ChatEngine
from scheduler import Scheduler
from ModelInfo import ModelList

app = fastapi.FastAPI()

scheduler = None
check_instance_loop_start = False

log_path = os.getenv("LOG_PATH", "/root/logs")

async def check_instance_loop():
    global scheduler
    await scheduler.check_instance_loop(30)
    # never return
    raise RuntimeError("Check instance loop returned.")

class RequestModel(BaseModel):
    id: int
    model: str
    prompt: str
    stream: Optional[bool] = False

@app.post('/')
async def request_handler(request: RequestModel, raw_request: Request, background_task: BackgroundTasks):
    global scheduler
    global check_instance_loop_start
    global RequestCounter

    request_id = request.id
    prompt = request.prompt
    model = request.model
    stream = request.stream
    if stream:
        prompt = eval(prompt)
    print(f"Received request [{request_id}] for model = {model}")

    if model not in scheduler.model_list:
        return JSONResponse(content="Model not found in ModelList",
                            status_code=404)

    start_time = time.time()
    try:
        reply = await scheduler.process_request(request_id, model, prompt, stream=stream)
    except Exception as e:
        exc_info = sys.exc_info()
        print(f"Request [{request_id}] gets exception: {e}")
        print("".join(traceback.format_exception(*exc_info)))
    elapsed_time = time.time() - start_time

    if not check_instance_loop_start:
        check_instance_loop_start = True
        background_task.add_task(check_instance_loop)

    if stream:
        return StreamingResponse(content=reply,
                                 media_type="text/event-stream")
    else:
        assert isinstance(reply, str)
        return JSONResponse(content=reply)

if __name__ == '__main__':
    config.load_kube_config()
    apps_api = client.AppsV1Api()
    core_api = client.CoreV1Api()

    random.seed(998244353)

    os.makedirs(log_path, exist_ok=True)
    scheduler = Scheduler(core_api, apps_api, ModelList)
    uvicorn.run(app,
            host='0.0.0.0',
            port=9090,
            log_level='debug',
            timeout_keep_alive=5)