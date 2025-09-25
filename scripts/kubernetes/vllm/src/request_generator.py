import os
import sys
import pickle
import time
import requests
import random
import aiohttp
import asyncio
import errno
import httpx
import traceback
from random import sample
from openai import AsyncOpenAI

from result_analyzer import get_requests
from ModelInfo import ModelList

# Keep a model in running
async def keep_running(model, prompt):
    conn = aiohttp.TCPConnector(limit=0, ssl=False)
    timeout = aiohttp.ClientTimeout(total=1800)
    batch_size = 32
    tasks = []
    payload = {'id': 0, 'model': model, 'prompt': str(prompt), 'stream': 'True'}
    async def post_request(payload):
        payload_ = payload.copy()
        while True:
            payload_['id'] = random.randint(0, 1000000)
            resp_bytes = bytes()
            async with aiohttp.request('POST', "http://0.0.0.0:9090", json=payload_, connector=conn, timeout=timeout) as resp:
                async for data in resp.content.iter_any():
                    resp_bytes += data
            resp_str = resp_bytes.decode('utf-8')
            pos = resp_str.rfind('#')
            num_generated_tokens = int(resp_str[pos+1:])
            await asyncio.sleep(0)
    for _ in range(batch_size):
        tasks.append(asyncio.create_task(post_request(payload)))
    
    for task in tasks:
        await task

async def run_requests(req):
    serverless_llm = int(os.getenv('SERVERLESS_LLM', '0'))

    start_time = time.time()
    base_time = reqs[0][0] / time_scale

    req_start_time = [0] * len(reqs)
    first_token_time = [0] * len(reqs)
    generation_time = [0] * len(reqs)
    num_token = [0] * len(reqs)
    tasks = []

    if serverless_llm == 1:
        url = os.getenv('LLM_SERVER_URL') + "v1"
        timeout = httpx.Timeout(timeout=1800)
        custom_httpx_client = httpx.AsyncClient(timeout=timeout)
        client = AsyncOpenAI(
            base_url=url,
            api_key='serverlessllm',
            http_client=custom_httpx_client,
        )
    else:
        conn = aiohttp.TCPConnector(limit=0, ssl=False)
        timeout = aiohttp.ClientTimeout(total=1800)

    for index, req in enumerate(reqs):
        arrival_time, model_id, prompt = req
        arrival_time_in_second = arrival_time / time_scale
        while True:
            cur_time = time.time()
            if cur_time - start_time >= arrival_time_in_second - base_time:
                break
            await asyncio.sleep(0)
        print(f"Generate request {[index + 1]} for model {model_id}")
        payload = {'id': index + 1, 'model': model_id, 'prompt': str(prompt), 'stream': 'True'}

        async def post_request(index, payload):
            while True:
                try:
                    req_start_time[index] = time.time()
                    first_token_returned = False
                    if serverless_llm == 1:
                        prompt = [{"role": "user", "content": payload['prompt']}]
                        chat = await client.chat.completions.create(
                            model=payload['model'],
                            messages=prompt,
                            max_tokens=1024,
                            temperature=1,
                        )
                        if chat.created is not None:
                            generation_time[index] = time.time()
                            first_token_time[index] = generation_time[index] - chat.created - req_start_time[index]
                            if chat.usage is not None:
                                num_generated_tokens = chat.usage.completion_tokens
                            else:
                                num_generated_tokens = 0
                            first_token_returned = True
                        else:
                            print(f"Error: request [{index + 1}] does not contain first token time, usage = {chat.usage}. Retry.")
                            continue
                    else:
                        resp_bytes = bytes()
                        async with aiohttp.request('POST', "http://0.0.0.0:9090", json=payload, connector=conn, timeout=timeout) as resp:
                            async for data in resp.content.iter_any():
                                if not first_token_returned:
                                    first_token_time[index] = time.time() - req_start_time[index]
                                    first_token_returned = True
                                resp_bytes += data
                        resp_str = resp_bytes.decode('utf-8')
                        pos = resp_str.rfind('#')
                        num_generated_tokens = int(resp_str[pos+1:])
                        generation_time[index] = time.time()
                    if not first_token_returned:
                        print(f"Error: request [{index + 1}] for model {payload['model']} has no return.")
                    break
                except OSError as e:
                    if e.errno == errno.ECONNRESET:
                        print("Meet exception: Connection Reset By Peer. Retry.")
                    else:
                        exc_info = sys.exc_info()
                        print(f"Request [{index + 1}] gets exception: {e}")
                        if serverless_llm == 0:
                            print(f"current resp = {resp_bytes}")
                        print("".join(traceback.format_exception(*exc_info)))
                        break
                except Exception as e:
                    exc_info = sys.exc_info()
                    print(f"Request [{index + 1}] gets exception: {e}")
                    if serverless_llm == 0:
                        print(f"current resp = {resp_bytes}")
                    print("".join(traceback.format_exception(*exc_info)))
                    break
            if first_token_returned:
                decode_time = generation_time[index] - req_start_time[index] - first_token_time[index]
                num_decode_tokens = num_generated_tokens - 1
                tpot = (decode_time / num_decode_tokens) * 1000.0 if num_decode_tokens >= 1 else -1
                print(f"[{index + 1}]: TTFT = {'%.2f' % first_token_time[index]}s, TPOT = {'%.2f' % tpot} ms, #Tokens = {num_generated_tokens} ({payload['model']})", flush=True)
            else:
                print(f"[{index + 1}]: No Token Response!")
        tasks.append(asyncio.create_task(post_request(index, payload)))

    print(f"End of generating requests, elapsed = {time.time() - start_time}")

    for task in tasks:
        await task
    
    print(f"All tasks finished, elapsed = {time.time() - start_time}", flush=True)

    if serverless_llm == 0:
        conn.close()
    # for index in range(len(reqs)):
    #     print(f"[{index + 1}] TTFT = {first_token_time[index] - req_start_time[index]}")
    #     print(f"[{index + 1}] generation time cost = {generation_time[index] - req_start_time[index]}")

async def raw_load_generator():
    serverless_llm = int(os.getenv('SERVERLESS_LLM', '0'))

    if serverless_llm == 1:
        url = os.getenv('LLM_SERVER_URL') + "v1"
        client = AsyncOpenAI(
            base_url=url,
            api_key='serverlessllm'
        )
    else:
        conn = aiohttp.TCPConnector(limit=0, ssl=False)
        timeout = aiohttp.ClientTimeout(total=3600)
    
    async def post_request(payload, url):
        try:
            first_token_returned = False
            stime = time.time()
            if serverless_llm == 1:
                prompt = [{"role": "user", "content": payload['prompt']}]
                chat = await client.chat.completions.create(
                    model=payload['model'],
                    messages=prompt,
                    max_tokens=10,
                )
                generation_time = time.time()
                first_token_time = generation_time - chat.created - stime
                if chat.usage is not None:
                    num_generated_tokens = chat.usage.completion_tokens
                else:
                    num_generated_tokens = 0
                first_token_returned = True
            else:
                resp_bytes = bytes()
                async with aiohttp.request('POST', url, json=payload, connector=conn, timeout=timeout) as resp:
                    async for data in resp.content.iter_any():
                        if not first_token_returned:
                            first_token_time = time.time() - stime
                            first_token_returned = True
                        resp_bytes += data
                resp_str = resp_bytes.decode('utf-8')
                pos = resp_str.rfind('#')
                num_generated_tokens = int(resp_str[pos+1:])
                if not first_token_returned:
                    print(f"Error: request for model {model_id} has no return")
                generation_time = time.time()
        except Exception as e:
                exc_info = sys.exc_info()
                print(f"Request gets exception: {e}")
                print("".join(traceback.format_exception(*exc_info)))
        return first_token_time, generation_time - stime - first_token_time, num_generated_tokens
    
    expr_model_list = []
    for task, models in ModelList.items():
        for model, model_info in models.items():
            if model not in expr_model_list:
                expr_model_list.append(model)

    ttft_list = []
    tpot_list = []

    num_profile_points = 3
    prompt = [{'role': 'user', 'content': 'Write a story in 500 words'}]

    use_cache = False
    if serverless_llm == 1:
        backend = os.getenv("BACKEND", "hybrid")
        if backend == "a10":
            sllm_cache_model = "modelscope/Llama-2-7b-chat-ms"
        else:
            sllm_cache_model = "modelscope/Llama-2-13b-chat-ms"
        sllm_cache_payload = {'id': 0, 'model': sllm_cache_model, 'prompt': str(prompt), 'stream': 'True'}
        print("Init instances.")
        # init instance for each model first
        for model_id in expr_model_list:
            payload = {'id': 0, 'model': model_id, 'prompt': str(prompt), 'stream': 'True'}
            ttft, gt, num_tokens = await post_request(payload, "http://0.0.0.0:9090")
        print("Initialization Complete!")
        if int(os.getenv("USE_CACHE", "0")) == 1:
            use_cache = True

    cur_id = 0
    for model_id in expr_model_list:
        model_id_ = model_id
        if serverless_llm == 0:
            model_id_ += "/0"
        payload = {'id': cur_id, 'model': model_id_, 'prompt': str(prompt), 'stream': 'True'}
        cur_id += 1
        sum_ttft = 0
        sum_tpot = 0
        mn_ttft = 10000
        mn_tpot = 10000
        mx_ttft = 0
        mx_tpot = 0
        if use_cache:
            # post a request to generate cache
            ttft, gt, num_tokens = await post_request(payload, "http://0.0.0.0:9090")
            time.sleep(35)
        for i in range(num_profile_points):
            print(f"Start measure model {model_id} [{i}]", flush=True)
            if serverless_llm == 1 and not use_cache:
                # clear cache
                if model_id == sllm_cache_payload['model']:
                    if backend == "a10":
                        sllm_cache_model_ = "LLM-Research/Meta-Llama-3-8B-Instruct"
                    else:
                        sllm_cache_model_ = "facebook/opt-13b"
                    sllm_cache_payload_ = {'id': 0, 'model': sllm_cache_model_, 'prompt': str(prompt), 'stream': 'True'}
                    ttft, gt, num_tokens = await post_request(sllm_cache_payload_, "http://0.0.0.0:9090")
                else:
                    ttft, gt, num_tokens = await post_request(sllm_cache_payload, "http://0.0.0.0:9090")
                time.sleep(35)
            ttft, gt, num_tokens = await post_request(payload, "http://0.0.0.0:9090")
            tpot = gt / (num_tokens - 1)
            tpot *= 1000.0
            mn_ttft = min(mn_ttft, ttft)
            mx_ttft = max(mx_ttft, ttft)
            mn_tpot = min(mn_tpot, tpot)
            mx_tpot = max(mx_tpot, tpot)
            sum_ttft += ttft
            sum_tpot += tpot
            print(f"End measure model {model_id} [{i}], ttft = {ttft} s, tpot = {tpot} ms", flush=True)
            time.sleep(35)
        ttft = (sum_ttft - mx_ttft) / (num_profile_points - 1)
        tpot = (sum_tpot - mx_tpot) / (num_profile_points - 1)
        print(f"Model {model_id}: TTFT = {ttft} s, TPOT = {tpot} ms", flush=True)
        ttft_list.append(ttft)
        tpot_list.append(tpot)
    
    for index, model_id in enumerate(expr_model_list):
        print(f"Model {model_id}: TTFT = {ttft_list[index]} s, TPOT = {tpot_list[index]} ms", flush=True)

if __name__ == '__main__':
    # Warning: for expr1.1, remote serve may not be able to store so many models at a time, so you need to run two rounds to test all models.
    expr_1_1 = True if int(os.getenv("EXPR_1_1", "0")) == 1 else False
    expr_deployment = False
    if expr_1_1:
        asyncio.run(raw_load_generator())
        exit(0)
    
    workload_path = sys.argv[1]
    req_per_second = float(sys.argv[2])
    reqs, time_scale, mx_time_stamp = get_requests(workload_path, req_per_second)

    count = {}
    for index, req in enumerate(reqs):
        arrival_time, model_id, prompt = req
        if model_id not in count:
            count[model_id] = 1
        else:
            count[model_id] += 1
    count_list = []
    num_7b = 0
    num_7b_invoke = 0
    num_13b = 0
    num_13b_invoke = 0
    for key, value in count.items():
        count_list.append(value)
        if "7" in key:
            num_7b += 1
            num_7b_invoke += value
        else:
            num_13b += 1
            num_13b_invoke += value
    count_list.sort()
    print(f"#Functions = {len(count_list)}, Invocation Counts: {count_list}")
    print(f"#7B Models = {num_7b}, Invocation Counts: {num_7b_invoke}")
    print(f"#13B Models = {num_13b}, Invocation Counts: {num_13b_invoke}")

    if expr_deployment:
        # print the arrival time of the top model
        max_count = 0
        max_count_model = None
        for key, value in count.items():
            if value > max_count:
                max_count = value
                max_count_model = key
        arrivals = []
        for req in reqs:
            arrival_time, model_id, prompt = req
            if model_id == max_count_model:
                arrivals.append(arrival_time)
        path = sys.argv[3]
        with open(path, "w") as f:
            for arrival in arrivals:
                f.write(str(arrival) + '\n')
        exit(0)

    if len(sys.argv) == 4:
        # print all models into given file
        path = sys.argv[3]
        with open(path, "w") as f:
            for key, value in count.items():
                f.write(key + '\n')
        exit(0)

    print(f"Start generate {len(reqs)} requests in {mx_time_stamp} seconds")

    asyncio.run(run_requests(reqs))
