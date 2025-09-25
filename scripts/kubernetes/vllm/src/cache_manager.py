import os
import time
import json
import math
from typing import List
import asyncio
import threading
from utils import post_ayncio_request

class CacheManager:
    def __init__(self, storage_server_ip_list: List[str]):
        self.storage_servers = storage_server_ip_list
        self.storage_server_size = []
        # self.caches is a dict that maps model_id to a list that records every node_id that holds the model
        self.caches = {}
        self.tasks = set()

        flag = 2
        payload = flag.to_bytes(length=8, byteorder='little', signed=False)

        # Send request to each local storage server, clearing cache and obtaining total cache size
        for server in self.storage_servers:
            resp = asyncio.run(post_ayncio_request(server, 6666, payload))
            cache_size = int.from_bytes(resp, byteorder='little', signed=False)
            self.storage_server_size.append(cache_size)
        
    def add_cache(self, model_id: str, node_id: int):
        print(f"add_cache: model_id = {model_id}, node_id = {node_id}")
        if model_id not in self.caches:
            self.caches[model_id] = [node_id]
        else:
            self.caches[model_id].append(node_id)

    async def query_node(self, node_id: int, model_id: str, payload: bytes):
        resp = await post_ayncio_request(self.storage_servers[node_id], 6666, payload)
        flag = int.from_bytes(resp, byteorder='little', signed=False)
        if flag == 2:
            print(f"Error: model {model_id} not found in local storage server {node_id}")
        elif flag == 1:
            print(f"Model {model_id} has been evicted from local storage server {node_id}")
            self.caches[model_id].remove(node_id)
        elif flag == 3:
            # model still in cache
            pass
        else:
            print(f"Error: received unrecognized flag {flag}")
    
    '''
    Query which nodes have the model cache
    '''
    async def query_model_cache(self, model_id: str) -> List[int]:
        if model_id not in self.caches:
            return []
        flag = 3
        payload = flag.to_bytes(length=8, byteorder='little', signed=False) + bytes(model_id, encoding='utf-8')

        new_tasks = []
        node_list = self.caches[model_id].copy()
        for node_id in node_list:
            task = asyncio.create_task(self.query_node(node_id, model_id, payload))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            new_tasks.append(task)

        for task in new_tasks:
            await task

        return self.caches[model_id]

