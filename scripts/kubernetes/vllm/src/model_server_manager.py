import os
import sys
import time
import asyncio
from typing import List

from utils import get_node_list_nogpu, get_node_list_gpu, get_node_list_with_resource, create_deployment, delete_deployment, post_ayncio_request, post_ayncio_request_util_succ
from ImageInfo import ImageList

nas_path = os.getenv("MODEL_DIR", "/mnt")
max_num_clients = 8     # max number of clients that a remote server can serve
model_set = int(os.getenv("MODEL_SET", "3"))
select_largest = int(os.getenv("LARGEST", "0"))

'''
A ModelManager manages model servers
'''
class ModelServerManager:
    def __init__(self, core_api, apps_api):
        self.core_api = core_api
        self.apps_api = apps_api
        self.node_list_gpu = get_node_list_with_resource(core_api)
        self.node_list_nogpu = get_node_list_nogpu(core_api)
        assert len(self.node_list_nogpu.keys()) * max_num_clients >= len(self.node_list_gpu.keys())

    async def get_ip_list(self, pod_list: List[str]):
        num_servers = len(pod_list)
        ip_list = [None] * num_servers
        num_ready = 0
        while num_ready < num_servers:
            await asyncio.sleep(0.05)
            pods = self.core_api.list_namespaced_pod(namespace="default", watch=False)
            for pod in pods.items:
                pod_name = pod.metadata.labels["app"]
                if pod_name in pod_list:
                    if isinstance(pod.status.pod_ip, str):
                        rank = pod_list.index(pod_name)
                        if ip_list[rank] is None:
                            ip_list[rank] = pod.status.pod_ip
                            num_ready += 1
        return ip_list

    async def init_storage_server(self):
        # start remote storage server on node_list_nogpu
        self.remote_servers = []
        remote_servers_net = []
        image = ImageList["storage-remote"]
        env = {"MODELSCOPE_CACHE": "/mnt/model-cache", "MODEL_SET": str(model_set)}
        index = 0
        for node, net in self.node_list_nogpu.items():
            name = "storage-server-remote-" + str(index)
            create_deployment(self.apps_api, name, image, env, node, nas_path)
            self.remote_servers.append(name)
            remote_servers_net.append(net)
            index += 1
        
        # get ip list
        num_servers = len(self.remote_servers)
        self.server_ip_list = await self.get_ip_list(self.remote_servers)

        # start local storage server on node_list_gpu
        self.local_servers = []
        image = ImageList["storage-local"]
        env = {"USE_CACHE": os.getenv("USE_CACHE", "0")}
        if int(os.getenv("SLOW_EXPR", "0")) == 1:
            env["SLOW_EXPR"] = "1"
        index = 0
        for node_name, resource in self.node_list_gpu.items():
            name = "storage-server-local-" + str(index)
            # allocate remote server
            node_net = resource[2]
            remote_server_rank = -1
            for rank in range(num_servers):
                if remote_servers_net[rank] >= node_net:
                    if remote_server_rank == -1:
                        remote_server_rank = rank
                    else:
                        if select_largest and remote_servers_net[rank] > remote_servers_net[remote_server_rank]:
                            remote_server_rank = rank
                        elif not select_largest and remote_servers_net[rank] < remote_servers_net[remote_server_rank]:
                            remote_server_rank = rank
            if remote_server_rank == -1:
                # Try to find two instances
                node_net_ = node_net // 2
                rank_1 = -1
                rank_2 = -1
                for rank in range(num_servers):
                    if remote_servers_net[rank] >= node_net_:
                        if rank_1 == -1:
                            rank_1 = rank
                        else:
                            rank_2 = rank
                            break
                if rank_1 != -1 and rank_2 != -1:
                    remote_servers_net[rank_1] -= node_net_
                    remote_servers_net[rank_2] -= node_net_
                    env["REMOTE_SERVER_ADDR"] = self.server_ip_list[rank_1]
                    env["REMOTE_SERVER_ADDR_2"] = self.server_ip_list[rank_2]
                    env["NUM_REMOTE_SERVER"] = "2"
                else:
                    raise RuntimeError("No enough remote servers.")
            else:
                remote_servers_net[remote_server_rank] -= node_net
                env["REMOTE_SERVER_ADDR"] = self.server_ip_list[remote_server_rank]
                env["NUM_REMOTE_SERVER"] = "1"
            node_mem = int(resource[3] * 0.9)
            server_mem_limit = node_mem - (resource[1] + 4) * resource[0] - 2
            env["SHM_SIZE"] = str(server_mem_limit - 3)
            create_deployment(self.apps_api, name, image, env, node_name, nas_path, mem=server_mem_limit)
            self.local_servers.append(name)
            index += 1

        # get ip list
        self.local_ip_list = await self.get_ip_list(self.local_servers)

        # initialize models for each remote server and shared memory for each local server
        req = 0
        req_bytes = req.to_bytes(length=8, byteorder='little', signed=False)
        tasks = []
        for ip in self.server_ip_list:
            tasks.append(post_ayncio_request_util_succ(ip, 8888, req_bytes))
        for ip in self.local_ip_list:
            tasks.append(post_ayncio_request_util_succ(ip, 6666, req_bytes))
        await asyncio.gather(*tuple(tasks))
    
    async def remove_servers(self):
        self.local_servers = []
        for rank in range(len(self.node_list_gpu.keys())):
            self.local_servers.append("storage-server-local-" + str(rank))
        
        self.local_ip_list = await self.get_ip_list(self.local_servers)

        # free shared memory for each local server
        req = 1
        req_bytes = req.to_bytes(length=8, byteorder='little', signed=False)
        tasks = []
        for ip in self.local_ip_list:
            tasks.append(post_ayncio_request(ip, 6666, req_bytes))
        await asyncio.gather(*tuple(tasks))
