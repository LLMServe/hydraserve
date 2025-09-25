import os
import sys
import time
import json
import asyncio
import subprocess
from typing import List
from kubernetes import client, config

from utils import get_node_list_nogpu, get_node_list_gpu, get_node_list_with_resource, create_deployment, delete_all_deployments, post_request_util_succ
from ModelInfo import ModelList
from ImageInfo import ImageList

def get_ip_list(pod_list: List[str]):
    num_servers = len(pod_list)
    ip_list = [None] * num_servers
    num_ready = 0
    while num_ready < num_servers:
        pods = core_api.list_namespaced_pod(namespace="default", watch=False)
        for pod in pods.items:
            pod_name = pod.metadata.labels["app"]
            if pod_name in pod_list:
                if isinstance(pod.status.pod_ip, str):
                    rank = pod_list.index(pod_name)
                    if ip_list[rank] is None:
                        ip_list[rank] = pod.status.pod_ip
                        num_ready += 1
    return ip_list

nas_path = os.getenv("MODEL_DIR", "/mnt")
config.load_kube_config()
core_api = client.CoreV1Api()
apps_api = client.AppsV1Api()

if __name__ == '__main__':
    # Initialize storage server
    print("Start to initialize images...")

    gpu_node_list = get_node_list_gpu(core_api)
    node_list = get_node_list_gpu(core_api) + list(get_node_list_nogpu(core_api).keys())

    remote_servers = []
    workers = []
    local_servers = []
    image = ImageList["storage-remote"]
    index = 0
    env = {"MODELSCOPE_CACHE": os.path.join(nas_path, "model-cache")}
    for node in node_list:
        name = "storage-server-remote-" + str(index)
        create_deployment(apps_api, name, image, env, node, nas_path)
        remote_servers.append(name)
        index += 1
    
    index = 0
    for node in gpu_node_list:
        name = "worker-" + str(index)
        create_deployment(apps_api, name, ImageList["vllm"], {}, node, nas_path)
        workers.append(name)
        index += 1
    
    index = 0
    for node in gpu_node_list:
        name = "local-server-" + str(index)
        create_deployment(apps_api, name, ImageList["storage-local"], {}, node, nas_path)
        local_servers.append(name)
        index += 1

    # get ip list
    server_ip_list = get_ip_list(remote_servers)
    worker_ip_list = get_ip_list(workers)
    local_ip_list = get_ip_list(local_servers)

    os.system("kubectl delete deployment --all")
    print(f"All images downloaded!")