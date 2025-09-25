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

nas_path = "/mnt"
config.load_kube_config()
core_api = client.CoreV1Api()
apps_api = client.AppsV1Api()

if __name__ == '__main__':
    # Initialize storage server
    print("Start to initialize images...")

    node_list_gpu = get_node_list_gpu(core_api)
    node_list = get_node_list_gpu(core_api) + list(get_node_list_nogpu(core_api).keys())

    servers = []
    workers = []
    
    image = "registry.us-east-1.aliyuncs.com/kubernetes-fc/sllm-serve:v1"
    index = 0
    for node in node_list:
        name = "server-" + str(index)
        create_deployment(apps_api, name, image, {}, node, nas_path)
        servers.append(name)
        index += 1
    
    index = 0
    for node in node_list_gpu:
        name = "worker-" + str(index)
        create_deployment(apps_api, name, "registry.us-east-1.aliyuncs.com/kubernetes-fc/sllm-serve-worker:v1", {}, node, nas_path)
        workers.append(name)
        index += 1

    # get ip list
    server_ip_list = get_ip_list(servers)
    worker_ip_list = get_ip_list(workers)

    os.system("kubectl delete deployment --all")
    print(f"All images downloaded!")