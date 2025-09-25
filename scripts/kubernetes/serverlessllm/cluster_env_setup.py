'''
[DEPRECATED]
Initialize nodes for ServerlessLLM using conda
'''
import os
import sys
import time
import asyncio
import multiprocessing
from kubernetes import client, config

def start_pod(node_name):
    os.system(f"kubectl node-shell {node_name} -- sleep 1000")

if __name__ == '__main__':
    config.load_kube_config()
    core_api = client.CoreV1Api()

    node_list = core_api.list_node()
    tasks = []
    gpu_nodes = []
    for node in node_list.items:
        node_labels = node.metadata.labels
        if "aliyun.accelerator/nvidia_name" in node_labels:
            node_name = node_labels["kubernetes.io/hostname"]
            gpu_nodes.append(node_name)

            p = multiprocessing.Process(target=start_pod, args=(node_name,))
            tasks.append(p)
            p.start()
    
    print(f"Prepare environment for nodes {gpu_nodes}")

    time.sleep(3)
    
    gpu_pods = [-1] * len(gpu_nodes)
    pods = core_api.list_namespaced_pod(namespace="default", watch=False)
    for pod in pods.items:
        pod_name = pod.metadata.name
        node_name = pod.spec.node_name
        if node_name in gpu_nodes:
            index = gpu_nodes.index(node_name)
            gpu_pods[index] = pod_name
    
    for rank in range(len(gpu_nodes)):
        print(f"Prepare {gpu_nodes[rank]}!")
        basic_comm = f"kubectl node-shell {gpu_nodes[rank]} --"

        os.system(f"kubectl cp setup.sh default/{gpu_pods[rank]}:/root")
        os.system(f"kubectl cp condarc default/{gpu_pods[rank]}:/root")
        os.system(f"kubectl cp dist default/{gpu_pods[rank]}:/root")
        os.system(f"kubectl exec {gpu_pods[rank]} /bin/bash /root/setup.sh 1 > log{rank}.txt 2>&1")

        print(f"Finished preparation for {gpu_nodes[rank]}!")

    os.system("sh./setup.sh 0 > log_master.txt 2>&1")
