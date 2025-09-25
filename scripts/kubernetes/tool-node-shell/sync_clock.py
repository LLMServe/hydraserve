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
    nodes = []
    for node in node_list.items:
        node_labels = node.metadata.labels
        node_name = node_labels["kubernetes.io/hostname"]
        nodes.append(node_name)

        p = multiprocessing.Process(target=start_pod, args=(node_name,))
        tasks.append(p)
        p.start()
    
    print(f"Prepare environment for nodes {nodes}")

    time.sleep(10)
    
    node_pods = [-1] * len(nodes)
    pods = core_api.list_namespaced_pod(namespace="default", watch=False)
    for pod in pods.items:
        pod_name = pod.metadata.name
        node_name = pod.spec.node_name
        if node_name in nodes:
            index = nodes.index(node_name)
            node_pods[index] = pod_name
    
    for rank in range(len(nodes)):
        print(f"Prepare {nodes[rank]}!")
        basic_comm = f"kubectl node-shell {nodes[rank]} -- yum install -y chrony"

        print(f"Finished preparation for {nodes[rank]}!")

    os.sytem("kubectl delete pod --all")
