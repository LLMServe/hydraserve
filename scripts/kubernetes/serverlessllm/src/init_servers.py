import os
import sys
import time
import json
import asyncio
import subprocess
from typing import List
from kubernetes import client, config

from utils import get_node_list_nogpu, get_node_list_gpu, get_node_list_with_resource, create_deployment, delete_all_deployments, post_ayncio_request_util_succ
from ModelInfo import ModelList

nas_path = "/mnt"
model_set = int(os.getenv("MODEL_SET", "3"))
select_largest = int(os.getenv("LARGEST", "0"))
config.load_kube_config()
core_api = client.CoreV1Api()
apps_api = client.AppsV1Api()

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

if __name__ == '__main__':
    expr_1_1 = True if int(os.getenv("EXPR_1_1", "0")) == 1 else False
    download_model = True if int(os.getenv("DOWNLOAD_MODEL", "0")) == 1 else False
    print("Delete existing deployment...")
    os.system("kubectl delete deployment --all")
    time.sleep(1)

    # Initialize storage server
    print("Start to initialize storage server...")

    node_list_nogpu = get_node_list_nogpu(core_api)
    node_list_gpu = get_node_list_with_resource(core_api)

    remote_servers = []
    network_limits = []
    image = "chlou/vllm-storage-server-remote:v1"
    env = {"MODELSCOPE_CACHE": "/models/vllm", "CACHE_TYPE": "serverlessllm", "MODEL_SET": str(model_set)}
    index = 0
    for node, net in node_list_nogpu.items():
        name = "storage-server-remote-" + str(index)
        create_deployment(apps_api, name, image, env, node, nas_path)
        remote_servers.append(name)
        network_limits.append(net)
        index += 1
    node_list_nogpu = list(node_list_nogpu.keys())

    # get ip list
    num_servers = len(remote_servers)
    server_ip_list = get_ip_list(remote_servers)

    time.sleep(3)

    print("Storage server initialized!")

    # Initialize head
    head_name = "sllm-server"
    env = {"GRACE_PERIOD": "30",
           "NUM_CPU": "40",
           "BATCH_SIZE": "8",
           }
    create_deployment(apps_api, head_name, "registry.us-east-1.aliyuncs.com/kubernetes-fc/sllm-serve:v1", env, node_list_nogpu[-1], nas_path)
    head_ip = get_ip_list([head_name])[0]

    print(f"----- Head IP = {head_ip} -----")
    print(f"export SERVER_POD_IP={head_ip}")

    with open("head_ip.txt", "w") as f:
        print(head_ip, file=f)
    
    print(f"head ip printed to head_ip.txt")

    # Initialize storage nodes. These nodes are control nodes that help decide autoscaling policy.
    storage_ray_nodes = []
    for index in range(0, len(node_list_nogpu) - 1):
        node = node_list_nogpu[index]
        pod_name = "ray-storage-" + str(index)
        env = {"HEAD_IP": head_ip, "STORAGE_SERVER": "1", "NUM_CPU": "40", "BATCH_SIZE": "8"}
        create_deployment(apps_api, pod_name, "registry.us-east-1.aliyuncs.com/kubernetes-fc/sllm-serve:v1", env, node)
        storage_ray_nodes.append(pod_name)
    storage_ray_nodes_ip = get_ip_list(storage_ray_nodes)

    # Initialize workers
    workers = []
    worker_resource = []
    index = 0
    for node, resource in node_list_gpu.items():
        name = "sllm-worker-" + str(index)
        node_mem = int(resource[3] * 0.9)
        node_net = resource[2]
        node_num_gpu = resource[0]
        node_gpu_mem = resource[1]
        max_mem = node_mem - (resource[1] + 4) * resource[0] - 5 # Additional memory used by worker. Each GPU worker needs (gpu_mem+4) cpu memory. Ray and controller nees 5GB cpu memory.
        env = {"WORKER_ID": str(index),
               "HEAD_IP": head_ip,
               "MEM_POOL_SIZE": str(max_mem),
               "DISK_SIZE": "99999999",
               "NUM_CPU": str(resource[4]-4),
               "BATCH_SIZE": "8",
               }
        if expr_1_1:
            backend = os.getenv("BACKEND", "hybrid")
            if backend == "a10":
                env["MEM_POOL_SIZE"] = "16"
            else:
                env["MEM_POOL_SIZE"] = "27"
        # Allocate remote server
        remote_server_rank = -1
        for rank in range(num_servers):
            if network_limits[rank] >= node_net:
                if remote_server_rank == -1:
                    remote_server_rank = rank
                else:
                    if select_largest and network_limits[rank] > network_limits[remote_server_rank]:
                        remote_server_rank = rank
                    elif not select_largest and network_limits[rank] < network_limits[remote_server_rank]:
                        remote_server_rank = rank
        if remote_server_rank == -1:
            # Try to find two instances
            node_net_ = node_net // 2
            rank_1 = -1
            rank_2 = -1
            for rank in range(num_servers):
                if network_limits[rank] >= node_net_:
                    if rank_1 == -1:
                        rank_1 = rank
                    else:
                        rank_2 = rank
                        break
            if rank_1 != -1 and rank_2 != -1:
                network_limits[rank_1] -= node_net_
                network_limits[rank_2] -= node_net_
                env["STORAGE_IP"] = server_ip_list[rank_1]
                env["STORAGE_IP_2"] = server_ip_list[rank_2]
                env["NUM_STORAGE"] = "2"
            else:
                raise RuntimeError("No enough network resource.")
        else:
            network_limits[remote_server_rank] -= node_net
            env["STORAGE_IP"] = server_ip_list[remote_server_rank]
            env["NUM_STORAGE"] = "1"
        create_deployment(apps_api, name, "registry.us-east-1.aliyuncs.com/kubernetes-fc/sllm-serve-worker:v1", env, node, nas_path, gpu=node_num_gpu, mem=node_mem)
        workers.append(name)
        worker_resource.append((max_mem, node_net, node_gpu_mem))
        index += 1

        if expr_1_1:
            print("Expr 1.1: only initialize single worker.")
            with open("worker_name.txt", "w") as f:
                print(node, file=f)
            print(f"Worker name printed to worker_name.txt")
            break
    
    worker_ips = get_ip_list(workers)

    time.sleep(3)
    print(f"Head and Worker initialized!")

    # Generate hardware config
    hardware_config = {}
    for index in range(len(worker_resource)):
        mem = worker_resource[index][0]
        net = worker_resource[index][1] / 8
        config = {
            "host_size": str(mem) + "GB",
            "host_bandwidth": "0GB/s",         # no use
            "disk_size": "99999999GB",
            "disk_bandwidth": str(net) + "GB/s",
            "network_bandwidth": "0GB/s",      # no use
        }
        hardware_config[str(index)] = config
    with open ('/root/hardware_config.json', 'w') as f:
        json.dump(hardware_config, f)
    
    # Generate model configs
    model_config_path = '/root/model_configs'
    model_cur_index = {}
    for task_type, model_dict in ModelList.items():
        for model, info in model_dict.items():
            suitable_workers = []
            for index in range(len(worker_resource)):
                gpu_mem_per_card = worker_resource[index][2]
                if gpu_mem_per_card >= info[0]:
                    suitable_workers.append(str(index))
            if model not in model_cur_index:
                model_cur_index[model] = 0
                cur_index = 0
            else:
                cur_index = model_cur_index[model]
            num_submodel = info[4]
            for idx in range(num_submodel):
                model_name = model + "/" + str(idx + cur_index)
                model_path = os.path.join("/models/vllm", model_name)
                config = {
                    "model": model_name,
                    "backend": "vllm",
                    "num_gpus": 1,
                    "auto_scaling_config": {
                        "metric": "concurrency",
                        "target": 8,
                        "min_instances": 0,
                        "max_instances": 100
                    },
                    "placement_config": {
                        "target_nodes": suitable_workers
                    },
                    "backend_config": {
                        "pretrained_model_name_or_path": model_path,
                        "device_map": "auto",
                        "torch_dtype": "float16"
                    }
                }
                path = os.path.join(model_config_path, model_name)
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "config.json"), 'w') as f:
                    json.dump(config, f)
            model_cur_index[model] += num_submodel
    
    # Get server pod name
    cmd = "kubectl get pod -o name | grep " + head_name + "-"
    pod_name = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('gbk')
    pod_name = pod_name[4:-1]

    # Copy hardware config
    cmd = "kubectl cp /root/hardware_config.json " + pod_name + ":/app/hardware_config.json"
    os.system(cmd)

    if not download_model:
        # Wait for remote storage startup
        print(f"Waiting for remote storage startup...")
        req = -1
        req_bytes = req.to_bytes(length=4, byteorder='little', signed=True)
        for ip in server_ip_list:
            # post request util success
            asyncio.run(post_ayncio_request_util_succ(ip, 8888, req_bytes))

    # Run server
    cmd = "kubectl exec " + pod_name + " -- sh -c \"/opt/conda/bin/sllm-serve start --hardware-config /app/hardware_config.json\""
    os.system(cmd)
