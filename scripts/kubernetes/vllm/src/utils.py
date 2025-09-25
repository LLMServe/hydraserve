import os
import sys
import time
import asyncio
import requests
import aiohttp
import subprocess
from typing import Dict, List, Tuple, Optional
from kubernetes import client, config

from ECSInstanceInfo import AliyunECSInstanceInfo

conn = None
log_path = os.getenv("LOG_PATH", "/root/logs")
used_machines = os.getenv("USED_MACHINES", "")
if used_machines:
    used_machines = used_machines.split(",")

def post_request(url: str, payload):
    return requests.post(url, json=payload)

def post_request_util_succ(url: str, payload):
    while True:
        try:
            resp = post_request(url, payload)
            return resp
        except Exception as e:
            time.sleep(0.05)

# Used to communicate with vllm pods
async def post_request_async(url: str, payload):
    global conn
    if conn is None:
        conn = aiohttp.TCPConnector(limit=0, ssl=False)
    async with aiohttp.request('POST', url, json=payload, connector=conn) as resp:
        return await resp.json()

async def post_request_util_succ_async(url: str, payload):
    counter = 0
    while True:
        try:
            resp = await post_request_async(url, payload)
            return resp
        except Exception as e:
            await asyncio.sleep(0.05)
            counter = counter + 1
            if counter == 4000:
                print(f"Error: failed to post aiohttp request, url = {url}, payload = {payload}")
                print(f"Exception: %s", e)
                break

# Used to communicate with local storage server
async def post_ayncio_request(url: str, port: int, payload):
    # stime = time.time()
    reader, writer = await asyncio.open_connection(url, port)

    writer.write(payload)
    await writer.drain()

    ret = bytes()
    while True:
        resp = await reader.read(50)
        if len(resp) == 0:
            break
        ret += resp
        await asyncio.sleep(0.1)
    writer.close()
    await writer.wait_closed()
    # print(f"post_ayncio_request elapsed {time.time() - stime} seconds")

    return ret

async def post_ayncio_request_util_succ(url: str, port: int, payload):
    while True:
        try:
            ret = await post_ayncio_request(url, port, payload)
            return ret
        except Exception as e:
            await asyncio.sleep(0.05)

'''
Note:
gpu unit: (mem in GB, core in %)
net unit: Gbps
'''
def create_deployment(apps_api, name: str, image: str, env: Dict[str, str], node: Optional[str] = None, nas_path: Optional[str] = None, gpu: Optional[Tuple[int, int]] = None, net: Optional[int] = None, mem: Optional[int] = None):
    env_list = []
    for key, value in env.items():
        env_list.append(client.V1EnvVar(key, value))

    volumes = [client.V1Volume(name=name + "-shm-vol", host_path=client.models.V1HostPathVolumeSource(path="/dev/shm", type="Directory"))]
    volumeMounts = [client.V1VolumeMount(name=name + "-shm-vol", mount_path="/dev/shm")]
    if nas_path is not None:
        # volumes = [client.V1Volume(name=name + "-vol", nfs=client.models.V1NFSVolumeSource(path="/", server=nfs_server_addr))]
        volumes.append(client.V1Volume(name=name + "-vol", host_path=client.models.V1HostPathVolumeSource(path=nas_path, type="Directory")))
        volumeMounts.append(client.V1VolumeMount(name=name + "-vol", mount_path="/mnt"))
    
    node_selector = None
    if node is not None:
        node_selector = {"kubernetes.io/hostname": node}

    resources = None
    if gpu is not None:
        if mem is not None:
            print(f"Warning: Want to set {mem} GB memory. However, the GPU pod's memory capacity is limited to {gpu[0] + 4}")
        resources = client.V1ResourceRequirements(limits={"aliyun.com/gpu-mem": str(gpu[0]), "memory": str(gpu[0] + 4) + "Gi"})
    elif mem is not None:
        resources = client.V1ResourceRequirements(limits={"memory": str(mem) + "Gi"})

    annotations = None
    # TODO(chiheng): since we are using hostpath, the limitation of net does not work.
    # if net is not None and net > 0:
    #     annotations = {"kubernetes.io/ingress-bandwidth": str(net) + "G"}

    container = client.V1Container(
        name=name,
        image=image,
        resources=resources,
        env=env_list,
        volume_mounts=volumeMounts,
        image_pull_policy="IfNotPresent", # "Always"
    )

    # Create and configure a spec section
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": name}, annotations=annotations),
        spec=client.V1PodSpec(containers=[container], volumes=volumes, node_selector=node_selector, host_ipc=True, termination_grace_period_seconds=0),
    )

    # Create the specification of deployment
    spec = client.V1DeploymentSpec(
        template=template, selector={
            "matchLabels":
            {"app": name}})
    
    # Instantiate the deployment object
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=name),
        spec=spec,
    )

    response = apps_api.create_namespaced_deployment(
        body=deployment, namespace="default"
    )

    print(f"Deployment {name} created.")

def delete_deployment(apps_api, name: str):
    # 1. Save logs
    # get pod name
    stime = time.time()
    cmd = "kubectl get pod -o name | grep " + name + "-"
    pod_name = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('gbk')
    # write log to a file
    try:
        cmd = "kubectl logs " + pod_name
        log_name = os.path.join(log_path, name + ".txt")
        mode = 'w' if os.path.isfile(log_name) else 'x'
        with open(log_name, mode) as output_file:
            process = subprocess.run(cmd, shell=True, stdout=output_file)
        print(f"Save logs elapsed {time.time() - stime} seconds")
        # 2. Delete deployment
        api_response = apps_api.delete_namespaced_deployment(
            name=name,
            namespace="default",
            body=client.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=0))
    except Exception as e:
        print(f"Error: save logs and delete deployment meet exception: {e}")

def delete_all_deployments(core_api, apps_api):
    pods = core_api.list_namespaced_pod(namespace="default", watch=False)
    for pod in pods.items:
        pod_name = pod.metadata.labels["app"]
        api_response = apps_api.delete_namespaced_deployment(
            name=pod_name,
            namespace="default",
            body=client.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=0))

def check_deployment_fin(apps_api, name: str) -> bool:
    deployments = apps_api.list_namespaced_deployment(namespace="default")
    for deployment in deployments.items:
        if deployment.metadata.name == name:
            return False
    print(f"Deployment {name} deleted.")
    return True

def create_job(batch_api, name: str, image: str, env: Dict[str, str], node: Optional[str] = None, nas_path: Optional[str] = None):
    env_list = []
    for key, value in env.items():
        env_list.append(client.models.V1EnvVar(key, value))

    volumes = None
    volumeMounts = None
    if nas_path is not None:
        # volumes = [client.V1Volume(name=name + "-vol", nfs=client.models.V1NFSVolumeSource(path="/", server=nfs_server_addr))]
        volumes = [client.V1Volume(name=name + "-vol", host_path=client.models.V1HostPathVolumeSource(path=nas_path, type="Directory"))]
        volumeMounts = [client.V1VolumeMount(name=name + "-vol", mount_path="/mnt")]

    node_selector = None
    if node is not None:
        node_selector = {"kubernetes.io/hostname": node}

    # Configureate Pod template container
    container = client.V1Container(
        name=name,
        image=image,
        env=env_list,
        volume_mounts=volumeMounts,
        image_pull_policy="Always",
    )

    # Create and configure a spec section
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": name}),
        spec=client.V1PodSpec(containers=[container], volumes=volumes, restart_policy="Never", node_selector=node_selector),
    )

    # Create the specification of deployment
    spec = client.V1JobSpec(
        template=template,
        backoff_limit=0)
    
    # Instantiate the deployment object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=name),
        spec=spec,
    )

    response = batch_api.create_namespaced_job(
        body=job, namespace="default"
    )

    print(f"Job {name} created.")

def check_job_fin(batch_api, name: str) -> bool:
    api_response = batch_api.read_namespaced_job_status(
        name=name,
        namespace="default")
    if api_response.status.succeeded is not None or \
            api_response.status.failed is not None:
        return True
    return False

def delete_job(batch_api, name: str):
    api_response = batch_api.delete_namespaced_job(
        name=name,
        namespace="default",
        body=client.V1DeleteOptions(
            propagation_policy='Foreground',
            grace_period_seconds=0))
    print(f"Job {name} deleted.")

def get_node_list_gpu(core_api) -> List[str]:
    node_list = core_api.list_node()
    labels = []
    for node in node_list.items:
        node_labels = node.metadata.labels
        node_type = node_labels["node.kubernetes.io/instance-type"]
        if used_machines and node_type not in used_machines:
            continue
        if "gpushare" in node_labels or "ack.node.gpu.schedule" in node_labels:
            labels.append(node_labels["kubernetes.io/hostname"])
    return labels

def get_node_list_nogpu(core_api) -> Dict[str, int]:
    node_list = core_api.list_node()
    labels = {}
    for node in node_list.items:
        node_labels = node.metadata.labels
        node_type = node_labels["node.kubernetes.io/instance-type"]
        if "gpushare" not in node_labels and "ack.node.gpu.schedule" not in node_labels:
            if node_type not in AliyunECSInstanceInfo:
                net = 0
            else:
                net = AliyunECSInstanceInfo[node_type]
                if isinstance(net, tuple):
                    net = net[2]
            labels[node_labels["kubernetes.io/hostname"]] = net
        elif used_machines and node_type not in used_machines and node_type in AliyunECSInstanceInfo:
            net = AliyunECSInstanceInfo[node_type][2]
            labels[node_labels["kubernetes.io/hostname"]] = net
    return labels

'''
Get GPU nodes and their instance information [format: (gpu number, gpu mem in GB, network bandwidth in Gbps)]
'''
def get_node_list_with_resource(core_api) -> Dict[str, Tuple[int, int, int]]:
    node_list = core_api.list_node()
    nodes = {}
    for node in node_list.items:
        node_labels = node.metadata.labels
        if "gpushare" in node_labels or "ack.node.gpu.schedule" in node_labels:
            node_name = node_labels["kubernetes.io/hostname"]
            node_type = node_labels["node.kubernetes.io/instance-type"]
            if used_machines and node_type not in used_machines:
                continue
            if node_type not in AliyunECSInstanceInfo:
                print(f"Error: node type {node_type} unrecognized, set resources to zero")
                resource = (0, 0, 0, 0)
            else:
                resource = AliyunECSInstanceInfo[node_type]
            nodes[node_name] = resource
    return nodes