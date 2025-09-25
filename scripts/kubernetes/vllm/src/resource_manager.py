import os
import time
import asyncio
from typing import List, Dict, Tuple
from utils import create_deployment, delete_deployment, get_node_list_with_resource, post_request_util_succ_async, check_deployment_fin
from dataclasses import dataclass

from ImageInfo import ImageList

nas_path = os.getenv("MODEL_DIR", "/mnt")

@dataclass
class ResourceUsage:
    pod_name: str
    node_id: int
    gpu_id: int
    # allocated gpu memory
    alloc_gpu_mem: int
    # actual gpu memory used
    num_gpu_mem: int

@dataclass
class AllocatedResource:
    num_nodes: int
    nodes: List[int]
    gpu_mem: List[int]
    net_usage: List[int]
    will_scale_down: bool = False

    def print_info(self, model):
        print(f"Size = {len(self.nodes)}, Nodes = {self.nodes}")
        print(f"GPU Usage: {self.gpu_mem}")
        print(f"Net Usage: {self.net_usage}")
        print(f"Current cost: {'%.1f' % model.model_stats.get_cur_cost()}; output tokens: {model.model_stats.num_output_tokens}")
        print(f"Profiled tpot: {model.model_stats.tpot} ms")

'''
A NetworkEvent defines a model pulling process
'''
class NetworkEvent:
    def __init__(self, id: int, model_size: float, start_time: float, slo: float, bandwidth: float):
        self.id = id
        self.model_size = model_size
        self.start_time = start_time
        self.slo = slo
        self.bandwidth = bandwidth

    def update_status(self, cur_time: float) -> bool:
        self.model_size -= (cur_time - self.start_time) * self.bandwidth
        self.slo -= cur_time - self.start_time
        self.start_time = cur_time
        return True if self.model_size > 0 else False
    
    # Check whether we can change the network bandwidth to a new value
    def check_new_bandwidth(self, new_bandwidth: float) -> bool:
        if self.slo * new_bandwidth >= self.model_size:
            return True
    
    def change_bandwidth(self, new_bandwidth: float, cur_time: float):
        self.model_size -= (cur_time - self.start_time) * self.bandwidth
        self.slo -= cur_time - self.start_time
        self.start_time = cur_time
        self.bandwidth = new_bandwidth

'''
A NetManager stores resource information on a machine
'''
class NetManager:
    def __init__(self, node_id: int, bandwidth: int):
        self.node_id = node_id
        self.bandwidth = bandwidth
        self.events = []
        self.event_counter = 0
        self.in_exclusive_status = False     # There is a network event that must occupy full bandwidth
    
    # Check whether we can create a new network event on this node that do not impact slo of existing pods
    def get_avail_net(self, cur_time: float):
        if self.in_exclusive_status:
            return 0
        # 1. Updat event status
        new_events = []
        for event in self.events:
            if event.update_status(cur_time):
                new_events.append(event)
        self.events = new_events
        # 2. Check bandwidth
        num_events = len(self.events)
        if num_events == 0:
            return self.bandwidth
        new_bandwidth = self.bandwidth / (num_events + 1)
        for event in self.events:
            if not event.check_new_bandwidth(new_bandwidth):
                return 0
        return new_bandwidth
    
    '''
    Create a new network event, return event_id
    '''
    def create_net_event(self, model_size: float, slo: float, cur_time: float) -> int:
        # assert not self.in_exclusive_status
        if self.in_exclusive_status:
            print(f"Error: create_net_event called in exclusive status. node = {self.node_id}")
        new_bandwidth = self.bandwidth / (len(self.events) + 1)
        for event in self.events:
            event.change_bandwidth(new_bandwidth, cur_time)
        self.event_counter += 1
        self.events.append(NetworkEvent(self.event_counter, model_size, cur_time, slo, new_bandwidth))
        return self.event_counter
    
    def release_net_event(self, event_id: int):
        for index, event in enumerate(self.events):
            if event.id == event_id:
                self.events.pop(index)
                new_num_event = len(self.events)
                if new_num_event > 0:
                    new_bandwidth = self.bandwidth / new_num_event
                    for event in self.events:
                        event.change_bandwidth(new_bandwidth, time.time())
                break
    
    def check_claim_exclusive(self, cur_time: float) -> bool:
        if self.in_exclusive_status:
            return False
        bandwidth = self.get_avail_net(cur_time)
        if len(self.events) == 0:
            return True
        return False
    
    def claim_exclusive(self):
        # assert not self.in_exclusive_status
        if self.in_exclusive_status:
            print(f"Error: claim_exclusive called in exclusive status. node = {self.node_id}")
        assert len(self.events) == 0
        self.in_exclusive_status = True
    
    def release_full_net(self):
        assert self.in_exclusive_status
        self.in_exclusive_status = False


'''
A ResourceManager stores resource information in the whole cluster
'''
class ResourceManager:
    def __init__(self, core_api, apps_api):
        self.core_api = core_api
        self.apps_api = apps_api

        self.node_list = []
        self.gpu_total_mems = []
        self.rgpus = []
        self.net_managers = []
        self.mems = []
        nodes = get_node_list_with_resource(self.core_api)
        if len(nodes) == 0:
            raise RuntimeError("No available GPU nodes found.")
        print("Total GPU Resouces: ")
        node_id = 0
        for node_name, resource in nodes.items():
            self.node_list.append(node_name)
            self.rgpus.append([resource[1]] * resource[0])
            self.gpu_total_mems.append([resource[1]] * resource[0])
            self.net_managers.append(NetManager(node_id, resource[2]))
            self.mems.append(resource[3])
            print(f"Node {node_id}: {resource[1]} GB * {resource[0]}")
            node_id += 1
        self.get_storage_server_ip()
    
        self.pods = [[] for _ in self.node_list]
        self.counters = [0 for _ in self.node_list]
    
        self.slow_expr = int(os.getenv("SLOW_EXPR", "0"))
    
    def get_storage_server_ip(self):
        self.storage_server_ip = [''] * len(self.node_list)
        pods = self.core_api.list_namespaced_pod(namespace="default", watch=False)
        for pod in pods.items:
            pod_name = pod.metadata.labels["app"]
            if "storage-server-local" in pod_name:
                node_name = pod.spec.node_name
                index = self.node_list.index(node_name)
                self.storage_server_ip[index] = pod.status.pod_ip

    def get_rgpus(self):
        return self.rgpus
    
    def get_rnets(self, cur_time: float):
        return [net_manager.get_avail_net(cur_time) for net_manager in self.net_managers]
    
    def get_node_rnet(self, node_id: int, cur_time: float):
        return self.net_managers[node_id].get_avail_net(cur_time)
    
    '''
    Claim network resources on some nodes, return net event ids
    '''
    def claim_net(self, node_ids: List[int], model_size: int, slo: float, cur_time: float) -> List[int]:
        slo /= 1000.0       # unit: ms -> s
        model_size *= 8.0   # unit: GB -> Gb
        # model weights may not be equally partitioned
        num_nodes = len(node_ids)
        if num_nodes > 1:
            avg_model_size = model_size / num_nodes / 0.9
        else:
            avg_model_size = model_size
        net_events = []
        for node_id in node_ids:
            net_events.append(self.net_managers[node_id].create_net_event(avg_model_size, slo, cur_time))
        return net_events
    
    '''
    Release network usage.
    '''
    def release_net(self, node_ids: List[int], net_event_ids: List[int]):
        for idx in range(len(node_ids)):
            self.net_managers[node_ids[idx]].release_net_event(net_event_ids[idx])

    '''
    Claim whether all nodes have full network resources
    '''
    def try_claim_full_net(self, nodes: List[int]) -> bool:
        cur_time = time.time()
        for node_id in nodes:
            if not self.net_managers[node_id].check_claim_exclusive(cur_time):
                return False
        # Success
        for node_id in nodes:
            self.net_managers[node_id].claim_exclusive()
        return True
    
    '''
    Release the claim of all network resource on a node
    '''
    def release_full_net(self, nodes: List[int]):
        for node_id in nodes:
            self.net_managers[node_id].release_full_net()

    '''
    Claim gpu resource on a node and return the gpu id that claimed.
    '''
    def claim_resource(self, node_id: int, gpu: int) -> int:
        mn_gpu_id = -1
        for index, remain_gpu in enumerate(self.rgpus[node_id]):
            if remain_gpu >= gpu:
                if mn_gpu_id == -1 or remain_gpu < self.rgpus[node_id][mn_gpu_id]:
                    mn_gpu_id = index
        if mn_gpu_id >= 0:
            self.rgpus[node_id][mn_gpu_id] -= gpu

        return mn_gpu_id

    '''
    Claim gpu resource for allocated resources and return resource usages.
    '''
    def claim_allocated_resource(self, allocated_resource: AllocatedResource) -> List[ResourceUsage]:
        resource_usages = []
        for index in range(allocated_resource.num_nodes):
            node_id = allocated_resource.nodes[index]
            pod_name =  "node" + str(node_id) + "-" + str(self.counters[node_id])
            self.counters[node_id] += 1
            # claim resource
            gpu_usage = allocated_resource.gpu_mem[index]
            gpu_id = self.claim_resource(node_id, gpu_usage)
            if gpu_id == -1:
                print(f"resource_manager: claim_resource error, node_id = {node_id}, gpu_usage = {gpu_usage}, rgpus = {self.rgpus[node_id]}")
                return None
            if self.rgpus[node_id][gpu_id] <= 6:
                # If there is only <=6 GB gpu memory remained, use these memory
                # Note that these memory will be added to instance's cost
                gpu_usage += self.rgpus[node_id][gpu_id]
                self.rgpus[node_id][gpu_id] = 0
            resource_usage = ResourceUsage(pod_name, node_id, gpu_id, allocated_resource.gpu_mem[index], gpu_usage)
            resource_usages.append(resource_usage)
            self.pods[node_id].append(pod_name)
        return resource_usages

    '''
    Release gpu resource that some pods use
    '''
    async def release_resource(self, resource_usages: List[ResourceUsage]):
        for resource_usage in resource_usages:
            delete_deployment(self.apps_api, resource_usage.pod_name)

        for resource_usage in resource_usages:
            while not check_deployment_fin(self.apps_api, resource_usage.pod_name):
                await asyncio.sleep(0.1)
            self.rgpus[resource_usage.node_id][resource_usage.gpu_id] += resource_usage.num_gpu_mem
            self.pods[resource_usage.node_id].remove(resource_usage.pod_name)
    
    '''
    Create pods on given nodes and return their ip.
    '''
    async def create_pods(self, model_id: str, allocated_resource: AllocatedResource, resource_usages: List[ResourceUsage]) -> List[str]:
        image = ImageList["vllm"]
        
        pp_size = allocated_resource.num_nodes
        dest_pp_size = [pp_size] * pp_size
        if allocated_resource.will_scale_down:
            for index in range(pp_size):
                if allocated_resource.gpu_mem[index] == allocated_resource.gpu_mem[0]:
                    dest_pp_size[index] = 1
                else:
                    break

        pos = model_id.rfind('/')
        env = {
            "MODEL_ID": model_id[:pos],
            "REMOTE_MODEL_ID": model_id,
            "MODELSCOPE_CACHE": "/mnt/model-cache",
            "PP_SIZE": str(pp_size),
            "ENABLE_PARA": "0",
            "LOCAL_SERVER": "1",
            "PYTHONUNBUFFERED": "1",
            "MAX_MODEL_LEN": "3072",
        }

        if self.slow_expr == 1:
            env["NO_PRE_INIT"] = "1"
            env["FAST_COLDSTART"] = "0"
            env["GPU_MEMORY_UTILIZATION"] = "0.8"
        
        pods_name = []
        for index, resource_usage in enumerate(resource_usages):
            node_id = resource_usage.node_id
            pod_name = resource_usage.pod_name
            gpu_usage = resource_usage.num_gpu_mem
            gpu_id = resource_usage.gpu_id
            # create pod
            env["PP_RANK"] = str(index)
            env["TOTAL_GPU"] = str(gpu_usage)
            env["STORAGE_IP"] = self.storage_server_ip[node_id]
            env["DEST_PP_SIZE"] = str(dest_pp_size[index])
            env["BATCH_SIZE"] = "8"
            core_percentage = int(gpu_usage / self.gpu_total_mems[node_id][gpu_id] * 100)
            core_percentage -= core_percentage % 5      # core percentage should be a multiple of 5
            create_deployment(self.apps_api, pod_name, image, env,
                              node=self.node_list[node_id],
                              nas_path=nas_path,
                              gpu=(gpu_usage, core_percentage))
            pods_name.append(pod_name)
        
        # Get ip list
        ip_list = [None] * allocated_resource.num_nodes
        num_ready = 0
        while num_ready < allocated_resource.num_nodes:
            await asyncio.sleep(0.1)
            pods = self.core_api.list_namespaced_pod(namespace="default", watch=False)
            for pod in pods.items:
                pod_name = pod.metadata.labels["app"]
                if pod_name in pods_name:
                    if isinstance(pod.status.pod_ip, str):
                        rank = pods_name.index(pod_name)
                        if ip_list[rank] is None:
                            ip_list[rank] = pod.status.pod_ip
                            num_ready += 1
        
        return ip_list

    '''
    Initialize pods by visiting xxx/init
    '''
    async def initialize_pods(self, ip_list):
        # Initialize pods
        stime = time.time()
        
        payload = {"ip_list": ip_list}
        pp_size = len(ip_list)
        tasks = []
        for rank in range(pp_size):
            init_url = "http://" + ip_list[rank] + ":8080/init"
            tasks.append(post_request_util_succ_async(init_url, payload))

        await asyncio.gather(*tuple(tasks))
        print(f"initialize pods for {ip_list} time cost = {time.time() - stime} seconds")
