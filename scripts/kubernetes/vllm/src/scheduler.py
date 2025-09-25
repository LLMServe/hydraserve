import os
import sys
import time
import json
import math
import traceback
from collections import deque
import asyncio
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional

from engine import ChatEngine
from resource_manager import ResourceUsage, ResourceManager, AllocatedResource
from cache_manager import CacheManager
from utils import post_request_util_succ_async, post_ayncio_request, post_request_async

KVCacheRequiredSize = 5

@dataclass
class ModelStats:
    model_size: float
    # slo
    ttft_slo: float                    # hard limit, all ttft for each request must smaller than given slo
    tpot_slo: float                    # hard limit, all average tpot for each request must smaller than given slo
    cost_slo: float                    # soft limit, average cost_per_token must smaller than given slo
    # profile information
    num_profile_points: int = 0      # number of profile points till now
    tpot: float = 0                  # tpot for pp_size=1
    prefill_time: float = 0          # prefill time cost for pp_size=1
    # cost record
    cur_gpu_usage: int = 0           # current gpu memory usage
    cur_usage_start_time: float = 0  # when did the model obtained those gpu
    past_cost: float = 0             # cost of killed instances
    num_output_tokens: int = 0       # number of output tokens till now

    def get_cur_cost(self) -> float:
        cur_time = time.time()
        return self.past_cost + self.cur_gpu_usage * (cur_time - self.cur_usage_start_time) * 1000.0

    def add_gpu_mem(self, resource_usages: List[ResourceUsage]):
        cur_time = time.time()
        num_gpu_mem = 0
        for resource_usage in resource_usages:
            num_gpu_mem += resource_usage.alloc_gpu_mem
        if self.cur_gpu_usage == 0:
            self.cur_gpu_usage = num_gpu_mem
            self.cur_usage_start_time = cur_time
        else:
            self.cur_gpu_usage += num_gpu_mem
            self.past_cost -= num_gpu_mem * (cur_time - self.cur_usage_start_time) * 1000.0
    
    def remove_gpu_mem(self, resource_usages: List[ResourceUsage]):
        cur_time = time.time()
        num_gpu_mem = 0
        for resource_usage in resource_usages:
            num_gpu_mem += resource_usage.alloc_gpu_mem
        self.cur_gpu_usage -= num_gpu_mem
        self.past_cost += num_gpu_mem * (cur_time - self.cur_usage_start_time) * 1000.0


instance_list = []

'''
A Instance is some pods that use pipeline parallel to serve a model.
'''
class Instance:
    InstanceId: int = 0

    def __init__(self, model, pp_size: int, will_scale_down: bool, cur_time: float, ip_list: Optional[List[str]] = None, resource_usages: Optional[List[ResourceUsage]] = None):
        self.model = model
        self.model_id = model.model_id
        self.model_batch_size = model.batch_size
        self.batch_size = self.model_batch_size * pp_size
        self.num_reqs = 0
        self.num_coldstart_reqs = 0
        self.last_chat_time = cur_time
        self.pp_size = pp_size
        self.id = Instance.InstanceId
        Instance.InstanceId += 1
        self.will_scale_down = will_scale_down
        self.in_scaling_down = False
        self.child = []
        self.num_child = 0
        if resource_usages is not None:
            self.init(ip_list, resource_usages)
        else:
            self.inited = False

    def init(self, ip_list: List[str], resource_usages: List[ResourceUsage], cur_time: float = None):
        self.pp_size = len(ip_list)
        self.batch_size = self.model_batch_size * self.pp_size
        self.ip_list = ip_list
        self.resource_usages = resource_usages
        self.chat_engine = ChatEngine(ip_list[0], self.model_id)
        if cur_time is not None:
            self.last_chat_time = cur_time
        self.inited = True

    def add_child(self, instance):
        self.child.append(instance)
        self.num_child += 1
    
    def check_can_scale_down(self) -> bool:
        if self.inited and self.pp_size > 1 and not self.in_scaling_down and self.will_scale_down:
            return True
        return False

    def resp_generator(self, request_id: int, chat, is_coldstart_req: bool):
        stime = time.time()
        profile = False
        is_first_resp = True
        if not is_coldstart_req:
            profile = True
            origin_num_points = self.model.model_stats.num_profile_points
        for stream_response in chat:
            if stream_response.choices[0].delta.content is not None:
                if profile and is_first_resp:
                    prefill_elapsed_time = (time.time() - stime) * 1000.0
                    self.model.model_stats.prefill_time = (origin_num_points * self.model.model_stats.prefill_time + prefill_elapsed_time) / (origin_num_points + 1)
                    is_first_resp = False
                yield stream_response.choices[0].delta.content
        if stream_response.usage is None:
            print(f"Error: stream_response final return has null usage")
        else:
            # produce num_completion_tokens
            yield f"#{stream_response.usage.completion_tokens}"
            self.model.model_stats.num_output_tokens += stream_response.usage.completion_tokens
            elapsed_time = (time.time() - stime) * 1000.0
            if profile:
                if stream_response.usage.completion_tokens > 1:
                    new_tpot = (elapsed_time - prefill_elapsed_time) / (stream_response.usage.completion_tokens - 1)
                    self.model.model_stats.tpot = (origin_num_points * self.model.model_stats.tpot + new_tpot) / (origin_num_points + 1)
                self.model.model_stats.num_profile_points += 1
            print(f"Request [{request_id}] generation finished, #Tokens = {stream_response.usage.completion_tokens}, elapsed {'%.1f' % (elapsed_time / 1000.0)} s")

        self.num_reqs -= 1
        if is_coldstart_req:
            self.num_coldstart_reqs -= 1
        self.last_chat_time = time.time()

    async def chat(self, request_id: int, prompt: Union[List, str], stream: Optional[bool] = False):
        # Count num_coldstart_reqs whenever it is a coldstart instance because we should avoid the temporary workers being shutdown before this request complete.
        self.num_coldstart_reqs += 1
        while not self.inited:
            await asyncio.sleep(0)
        is_coldstart_req = True if self.pp_size > 1 else False
        if not is_coldstart_req:
            self.num_coldstart_reqs -= 1
        try:
            reply = await asyncio.wait_for(self.chat_engine.chat(prompt, stream), timeout=5)
        except Exception as e:
            exc_info = sys.exc_info()
            print(f"Request [{request_id}] chat gets exception: {e}")
            print("".join(traceback.format_exception(*exc_info)))
        if not stream:
            assert isinstance(reply, str)
            self.num_reqs -= 1
            if is_coldstart_req:
                self.num_coldstart_reqs -= 1
            self.last_chat_time = time.time()
            return reply
        else:
            return self.resp_generator(request_id, reply, is_coldstart_req)

class Request:
    def __init__(self, request_id: int, model_id: str, prompt: Union[List, str], stream: bool, cur_time: float):
        self.request_id = request_id
        self.model_id = model_id
        self.prompt = prompt
        self.stream = stream
        self.arrival_time = cur_time
        self.resp_func = None
        self.resp_event = asyncio.Event()
    
    def allocate_instance(self, instance: Instance):
        print(f"request [{self.request_id}] allocated to instance [{instance.id}], waiting time = {'%.1f' % (time.time() - self.arrival_time)} seconds")
        instance.num_reqs += 1
        self.resp_func = instance.chat(self.request_id, self.prompt, stream=self.stream)
        self.resp_event.set()

'''
A Model contains all instances that serve the model.
'''
class Model:
    def __init__(self, model_id: str, model_info: Tuple[int, int, int, int, int], batch_size: int = 8):
        self.model_id = model_id
        self.model_stats = ModelStats(
            model_size=model_info[0],
            ttft_slo=model_info[1],
            tpot_slo=model_info[2],
            cost_slo=model_info[3])
        self.instances = []
        self.batch_size = batch_size
        self.waiting_queue = deque()
        self.new_requests_event = asyncio.Event()
        self.loop_started = False
        self.history_request_times = deque()
        # We control number of instances based on the predicted number of requests in the following $autoscaling_time_window seconds
        self.autoscaling_time_window = 1
    
    # Check available instances and serve requests in waiting queue. Return number of un-initialized instances.
    async def check_avail_instance(self) -> int:
        new_instance_list = []
        avail_instance = None
        num_uninited_capabilities = 0
        instances = self.instances
        for instance in instances:
            if instance.num_reqs == -1:
                # instance has been deleted
                self.instances.remove(instance)
                continue
            if instance.inited:
                max_served_reqs = instance.batch_size - (instance.num_reqs + (1 if not instance.inited else 0))
                if max_served_reqs > 0:
                    has_reqs = await self.serve_request(instance, max_served_reqs)
                    if not has_reqs:
                        break
            else:
                num_uninited_capabilities += instance.batch_size
        return num_uninited_capabilities
    
    # Add a initialized instance
    def add_instance(self, ip_list: List[str], resource_usages: List[ResourceUsage], cur_time: float, will_scale_down: bool = False) -> Instance:
        global instance_list
        new_instance = Instance(self, len(ip_list), will_scale_down, cur_time, ip_list, resource_usages)
        self.instances.append(new_instance)
        instance_list.append(new_instance)
        return new_instance
    
    def create_instance(self, pp_size: int, cur_time: float, will_scale_down: bool = False) -> Instance:
        global instance_list
        new_instance = Instance(self, pp_size, will_scale_down, cur_time)
        self.instances.append(new_instance)
        instance_list.append(new_instance)
        return new_instance
    
    def add_request(self, request: Request):
        self.history_request_times.append(request.arrival_time)
        self.waiting_queue.append(request)
        self.new_requests_event.set()
    
    # Serve the earliest request in waiting queue with a instance, return true if there still have requests unserved
    async def serve_request(self, instance: Instance, max_served_reqs: int) -> bool:
        # we add up the instance's request counter in prior to prevent it from being deleted by scheduler
        instance.num_reqs += 1
        while self.waiting_queue and max_served_reqs > 0:
            request = self.waiting_queue.popleft()
            request.allocate_instance(instance)
            max_served_reqs -= 1
        instance.num_reqs -= 1
        if self.waiting_queue:
            return True
        return False

class Scheduler:
    def __init__(self, core_api, apps_api, models: Dict[str, Dict[str, Tuple[int, int, int, int, int]]]):
        self.resource_manager = ResourceManager(core_api, apps_api)
        self.model_list = []
        self.models = []
        model_cur_index = {}
        for task_type, model_dict in models.items():
            for model_id, model_info in model_dict.items():
                if model_id not in model_cur_index:
                    model_cur_index[model_id] = 0
                    cur_index = 0
                else:
                    cur_index = model_cur_index[model_id]
                for index in range(model_info[4]):
                    split_model_id = model_id + "/" + str(index + cur_index)
                    self.model_list.append(split_model_id)
                    self.models.append(Model(split_model_id, model_info))
                model_cur_index[model_id] += model_info[4]
        self.use_cache = int(os.getenv("USE_CACHE", "0"))
        if self.use_cache:
            self.cache_manager = CacheManager(self.resource_manager.storage_server_ip)

        self.slow_expr = int(os.getenv("SLOW_EXPR", "0"))
        self.use_static_parallelism = False
        self.parallelism_size = -1

        from algs import naive, pipeline_parallel_static, pipeline_parallel_static_scale_down, ours

        if self.slow_expr:
            self.alg = naive
        else:
            max_pp_size = int(os.getenv("MAX_PP_SIZE", "-1"))
            if max_pp_size != -1:
                # max_pp_size is configured in experiment 1.1
                self.use_static_parallelism = True
                self.parallelism_size = max_pp_size
                self.alg = pipeline_parallel_static
            else:
                self.alg = ours
        print(f"Use algorithm = {self.alg}")
        self.num_instance_per_coldstart = 4
        self.tasks = set()
    
    def spawn_new_task(self, func):
        task = asyncio.create_task(func)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def check_instance_loop(self, grace_period: int):
        global instance_list
        loop = asyncio.get_event_loop()
        while True:
            cur_time = time.time()
            new_instance_list = []
            for instance in instance_list:
                if instance.inited and instance.num_reqs == 0 and cur_time - instance.last_chat_time > grace_period:
                    if instance.in_scaling_down and instance.num_child > 0:
                        # For instance that has children, do not kill it during scaling down
                        continue
                    # TODO (BUGFIX): delete a instance that is scaling down will cause exception in aiohttp.
                    # delete this instance
                    instance.num_reqs = -1
                    instance.model.model_stats.remove_gpu_mem(instance.resource_usages)
                    self.spawn_new_task(self.resource_manager.release_resource(instance.resource_usages))
                    print(f"Deleting instance [{instance.id}] for model {instance.model_id}...")
                else:
                    if instance.check_can_scale_down():
                        # try to scale down the instance
                        self.spawn_new_task(self.scale_down(instance, cur_time))
                    new_instance_list.append(instance)
            instance_list = new_instance_list

            print(f"Current GPU Resources: {self.resource_manager.get_rgpus()}")

            await asyncio.sleep(1)
    
    async def check_model_loop(self, model: Model):
        while True:
            if not model.waiting_queue:
                # model.new_requests_event.clear()
                # TODO(BUGFIX): use `model.new_requests_event.clear()` leads to loop not be awakened when it is setted.
                model.new_requests_event = asyncio.Event()
                await model.new_requests_event.wait()

            # Check whether incoming request can be served by existing instances
            num_uninited_capabilities = await model.check_avail_instance()
            num_request = len(model.waiting_queue) - num_uninited_capabilities
            if num_request > 0:
                # We need to start new instance
                # 1. Perform autoscaling strategy
                cur_time = time.time()
                min_time = cur_time - model.autoscaling_time_window
                while len(model.history_request_times) > 0 and model.history_request_times[0] < min_time:
                    model.history_request_times.popleft()
                num_request_in_next_window = num_request + len(model.history_request_times)
                num_new_instance = math.ceil(num_request / model.batch_size)
                # 2. Try to initialize new instances
                max_waiting_time = cur_time - model.waiting_queue[0].arrival_time
                left_num_instance = num_new_instance
                coldstart_tasks = []
                cache_tasks = []
                use_cache = False
                # 2.1. Check cache
                if self.use_cache:
                    avail_nodes = await self.cache_manager.query_model_cache(model.model_id)
                    require_gpu_size = math.ceil(model.model_stats.model_size + KVCacheRequiredSize)
                    rgpus = self.resource_manager.get_rgpus()
                    rgpus = [max(rgpu) for rgpu in rgpus]
                    for node_id in avail_nodes:
                        cache_hit = False
                        if rgpus[node_id] >= require_gpu_size:
                            cache_hit = True
                            allocated_resource = AllocatedResource(1, [node_id], [require_gpu_size], [0])
                            print(f"Cache Hit! model = {model.model_id}, node = {node_id}, required_size = {require_gpu_size}")
                        if cache_hit:
                            resource_usages = self.resource_manager.claim_allocated_resource(allocated_resource)
                            if resource_usages is None:
                                # ERROR
                                while resource_usages is None:
                                    await asyncio.sleep(1)
                                    resource_usages = self.resource_manager.claim_allocated_resource(allocated_resource)
                            cache_tasks.append(asyncio.create_task(self.create_cold_start_instance(model, 1, cur_time, max_waiting_time * 1000.0, allocated_resource, resource_usages)))
                            left_num_instance -= 1
                            if left_num_instance == 0:
                                break
                # 2.2. Create coldstart instance
                num_instance_per_coldstart = self.num_instance_per_coldstart
                for i in range(0, left_num_instance, num_instance_per_coldstart):
                    required_pp_size = min(left_num_instance - i, num_instance_per_coldstart)
                    coldstart_tasks.append(asyncio.create_task(self.create_cold_start_instance(model, required_pp_size, cur_time, max_waiting_time * 1000.0)))
                for task in cache_tasks:
                    instance = await task
                    await model.check_avail_instance()      # allocate requests in waiting queue to the instance
                for task in coldstart_tasks:
                    instance = await task
                    if instance is not None:
                        left_num_instance -= instance.pp_size
                        await model.check_avail_instance()      # allocate requests in waiting queue to the instance
                if left_num_instance < num_new_instance and left_num_instance > 0:
                    print(f"Warning: Model {model.model_id} wanted to create {num_new_instance} instances and still has {left_num_instance} instances to be created")

            await asyncio.sleep(0)

    '''
    Post request to a instance to perform scale down.
    '''
    async def perform_scale_down(self, instance: Instance, orig_instance: Instance, orig_rank: int, cur_time: float):
        # wait util node has enough network
        stime = cur_time
        master_node_id = orig_instance.resource_usages[orig_rank].node_id
        unloaded_model_size = instance.model.model_stats.model_size * (instance.pp_size - 1) / instance.pp_size
        while instance.num_reqs != -1:
            ret = self.resource_manager.get_node_rnet(master_node_id, cur_time)
            if ret > 0:
                net_event_id = self.resource_manager.claim_net([master_node_id], unloaded_model_size, 1, cur_time)[0]
                break
            await asyncio.sleep(0.1)
            cur_time = time.time()
        
        print(f"scheduler: instance [{instance.id}] starts to load dest at {cur_time - stime} s")

        if instance.num_reqs == -1:
            # instance has been deleted
            return

        # 1. load model in local storage server
        req_header = {"model": instance.model_id, "pp_rank": orig_rank, "pp_size": instance.pp_size, "is_dest": True, "pre_load": "yes"}
        req_header_bytes = json.dumps(req_header).encode('utf-8')
        req_header_size = len(req_header_bytes)
        req_header_size_bytes = req_header_size.to_bytes(length=8, byteorder='little', signed=False)
        self.spawn_new_task(post_ayncio_request(self.resource_manager.storage_server_ip[master_node_id], 6666, req_header_size_bytes + req_header_bytes))

        # 2. post init_dest request to vllm worker
        instance_ip = orig_instance.ip_list[orig_rank]
        init_dest_url = "http://" + instance_ip + ":8080/utils/dest"
        payload = {"dest_pp_rank": "0",
                   "ip_list": [instance_ip],
                   "origin_ranks": {"0": str(orig_rank)}}
        try:
            await post_request_async(init_dest_url, payload)
        except Exception as e:
            print(f"post_request_async in scale down for instance [{instance.id}] meets exception: {e}")

        self.resource_manager.release_net([master_node_id], [net_event_id])

        etime = time.time()
        print(f"scheduler: instance [{instance.id}] load dest time cost = {etime - cur_time} s")

        if instance.num_reqs == -1:
            # instance has been deleted
            return

        # 3. migrate kv cache
        if orig_rank == 0:
            # Wait util all cold start requests finish or we can migrate kv cache.
            # Note that the requests that arrive at engine now is tagged as 'use_dest', and it will be scheduled in a different group from no_dest requests.
            # Other stages that will load dest model wait for the head to finish migrating kv cache, then start to serve new requests.
            success = False
            nodes = [resource_usage.node_id for resource_usage in instance.resource_usages]
            nodes = nodes.copy()
            while instance.num_coldstart_reqs > 0:
                # Only when the number of coldstart requests is reduced to model_batch_size, starts to migrate kv cache.
                success = self.resource_manager.try_claim_full_net(nodes)
                if success:
                    break
                await asyncio.sleep(0.05)
            etime_1 = time.time()
            print(f"scheduler: instance [{instance.id}] wait for claim all net resource time cost = {etime_1 - etime} s")
            if success:
                # send migrate kv cache request
                stime = time.time()
                dest_url = "http://" + instance.ip_list[0] + ":8080/utils/migration"
                payload = {"num_coldstart_reqs": str(instance.num_coldstart_reqs)}
                await post_request_async(dest_url, payload)
                self.resource_manager.release_full_net(nodes)
            print(f"scheduler: instance [{instance.id}] perform kv migration time cost = {time.time() - etime_1} s")

        # 4. set pp_size, batch_size
        if orig_rank == 0:
            if instance.num_reqs == -1:
                # instance has been deleted
                return
            instance.pp_size = 1
            instance.batch_size = instance.model_batch_size

        # wait util the scale down for original instance finished due to kv cache migration
        while orig_instance.num_reqs != -1 and orig_instance.pp_size > 1:
            await asyncio.sleep(1)

        if orig_rank > 0:
            # post request to shutdown initial worker
            init_dest_url = "http://" + instance_ip + ":8080/utils/stop_origin_serve"
            payload = {"shutdown": "true"}
            await post_request_async(init_dest_url, payload)
            etime = time.time()
            instance_ip = orig_instance.ip_list[orig_rank]
            instance.init([instance_ip], [orig_instance.resource_usages[orig_rank]], etime)
            print(f"scheduler: create new instance [{instance.id}] from [{orig_instance.id}], time cost = {'%.1f' % (etime - cur_time)} seconds")
        else:
            print(f"scheduler: scale down instance [{orig_instance.id}], time cost = {'%.1f' % (time.time() - cur_time)} seconds")


    '''
    Scale down a instance with pp_size > 1.
    '''
    async def scale_down(self, instance: Instance, cur_time: float):
        instance.in_scaling_down = True
        
        model = instance.model
        resource_usages_to_release_idx = []
        resource_usages_to_release = []
        new_tasks = []
        for index in range(1, instance.pp_size):
            if instance.resource_usages[index].alloc_gpu_mem == instance.resource_usages[0].alloc_gpu_mem:
                newInstance = model.create_instance(instance.pp_size, cur_time)
                instance.add_child(newInstance)
                new_tasks.append(self.spawn_new_task(self.perform_scale_down(newInstance, instance, index, cur_time)))
            else:
                resource_usages_to_release_idx.append(index)
                resource_usages_to_release.append(instance.resource_usages[index])
        
        await self.perform_scale_down(instance, instance, 0, cur_time)

        if instance.num_reqs == -1:
            # instance has been deleted
            return
        
        # Delete these resources from instance resource_usages to avoid instance getting killed while scaling down
        for index in resource_usages_to_release_idx:
            instance.resource_usages[index] = None
        instance.model.model_stats.remove_gpu_mem(resource_usages_to_release)
        await self.resource_manager.release_resource(resource_usages_to_release)

        for task in new_tasks:
            await task

        instance.ip_list = instance.ip_list[:1]
        instance.resource_usages = instance.resource_usages[:1]
        instance.in_scaling_down = False
    
    '''
    Create a new cold start instance for model_id and return ip list
    '''
    async def create_cold_start_instance(self, model: Model, required_pp_size: int, start_time: float, max_waiting_time: float, allocated_resource: AllocatedResource = None, resource_usages: List[ResourceUsage] = None) -> Instance:
        # First, check whether the model has available cache
        use_cache = False
        if allocated_resource is not None:
            use_cache = True
        if not use_cache:
            # Perform scheduling algorithm
            rgpus = self.resource_manager.get_rgpus()
            rgpu = [max(rgpu) for rgpu in rgpus]
            rnets = self.resource_manager.get_rnets(start_time)
            if self.use_static_parallelism:
                allocated_resource = self.alg(rgpu, rnets, model.model_stats, required_pp_size, max_waiting_time, self.parallelism_size)
            else:
                allocated_resource = self.alg(rgpu, rnets, model.model_stats, required_pp_size, max_waiting_time)
            if allocated_resource.num_nodes == 0:
                return None
            print(f"Current GPU Resouces: {rgpus}")
            print(f"Rnets = {rnets}")
            print(f"Model {model.model_id} allocated to {allocated_resource.nodes}")
        
        etime = time.time()
        instance = model.create_instance(allocated_resource.num_nodes, etime, allocated_resource.will_scale_down)
        allocate_time = etime - start_time

        if not use_cache:
            # Claim network usage. Note that we set model pulling's SLO to (TTFT_SLO - 2000) ms
            net_events = self.resource_manager.claim_net(allocated_resource.nodes, model.model_stats.model_size, model.model_stats.ttft_slo - max_waiting_time - 2000, etime)
        if not use_cache and self.slow_expr == 0:
            # Prefetch: inform local model server to download model
            pp_size = allocated_resource.num_nodes
            loop = asyncio.get_event_loop()
            for rank in range(pp_size):
                req_header = {"model": model.model_id, "pp_rank": rank, "pp_size": pp_size, "is_dest": False, "pre_load": "yes"}
                req_header_bytes = json.dumps(req_header).encode('utf-8')
                req_header_size = len(req_header_bytes)
                req_header_size_bytes = req_header_size.to_bytes(length=8, byteorder='little', signed=False)
                self.spawn_new_task(post_ayncio_request(self.resource_manager.storage_server_ip[allocated_resource.nodes[rank]], \
                                                                    6666, req_header_size_bytes + req_header_bytes))
        if not use_cache and self.use_cache and allocated_resource.num_nodes == 1:
            self.cache_manager.add_cache(model.model_id, allocated_resource.nodes[0])
        
        if not use_cache:
            resource_usages = self.resource_manager.claim_allocated_resource(allocated_resource)
            if resource_usages is None:
                # ERROR
                while resource_usages is None:
                    await asyncio.sleep(1)
                    resource_usages = self.resource_manager.claim_allocated_resource(allocated_resource)
        pods_name = [usage.pod_name for usage in resource_usages]

        ip_list = await self.resource_manager.create_pods(model.model_id, allocated_resource, resource_usages)

        etime_1 = time.time()
        create_pod_time = etime_1 - etime

        await self.resource_manager.initialize_pods(ip_list)

        if not use_cache:
            self.resource_manager.release_net(allocated_resource.nodes, net_events)
        cur_time = time.time()
        instance.init(ip_list, resource_usages, cur_time)
        model.model_stats.add_gpu_mem(resource_usages)

        print(f"scheduler: create instance [{instance.id}] for {model.model_id}, allocate_time = {'%.1f' % allocate_time} s, create_pod_time = {'%.1f' % create_pod_time} s, init_pod_time = {'%.1f' % (cur_time - etime_1)} s")
        print(f"pods: {pods_name}")
        allocated_resource.print_info(model)

        slo = (model.model_stats.ttft_slo - max_waiting_time) / 1000.0 - 2
        print(f"scheduler: model {model.model_id} coldstart time = {'%.1f' % (cur_time - start_time)} s, slo = {'%.1f' % slo}")
        if cur_time - start_time > slo:
            print(f"scheduler: model {model.model_id} violated slo!")

        return instance
    
    async def process_request(self, request_id: int, model_id: str, prompt: Union[List, str], stream: Optional[bool] = False):
        stime = time.time()
        request = Request(request_id, model_id, prompt, stream, stime)
        index = self.model_list.index(model_id)
        model = self.models[index]
        
        if not model.loop_started:
            model.loop_started = True
            self.spawn_new_task(self.check_model_loop(model))
        model.add_request(request)

        await request.resp_event.wait()
        return await request.resp_func

