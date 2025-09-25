from typing import Any, Dict, List, Optional, Set, Tuple
import time
import os
import threading
import asyncio
import pickle
import socket

from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sequence import ExecuteModelRequest, SamplerOutput
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        make_async)
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.utils import socket_send, socket_recv

import torch

logger = init_logger(__name__)


class GPUExecutor(ExecutorBase):

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.

        If speculative decoding is enabled, we instead create the speculative
        worker.
        """
        self.pp_rank = int(os.getenv('PP_RANK', '0'))
        self.pp_size = int(os.getenv('PP_SIZE', '1'))
        self.dest_pp_rank = -1
        self.dest_pp_size = int(os.getenv('DEST_PP_SIZE', '1'))
        self.is_running = False
        self.last_request_timer = None
        self.num_processed_request = 0
        self._background_loop_unshielded = None
        self.background_loop = None
        self.invoke_counter = 0
        self.write_counter = 0
        self.dest_write_counter = 0
        self.quit = False

        if self.speculative_config is None:
            self._init_non_spec_worker()
        else:
            self._init_spec_worker()

    def _get_worker_kwargs(
            self,
            local_rank: int = 0,
            rank: int = 0,
            distributed_init_method: Optional[str] = None) -> Dict[str, Any]:
        """Return worker init args for a given rank."""
        if distributed_init_method is None:
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port())
        return dict(
            model_config=self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            device_config=self.device_config,
            cache_config=self.cache_config,
            load_config=self.load_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            lora_config=self.lora_config,
            vision_language_config=self.vision_language_config,
            is_driver_worker=rank == 0,
        )

    def _create_worker(self,
                       local_rank: int = 0,
                       rank: int = 0,
                       distributed_init_method: Optional[str] = None):
        wrapper = WorkerWrapperBase(
            worker_module_name="vllm.worker.worker",
            worker_class_name="Worker",
        )
        wrapper.init_worker(**self._get_worker_kwargs(local_rank, rank,
                                                      distributed_init_method))
        return wrapper.worker

    def _init_non_spec_worker(self):
        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        stime = time.time()
        self.driver_worker = self._create_worker()
        etime_1 = time.time()
        self.driver_worker.init_device()
        etime_2 = time.time()
        self.driver_worker.load_model()
        etime_3 = time.time()
        print(f"GPU executor: init worker time cost: create worker={etime_1-stime} s, init device={etime_2-etime_1} s, load model={etime_3-etime_2} s")

    def init_dest_model_worker(self, ip_list: List[str], origin_ranks):
        print(f"start init_dest_model_worker.")

        self.dest_pp_rank = int(os.getenv('DEST_PP_RANK', '-1'))

        # change num_hidden_layers to new value
        stime = time.time()
        self.model_config.hf_config.num_hidden_layers = int(os.getenv('DEST_NUM_LAYERS', '1'))
        self.driver_worker.model_runner.load_dest_model()
        etime = time.time()
        print(f"init_dest_model_worker: load model time cost = {etime-stime} seconds")

        # init connection before init gpu cache
        self.init_connection(ip_list, origin_ranks, True)
        etime_0 = time.time()
        print(f"init_dest_model_worker: init connection model time cost = {etime_0-etime} seconds")

        # init a new gpu cache
        num_gpu_blocks, _ = self.driver_worker.determine_num_available_blocks()
        num_gpu_blocks = self.aggregate_gpu_blocks(num_gpu_blocks, True)
        print(f"init_dest_model_worker: allocate {num_gpu_blocks} blocks for dest model")
        self.driver_worker.cache_engine.extend_gpu_cache(num_gpu_blocks, "cuda")

        etime_1 = time.time()
        print(f"init_dest_model_worker: extend cache engine time cost = {etime_1-etime} seconds")

        self.dest_inited = True

        etime = time.time()
        print(f"init_dest_model_worker: total time cost = {etime-stime} seconds")

    def _init_spec_worker(self):
        """Initialize a SpecDecodeWorker, using a draft model for proposals.
        """
        assert self.speculative_config is not None

        from vllm.spec_decode.spec_decode_worker import SpecDecodeWorker

        target_worker = self._create_worker()

        draft_worker_kwargs = self._get_worker_kwargs()
        # Override draft-model specific worker args.
        draft_worker_kwargs.update(
            model_config=self.speculative_config.draft_model_config,
            parallel_config=self.speculative_config.draft_parallel_config,
            # TODO allow draft-model specific load config.
            #load_config=self.load_config,
        )

        spec_decode_worker = SpecDecodeWorker.create_worker(
            scorer_worker=target_worker,
            draft_worker_kwargs=draft_worker_kwargs,
        )

        assert self.parallel_config.world_size == 1, (
            "GPUExecutor only supports single GPU.")

        self.driver_worker = spec_decode_worker

        # Load model handled in spec decode worker.
        self.driver_worker.init_device()
    
    def aggregate_gpu_blocks(self, num_gpu_blocks: int, use_dest: bool = False):
        if use_dest:
            if self.dest_pp_size == 1:
                return num_gpu_blocks
            pp_rank = self.dest_pp_rank
            pp_size = self.dest_pp_size
            sockets = self.dest_invoke_sockets
        else:
            if self.pp_size == 1:
                return num_gpu_blocks
            pp_rank = self.pp_rank
            pp_size = self.pp_size
            sockets = self.invoke_sockets
        stime = time.time()
        if pp_rank == 0:
            for stage in range(1, pp_size):
                gpu_blocks_bytes = socket_recv(sockets[stage], num_gpu_blocks.to_bytes(length=8, byteorder='little', signed=False))
                gpu_blocks = int.from_bytes(gpu_blocks_bytes, byteorder='little', signed=False)
                num_gpu_blocks = min(num_gpu_blocks, gpu_blocks)
            print(f"aggregate_gpu_blocks: final num_gpu_blocks = {num_gpu_blocks}")
        else:
            socket_send(sockets[0], num_gpu_blocks.to_bytes(length=8, byteorder='little', signed=False))
        print(f"synchronize num_gpu_blocks time cost = {time.time() - stime} seconds")
        return num_gpu_blocks

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        num_gpu_blocks, num_cpu_blocks = self.driver_worker.determine_num_available_blocks()
        num_gpu_blocks = self.aggregate_gpu_blocks(num_gpu_blocks)
        return (num_gpu_blocks, num_cpu_blocks)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        self.driver_worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        stime = time.time()
        output = self.driver_worker.execute_model(execute_model_req)
        print(f"GPUexecutor: execute_model time cost = {(time.time() - stime) * 1000.0} ms")
        return output

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return self.driver_worker.remove_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.driver_worker.list_loras()

    def check_health(self) -> None:
        # GPUExecutor will always be healthy as long as
        # it's running.
        return
    
    def set_new_timer(self):
        # Reset timer due to the coming of new request
        self.last_request_timer = time.time()
        self.num_processed_request = 0

    def has_requests_in_progress(self):
        # If worker has inference for more than 2 rounds and has been idle for 10 seconds, we deem that there is no requests in progress
        if self.num_processed_request >= 2 and time.time() - self.last_request_timer >= 10:
            return False
        return True

    async def run_engine_loop(self):
        cur_time = time.time()
        print(f"GPU executor event loop started at {cur_time}.")
        # repeatedly read invocations and execute_model

        iteration_counter = 0
        while True:
            if self.quit:
                print(f"GPUexecutor: abort")
                return

            while True:
                output = socket_recv(self.invoke_sockets[0], blocking=False)
                if output is not None:
                    break
                await asyncio.sleep(0)
                if self.quit:
                    print(f"GPUexecutor: abort")
                    return

            stime = time.time()
            print(f"GPUexecutor: wait for receiving request time cost = {(stime - cur_time) * 1000.0} ms")
            
            execute_model_req = pickle.loads(output)

            if isinstance(execute_model_req, List):
                # request for collecting kv cache
                tensors, num_bytes_per_tensor = self.driver_worker.cache_engine.get_gpu_cache(execute_model_req)
                num_tensors = len(tensors)
                socket_send(self.invoke_sockets[0], num_tensors.to_bytes(length=8, byteorder='little', signed=False))
                stime = time.time()
                torch.ops.download_model.send_tensors(tensors, num_bytes_per_tensor, self.invoke_sockets[0].fileno())
                print(f"migrate kv cache: send tensors total time cost = {(time.time() - stime) * 1000} ms")
                continue

            use_dest = execute_model_req.use_dest
            
            etime = time.time()
            print(f"invoke: received {len(output)} bytes, process time cost = {(etime - stime) * 1000.0} ms")

            if use_dest and self.dest_pp_rank == -1:
                # do not involve in the calculation as we are not one of dest model
                print("GPUexecutor: execute_model_req use dest model, skip")
                continue

            output = self.execute_model(execute_model_req)

            if use_dest and self.dest_pp_rank == self.dest_pp_size - 1 \
                or not use_dest and self.pp_rank == self.pp_size - 1:
                    stime = time.time()

                    byte_string = pickle.dumps(output, protocol=-1)
                    socket_send(self.comm_sockets[0] if not use_dest else self.dest_comm_sockets[0], byte_string)

                    etime = time.time()
                    print(f"result: send {len(byte_string)} bytes, time cost = {(etime - stime) * 1000.0} ms")
            
            cur_time_1 = time.time()
            print(f"iteration [{iteration_counter}] finished, time cost = {(cur_time_1 - cur_time) * 1000.0} ms")
            iteration_counter += 1
            cur_time = cur_time_1
            self.last_request_timer = cur_time
            self.num_processed_request += 1

    def _call_back(self, context):
        print("Task completion received...")
        print("Name of the task:%s"%context.get_name())
        print("Wrapped coroutine object:%s"%context.get_coro())
        print("Task is done:%s"%context.done())
        print("Task has been cancelled:%s"%context.cancelled())
        print("Task result:%s"%context.result())
        print(type(context))
        print(context)

    def start_running(self):
        if self.is_running:
            print("worker has started running, skip initialization")
            return

        self.is_running = True
        self._background_loop_unshielded = asyncio.get_event_loop().create_task(self.run_engine_loop())
        self._background_loop_unshielded.add_done_callback(self._call_back)
        self.background_loop = asyncio.shield(self._background_loop_unshielded)
        print(f"background loop started at {time.time()}")

    def init_connection(self, ip_list: List[str], origin_ranks: List[int] = [], use_dest: bool = False):
        my_rank = self.pp_rank if not use_dest else self.dest_pp_rank
        num_client = self.pp_size if not use_dest else self.dest_pp_size
        if num_client == 1:
            return
        invoke_sockets = [None for _ in range(num_client)]
        comm_sockets = [None for _ in range(num_client)]

        print("start init connection")
        stime = time.time()

        # invocation sockets
        if not use_dest:
            if my_rank == 0:
                # we are server
                server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)
                server_socket.bind((ip_list[0], 8090 + (1 if use_dest else 0)))
                server_socket.listen(num_client - 1)
                for _ in range(num_client - 1):
                    conn, address = server_socket.accept()
                    rank = ip_list.index(address[0])
                    invoke_sockets[rank] = conn
            else:
                # we are client
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)
                while True:
                    try:
                        client_socket.connect((ip_list[0], 8090 + (1 if use_dest else 0)))
                        break
                    except Exception as e:
                        pass
                invoke_sockets[0] = client_socket
        else:
            # reuse previous invoke_sockets
            for i in range(1, num_client):
                invoke_sockets[i] = self.invoke_sockets[origin_ranks[i]]
                

        # communication sockets
        # 0 -> num_client - 1
        if my_rank == 0:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)
            while True:
                try:
                    client_socket.connect((ip_list[num_client - 1], 8092 + (1 if use_dest else 0)))
                    break       # break when successfully connected
                except Exception as e:
                    pass
            comm_sockets[num_client - 1] = client_socket

        # num_client - 1 accepts 0, and i accepts i+1 for i < num_client -1
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)
        server_socket.bind((ip_list[my_rank], 8092 + (1 if use_dest else 0)))
        server_socket.listen(1)
        conn, address = server_socket.accept()
        rank = ip_list.index(address[0])
        if rank != (my_rank + 1) % num_client:
            print(f"op_ip = {address[0]}, expected = {(my_rank + 1) % num_client}")
            raise ValueError("Received unexpected connection request")
        comm_sockets[rank] = conn
        
        # i -> i-1 for i > 0
        if my_rank > 0:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)
            while True:
                try:
                    client_socket.connect((ip_list[my_rank - 1], 8092 + (1 if use_dest else 0)))
                    break       # break when successfully connected
                except Exception as e:
                    pass
            comm_sockets[my_rank - 1] = client_socket

        if use_dest:
            self.dest_invoke_sockets = invoke_sockets
            self.dest_comm_sockets = comm_sockets
            self.driver_worker.model_runner.dest_model.set_socket(None if my_rank == 0 else comm_sockets[my_rank-1],
                                                                  None if my_rank == num_client - 1 else comm_sockets[my_rank+1])
        else:
            self.invoke_sockets = invoke_sockets
            self.comm_sockets = comm_sockets
            self.driver_worker.model_runner.model.set_socket(None if my_rank == 0 else comm_sockets[my_rank-1],
                                                             None if my_rank == num_client - 1 else comm_sockets[my_rank+1])
        
        print(f"init connection time cost = {time.time() - stime} seconds")

class GPUExecutorAsync(GPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        stime = time.time()
        output = await make_async(self.driver_worker.execute_model
                                )(execute_model_req=execute_model_req, )
        print(f"GPUexecutor: execute_model time cost = {(time.time() - stime) * 1000.0} ms")
        return output
