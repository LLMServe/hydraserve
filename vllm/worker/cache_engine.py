"""CacheEngine class for managing the KV cache."""
from typing import Dict, List

import os
import ctypes
import pickle
import time
import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, is_pin_memory_available

from typing import Optional

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(model_config.dtype)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(self.num_gpu_blocks, "cuda")
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

        # for dest model, it has an unique dest_gpu_cache
        self.dest_gpu_cache = None

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_layers):
            kv_cache.append(
                torch.empty(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        return kv_cache
    
    def get_gpu_cache(
        self,
        block_ids: List[int]
    ) -> bytes:
        """Get GPU KV cache at some given places"""
        prev_num_layers = int(os.getenv('PREV_NUM_LAYERS', '0'))
        transfer_layers = prev_num_layers if prev_num_layers > 0 else self.num_layers
        tensors = []
        stime = time.time()
        for index in range(transfer_layers):
            for id in block_ids:
                tensor = torch.select(self.gpu_cache[index], dim=1, index=id)
                tensors.append(tensor)
        '''
        final_tensor = torch.stack(tensors, dim=0)
        etime = time.time()
        print(f"get_gpu_cache: prepare time cost = {(etime - stime) * 1000.0} ms")
        final_tensor = final_tensor.to("cpu")
        etime_1 = time.time()
        print(f"get_gpu_cache: transfer time cost = {(etime_1 - etime) * 1000.0} ms")
        num_bytes = final_tensor.element_size() * final_tensor.nelement()
        byte_string = ctypes.string_at(final_tensor.data_ptr(), num_bytes)
        print(f"get_gpu_cache: process time cost = {(time.time() - etime_1) * 1000.0} ms")
        print(f"get_gpu_cache: total tensor in bytes: {len(byte_string) / 1024.0 / 1024.0} MB")
        return byte_string
        '''
        num_bytes_per_tensor = tensors[0].element_size() * tensors[0].nelement()
        return tensors, num_bytes_per_tensor
    
    # TODO: reuse get_gpu_cache() here
    def get_dest_gpu_cache(
        self,
        start_layer,
        num_layers,
        block_ids: List[int],
        # layers: bytes
    ):
        '''
        tensor_like = torch.select(self.gpu_cache[0], dim=1, index=0)
        num_bytes = tensor_like.element_size() * tensor_like.nelement()
        num_bytes_per_layer = num_bytes * len(block_ids)
        num_layers = len(layers) // num_bytes_per_layer
        end_layer = start_layer + num_layers
        shape = list(tensor_like.shape)
        shape[0] *= len(block_ids)
        num_element_per_layer = tensor_like.nelement() * len(block_ids)
        cur_start_pos = 0
        for i in range(start_layer, end_layer):
            input_tensor = torch.frombuffer(layers, dtype=tensor_like.dtype, offset=cur_start_pos, count=num_element_per_layer).view(shape).to(tensor_like.device)
            block_tensors = input_tensor.chunk(len(block_ids), dim=0)
            for index, id in enumerate(block_ids):
                self.gpu_cache[i][:,id] = block_tensors[index]
            cur_start_pos += num_bytes_per_layer
        assert cur_start_pos == len(layers)
        return num_layers
        '''
        end_layer = start_layer + num_layers
        origin_tensors = []
        for i in range(start_layer, end_layer):
            for id in block_ids:
                origin_tensors.append(torch.select(self.gpu_cache[i], dim=1, index=id))
        num_bytes_per_tensor = origin_tensors[0].element_size() * origin_tensors[0].nelement()
        return origin_tensors, num_bytes_per_tensor

    # For dest model: extend gpu cache to num_blocks
    def extend_gpu_cache(
        self,
        num_blocks: int,
        device: str,
    ):
        """Extend KV cache."""
        if num_blocks < self.num_gpu_blocks:
            raise ValueError("Current gpu blocks is more than expected.")
        num_blocks -= self.num_gpu_blocks

        num_layers = self.model_config.get_num_layers(self.parallel_config)
        if num_layers > self.num_layers:
            print(f"extend_gpu_cache: add extra {num_layers - self.num_layers} layers")
            os.environ["PREV_NUM_LAYERS"] = str(self.num_layers)
            self.num_layers = num_layers

        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_heads, self.head_size)
        # determine dim
        dim = 1
        for index, value in enumerate(kv_cache_shape):
            if value == num_blocks:
                dim = index
        if dim != 1:
            print(f"Warning: get_gpu_cache and copy_gpu_cache assumes the block dim is 1. However, the actual dim is {dim}. This will cause cache data error.")
        if num_blocks > 0:
            for i in range(len(self.gpu_cache)):
                self.gpu_cache[i] = torch.cat([self.gpu_cache[i], torch.empty(kv_cache_shape, dtype=self.dtype, device=device)], dim=dim)
                torch.cuda.empty_cache()
        if len(self.gpu_cache) < self.num_layers:
            tot_kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                self.num_gpu_blocks + num_blocks, self.block_size, self.num_heads, self.head_size)
            for i in range(0, self.num_layers - len(self.gpu_cache)):
                self.gpu_cache.append(
                    torch.empty(tot_kv_cache_shape,
                                dtype=self.dtype,
                                device=device))
        self.num_gpu_blocks += num_blocks

        free_gpu_memory = torch.cuda.mem_get_info()[0]
        print(f"current gpu memory after extend gpu cache = {free_gpu_memory / 1000000000.0} GB")

    def swap_in(self, src_to_dst: Dict[int, int], use_dest: Optional[bool] = False) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.dest_gpu_cache[i] if use_dest else self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int], use_dest: Optional[bool] = False) -> None:
        for i in range(self.num_layers):
            self.attn_backend.swap_blocks(self.dest_gpu_cache[i] if use_dest else self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: Dict[int, List[int]], use_dest: Optional[bool] = False) -> None:
        self.attn_backend.copy_blocks(self.dest_gpu_cache[i] if use_dest else self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = _get_dtype_size(dtype)

        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
