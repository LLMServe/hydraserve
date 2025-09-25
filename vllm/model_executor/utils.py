"""Utils for model executor."""
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import time
import ctypes
import json
import re
import os

from vllm.utils import socket_send, socket_recv

def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(
            weight, key), (f"Overwriting existing tensor attribute: {key}")
        setattr(weight, key, value)

# Send hidden states from previous pipeline parallel stage
def send_states(sock, hidden_states: torch.Tensor):
    torch.cuda.synchronize()

    stime = time.time()

    # write intermediate results to latter stage
    hidden_states_cpu = hidden_states.to("cpu")
    num_bytes = hidden_states_cpu.element_size() * hidden_states_cpu.nelement()
    byte_string = ctypes.string_at(hidden_states_cpu.data_ptr(), num_bytes)

    socket_send(sock, byte_string)

    print(f"send_states: send {len(byte_string)} bytes, time cost = {(time.time() - stime) * 1000.0} ms")

# Recv hidden states from previous pipeline parallel stage
def recv_states(sock, dtype, device, shape) -> torch.Tensor:
    stime = time.time()

    output = socket_recv(sock)

    etime = time.time()
    print(f"recv_states: wait for read hidden_states time cost = {(etime - stime) * 1000.0} ms")
    stime = etime

    hidden_states = torch.frombuffer(output, dtype=dtype)
    hidden_states = hidden_states.view(shape).to(device)

    etime = time.time()
    print(f"recv_states: read {len(output)} bytes, time cost = {(etime - stime) * 1000.0} ms")

    return hidden_states

# Get duplicate state dict if we are initializing normal model and dest model will be loaded later
def get_duplicate_state_dict() -> Dict[str, str]:
    if os.getenv('DEST_PP_RANK', '-1') == '-1':
        return {}
    print("Checking duplicate state dict...")
    dest_model_path = os.getenv('DEST_MODEL_PATH')
    duplicate_state_dict_path = dest_model_path.replace('dest_model', 'duplicate_state_dict').replace('.safetensors', '.json')
    with open(duplicate_state_dict_path, "r") as f:
        duplicate_state_dict_json = json.load(f)
    return duplicate_state_dict_json

# Extract dest layer id from (prev_name, after_name) pair in duplicate_state_dict
def extract_dest_layer(prev_name: str, after_name: str) -> Tuple[int, int]:
    pattern = re.compile(r'\d+')
    substrs_prev = pattern.findall(prev_name)
    substrs_after = pattern.findall(after_name)
    if len(substrs_prev) == 0 or len(substrs_after) == 0:
        return -1, -1
    return int(substrs_prev[0]), int(substrs_after[0])