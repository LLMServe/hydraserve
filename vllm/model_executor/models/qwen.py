# coding=utf-8
# Adapted from
# https://huggingface.co/Qwen/Qwen-7B/blob/main/modeling_qwen.py
# Copyright (c) Alibaba Cloud.
# LICENSE: https://huggingface.co/Qwen/Qwen-7B/blob/main/LICENSE
"""Inference-only QWen model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple
import time
import os

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import send_states, recv_states, get_duplicate_state_dict, extract_dest_layer
from vllm.sequence import SamplerOutput


class QWenMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.c_proj = RowParallelLinear(intermediate_size,
                                        hidden_size,
                                        bias=False,
                                        quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class QWenAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size(
        )
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = (self.total_num_heads //
                          tensor_model_parallel_world_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.c_attn = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            bias=True,
            quant_config=quant_config,
        )
        self.c_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )
        self.scaling = self.head_dim**-0.5

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.c_proj(attn_output)
        return output


class QWenBlock(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        self.attn = QWenAttention(config.hidden_size,
                                  config.num_attention_heads,
                                  config.max_position_embeddings,
                                  rope_theta=rope_theta,
                                  rope_scaling=rope_scaling,
                                  quant_config=quant_config)

        self.ln_2 = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = QWenMLP(config.hidden_size,
                           config.intermediate_size // 2,
                           quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.ln_1(hidden_states)
        else:
            hidden_states, residual = self.ln_1(hidden_states, residual)
        hidden_states = self.attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.ln_2(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class QWenModel(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pp_rank = int(os.getenv('PP_RANK', '0'))
        self.pp_size = int(os.getenv('PP_SIZE', '1'))

        if self.pp_rank == 0:
            if QWenLMHeadModel.duplicate_embed_tokens is not None:
                self.wte = QWenLMHeadModel.duplicate_embed_tokens
            else:
                self.wte = VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                )
        if len(QWenLMHeadModel.duplicate_decoder_layer) > 0:
            layer_list = []
            for i in range(config.num_hidden_layers):
                if i in QWenLMHeadModel.duplicate_decoder_layer:
                    layer_list.append(QWenLMHeadModel.duplicate_decoder_layer[i])
                else:
                    layer_list.append(QWenBlock(config, quant_config))
            self.h = nn.ModuleList(layer_list)
        else:
            self.h = nn.ModuleList([
                QWenBlock(config, quant_config)
                for _ in range(config.num_hidden_layers)
            ])
        if self.pp_rank == self.pp_size - 1:
            if QWenLMHeadModel.duplicate_norm is not None:
                self.ln_f = QWenLMHeadModel.duplicate_norm
            else:
                self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.wte(input_ids)
        residual = None
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        if self.pp_rank == self.pp_size - 1:
            hidden_states, _ = self.ln_f(hidden_states, residual)
        else:
            # reserve residual to next stage
            hidden_states = hidden_states + residual
        return hidden_states


class QWenLMHeadModel(nn.Module):

    # For duplicate state dict
    prev_model = None
    duplicate_embed_tokens = None
    duplicate_norm = None
    duplicate_lm_head = None
    duplicate_decoder_layer = {}

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.check_duplicate()
        self.config = config
        self.quant_config = quant_config
        self.transformer = QWenModel(config, quant_config)

        self.pp_rank = self.transformer.pp_rank
        self.pp_size = self.transformer.pp_size
        if self.pp_rank == self.pp_size - 1:
            if QWenLMHeadModel.duplicate_lm_head is not None:
                self.lm_head = QWenLMHeadModel.duplicate_lm_head
            else:
                self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
            self.logits_processor = LogitsProcessor(config.vocab_size)
            self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.pp_rank > 0:
            stime = time.time()

            device = input_ids.device
            shape = [input_ids.shape[0], self.config.hidden_size]
            hidden_states = recv_states(self.read_socket, self.config.torch_dtype, device, shape)

            hidden_states = self.transformer(input_ids, positions, kv_caches,
                                    attn_metadata, inputs_embeds=hidden_states)
        else:
            hidden_states = self.transformer(input_ids, positions, kv_caches,
                                    attn_metadata)

        if self.pp_rank < self.pp_size - 1:
            send_states(self.write_socket, hidden_states)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        if self.pp_rank < self.pp_size - 1:
            return hidden_states
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        if self.pp_rank < self.pp_size - 1:
            return None
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w2", 0),
            ("gate_up_proj", "w1", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    def check_duplicate(self):
        if QWenLMHeadModel.prev_model is None:
            # it is the first time to load model, so store a ptr to current model
            QWenLMHeadModel.prev_model = self
        else:
            # we are loading dest model
            duplicate_state_dict_json = get_duplicate_state_dict()
            if len(duplicate_state_dict_json) > 0:
                prev_model = QWenLMHeadModel.prev_model
                for prev_name, after_name in duplicate_state_dict_json.items():
                    if "rotary_emb.inv_freq" in prev_name:
                        continue
                    prev_layer_num, dest_layer_num = extract_dest_layer(prev_name, after_name)
                    if prev_layer_num == -1:
                        # not decoder layer
                        # may be embed, lm_head, final_norm, etc.
                        if prev_name != after_name:
                            print(f"prev = {prev_name}, after = {after_name}")
                            raise ValueError("Cannot extract param mapping.")
                        if 'lm_head' in prev_name:
                            QWenLMHeadModel.duplicate_lm_head = prev_model.lm_head
                        elif 'wte' in prev_name:
                            QWenLMHeadModel.duplicate_embed_tokens = prev_model.transformer.wte
                        elif 'ln_f' in prev_name:
                            QWenLMHeadModel.duplicate_norm = prev_model.transformer.ln_f
                        else:
                            print(f"param name = {prev_name}")
                            raise ValueError(f"Error: unrecognized parameter.")
                    else:
                        # decoder layer
                        QWenLMHeadModel.duplicate_decoder_layer[dest_layer_num] = prev_model.transformer.h[prev_layer_num]

    def set_socket(self, read_socket, write_socket):
        self.read_socket = read_socket
        self.write_socket = write_socket