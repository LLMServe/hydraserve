# -*- coding: utf-8 -*-
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


class InternLM2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config)
        self.w2 = RowParallelLinear(intermediate_size,
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
        x, _ = self.w2(x)
        return x


class InternLM2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.wqkv = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.wo = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.wqkv(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.wo(attn_output)
        return output


class InternLMDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        self.attention = InternLM2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
        )
        self.feed_forward = InternLM2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.attention_norm = RMSNorm(config.hidden_size,
                                      eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            hidden_states = self.attention_norm(hidden_states)
        else:
            hidden_states, residual = self.attention_norm(
                hidden_states, residual)
        hidden_states = self.attention(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states, residual


class InternLM2Model(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_rank = int(os.getenv('PP_RANK', '0'))
        self.pp_size = int(os.getenv('PP_SIZE', '1'))
        if self.pp_rank == 0:
            if InternLM2ForCausalLM.duplicate_embed_tokens is not None:
                self.tok_embeddings = InternLM2ForCausalLM.duplicate_embed_tokens
            else:
                self.tok_embeddings = VocabParallelEmbedding(
                    config.vocab_size,
                    config.hidden_size,
                )
        if len(InternLM2ForCausalLM.duplicate_decoder_layer) > 0:
            layer_list = []
            for i in range(config.num_hidden_layers):
                if i in InternLM2ForCausalLM.duplicate_decoder_layer:
                    layer_list.append(InternLM2ForCausalLM.duplicate_decoder_layer[i])
                else:
                    layer_list.append(InternLMDecoderLayer(config, quant_config))
            self.layers = nn.ModuleList(layer_list)
        else:
            self.layers = nn.ModuleList([
                InternLMDecoderLayer(config, quant_config)
                for _ in range(config.num_hidden_layers)
            ])
        if self.pp_rank == self.pp_size - 1:
            if InternLM2ForCausalLM.duplicate_norm is not None:
                self.norm = InternLM2ForCausalLM.duplicate_norm
            else:
                self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            hidden_states = self.tok_embeddings(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
            )
        if self.pp_rank == self.pp_size - 1:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            # reserve residual to next stage
            hidden_states = hidden_states + residual
        return hidden_states


class InternLM2ForCausalLM(nn.Module):

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
    ) -> None:
        super().__init__()
        self.check_duplicate()
        self.config = config
        self.quant_config = quant_config
        self.model = InternLM2Model(config, quant_config)
        self.pp_rank = self.model.pp_rank
        self.pp_size = self.model.pp_size
        if self.pp_rank == self.pp_size - 1:
            if InternLM2ForCausalLM.duplicate_lm_head is not None:
                self.output = InternLM2ForCausalLM.duplicate_lm_head
            else:
                self.output = ParallelLMHead(config.vocab_size, config.hidden_size)
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

            hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, inputs_embeds=hidden_states)
        else:
            hidden_states = self.model(input_ids, positions, kv_caches,
                                    attn_metadata)

        if self.pp_rank < self.pp_size - 1:
            send_states(self.write_socket, hidden_states)

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        if self.pp_rank < self.pp_size - 1:
            return hidden_states
        logits = self.logits_processor(self.output.weight, hidden_states,
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
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
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
                if "wqkv" in name:
                    config = self.config
                    kv_groups = (config.num_attention_heads //
                                 config.num_key_value_heads)
                    head_dim = config.hidden_size // config.num_attention_heads
                    loaded_weight = loaded_weight.view(-1, 2 + kv_groups,
                                                       head_dim,
                                                       loaded_weight.shape[-1])
                    wq, wk, wv = torch.split(loaded_weight, [kv_groups, 1, 1],
                                             dim=1)
                    wq = wq.reshape(-1, wq.shape[-1])
                    wk = wk.reshape(-1, wk.shape[-1])
                    wv = wv.reshape(-1, wv.shape[-1])
                    weight_loader = param.weight_loader
                    weight_loader(param, wq, 'q')
                    weight_loader(param, wk, 'k')
                    weight_loader(param, wv, 'v')
                else:
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)

    def check_duplicate(self):
        if InternLM2ForCausalLM.prev_model is None:
            # it is the first time to load model, so store a ptr to current model
            InternLM2ForCausalLM.prev_model = self
        else:
            # we are loading dest model
            duplicate_state_dict_json = get_duplicate_state_dict()
            if len(duplicate_state_dict_json) > 0:
                prev_model = InternLM2ForCausalLM.prev_model
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
                        if 'output' in prev_name:
                            InternLM2ForCausalLM.duplicate_lm_head = prev_model.output
                        elif 'embed' in prev_name:
                            InternLM2ForCausalLM.duplicate_embed_tokens = prev_model.model.tok_embeddings
                        elif 'norm' in prev_name:
                            InternLM2ForCausalLM.duplicate_norm = prev_model.model.norm
                        else:
                            print(f"param name = {prev_name}")
                            raise ValueError(f"Error: unrecognized parameter.")
                    else:
                        # decoder layer
                        InternLM2ForCausalLM.duplicate_decoder_layer[dest_layer_num] = prev_model.model.layers[prev_layer_num]

    def set_socket(self, read_socket, write_socket):
        self.read_socket = read_socket
        self.write_socket = write_socket