"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Adapted from llama_eagle3.py
"""Inference-only LLaMA-EAGLE3-MoE model compatible with HuggingFace weights."""

import copy
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModel, LlamaConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.utils import add_prefix


class LlamaMoE(nn.Module):
    """
    Mixture of Experts layer for Eagle3.
    Uses FusedMoE for efficient expert parallel computation.
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        if config.model_type == "llama4_text":
            intermediate_size = config.intermediate_size_mlp
        else:
            intermediate_size = config.intermediate_size

        # Gate for routing tokens to experts
        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        # TopK selector for routing
        self.topk = TopK(
            top_k=self.top_k,
            renormalize=True,
        )

        # Fused MoE experts
        self.experts = FusedMoE(
            num_experts=self.num_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

        # Optional shared expert (like DeepSeek-V3)
        self.shared_expert = None
        if hasattr(config, "has_shared_expert") and config.has_shared_expert:
            from sglang.srt.models.llama import LlamaMLP

            self.shared_expert = LlamaMLP(
                config.hidden_size,
                intermediate_size,
                config.hidden_act,
                quant_config,
                prefix=add_prefix("shared_expert", prefix),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MoE layer.

        Args:
            hidden_states: Input tensor of shape [num_tokens, hidden_size]

        Returns:
            Output tensor of shape [num_tokens, hidden_size]
        """
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # Compute routing weights
        router_logits, _ = self.gate(hidden_states)

        # Get top-k experts
        topk_output = self.topk(hidden_states, router_logits)

        # Process through experts
        final_hidden_states = self.experts(hidden_states, topk_output)

        # Add shared expert output if present
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states.view(orig_shape)


class LlamaDecoderLayerMoE(nn.Module):
    """
    Decoder layer with MoE instead of standard MLP.
    Modified from LlamaDecoderLayer to support Eagle3 with MoE.
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # Import attention layer
        from sglang.srt.models.llama import LlamaAttention

        # Attention layer with modified qkv input size for Eagle3
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        # Override qkv_proj to accept 2*hidden_size input (concat of embeds and hidden_states)
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("self_attn.qkv_proj", prefix),
        )

        # MoE layer instead of standard MLP
        self.moe_layer = LlamaMoE(
            config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("moe_layer", prefix),
        )

        # Layer norms
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for decoder layer with MoE.

        Args:
            positions: Position indices
            embeds: Input embeddings
            hidden_states: Hidden states from previous layer
            forward_batch: Batch information for forward pass
            residual: Residual connection tensor

        Returns:
            Tuple of (hidden_states, residual)
        """
        # Save residual
        residual = hidden_states

        # Normalize inputs
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)

        # Concatenate embeds and hidden states for Eagle3
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Self attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Add residual
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MoE layer
        hidden_states = self.moe_layer(hidden_states)

        return hidden_states, residual


class LlamaModelMoE(nn.Module):
    """
    Eagle3 Model with MoE support.
    """

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.is_mrope_enabled = (
            hasattr(config, "rope_scaling")
            and config.rope_scaling is not None
            and "mrope_section" in config.rope_scaling
        )
        # fix rope_scaling for qwen2.5-vl
        if self.is_mrope_enabled:
            config.rope_scaling["rope_type"] = "default"

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        if hasattr(config, "target_hidden_size"):
            self.hidden_size_in = config.target_hidden_size
        else:
            self.hidden_size_in = config.hidden_size

        self.fc = torch.nn.Linear(
            self.hidden_size_in * 3,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        # Use MoE decoder layer
        self.midlayer = LlamaDecoderLayerMoE(config, 0, quant_config, prefix)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        # idle batch
        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            forward_batch,
            residual,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )

        # For draft decode, we capture the hidden state before norm
        return hidden_states_to_logits, [hidden_states_to_aux]


class LlamaForCausalLMEagle3MoE(LlamaForCausalLM):
    """
    Eagle3 model with Mixture of Experts support.

    This model extends the standard Eagle3 architecture with MoE layers,
    allowing for more efficient scaling and specialization.

    Config requirements:
        - num_local_experts: Number of expert networks
        - num_experts_per_tok: Number of experts to activate per token (top-k)
        - has_shared_expert (optional): Whether to include a shared expert
    """

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        if self.config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")

        # Validate MoE config
        if not hasattr(config, "num_local_experts"):
            raise ValueError(
                "num_local_experts must be specified in config for MoE model"
            )
        if not hasattr(config, "num_experts_per_tok"):
            raise ValueError(
                "num_experts_per_tok must be specified in config for MoE model"
            )

        # Check tensor parallel compatibility
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > config.num_local_experts:
            raise ValueError(
                f"Tensor parallel size {tp_size} is greater than "
                f"the number of experts {config.num_local_experts}."
            )

        self.model = LlamaModelMoE(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # Llama 3.2 1B Instruct set tie_word_embeddings to True
        # Llama 3.1 8B Instruct set tie_word_embeddings to False
        self.load_lm_head_from_target = False
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if config.draft_vocab_size is None:
                self.load_lm_head_from_target = True
                config.draft_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(
                config.draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        config_ = copy.deepcopy(config)
        config_.vocab_size = (
            config_.draft_vocab_size
        )  # draft logits processor has it's own vocab size
        self.logits_processor = LogitsProcessor(config_)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())

        # Define the parameter mapping for stacked parameters
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "d2t" in name:
                # d2t stores diffs between draft id and target id
                self.hot_token_id = loaded_weight + torch.arange(
                    loaded_weight.shape[0]
                )
                continue

            if "t2d" in name:
                continue

            # Handle router/gate weights for MoE
            if "router" in name:
                name = name.replace("router", "gate")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param_name = f"model.{name}" if name not in params_dict else name
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Handle regular parameters
                param_name = name if name in params_dict else f"model.{name}"
                if param_name in params_dict:
                    param = params_dict[param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)

    def get_hot_token_id(self):
        return self.hot_token_id


# Register with AutoModel to support loading via auto_map
AutoModel.register(LlamaConfig, LlamaForCausalLMEagle3MoE, exist_ok=True)

EntryClass = [LlamaForCausalLMEagle3MoE]

