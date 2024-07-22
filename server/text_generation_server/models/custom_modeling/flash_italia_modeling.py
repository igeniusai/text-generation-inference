import torch
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, List, Tuple

from text_generation_server.layers import (
    SpeculativeHead,
    get_linear,
)
from text_generation_server.layers.layernorm import (
    FastLayerNorm,
)
from text_generation_server.models.custom_modeling import flash_neox_modeling


class ItaliaConfig(flash_neox_modeling.GPTNeoXConfig):
    def __init__(
        self,
        hidden_act="gelu_new",  
        *args,
        **kwargs,
    ):
        super().__init__(
            hidden_act=hidden_act,
            *args,
            **kwargs,
        )




class ItaliaMLP(flash_neox_modeling.FlashMLP):
    def __init__(self, config, prefix, weights):
        super().__init__(config, prefix, weights)
        act = config.hidden_act
        self.act = ACT2FN[act]



class ItaliaLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()

        layer_norm_eps = config.layer_norm_eps

        prefix = f"gpt_neox.layers.{layer_id}"

        self.input_layernorm = FastLayerNorm.load(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=layer_norm_eps
        )
        self.attention = flash_neox_modeling.FlashNeoxAttention(
            config, prefix=f"{prefix}.attention", weights=weights
        )

        self.mlp = ItaliaMLP(config, prefix=f"{prefix}.mlp", weights=weights)
        self.process_group = weights.process_group

    def forward(
        self,
        hidden_states,
        residual,
        cos,
        sin,
        cu_seqlen_prefill,
        kv_cache,
        block_tables,
        slots,
        input_lengths,
        max_s,
    ):
        ln1_hidden_states, _ = self.input_layernorm(hidden_states)

        attn_output = self.attention(
            ln1_hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )

        mlp_output = self.mlp(self.input_layernorm(hidden_states)[0])
        hidden_states = mlp_output + attn_output + hidden_states

        if self.process_group.size() > 1:
            torch.distributed.all_reduce(hidden_states, group=self.process_group)

        return hidden_states, None

    
flash_neox_modeling.FlashNeoXLayer = ItaliaLayer

class FlashItaliaForCausalLM(flash_neox_modeling.FlashGPTNeoXForCausalLM):
    pass
