from PIL.ImageOps import expand
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from torch import Tensor
from transformers.activations import ACT2FN

from MambaInLlama.mamba2.hybrid_mamba_config import MambaConfig
from MambaInLlama.mamba2.hybrid_mamba_layer import Mamba2

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, config: MambaConfig,
                 gate_proj_in_features=None,
                 gate_proj_out_features=None,
                 gate_proj_bias=False,
                 up_proj_in_features=None,
                 up_proj_out_features=None,
                 up_proj_bias=False,
                 down_proj_in_features=None,
                 down_proj_out_features=None,
                 down_proj_bias=False
                 ):
        super().__init__()
        self.config = config
        self.hidden_size = config.d_model
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size if gate_proj_in_features is None else gate_proj_in_features,
                                   self.intermediate_size if gate_proj_out_features is None else gate_proj_out_features,
                                   bias=gate_proj_bias)
        self.up_proj = nn.Linear(self.hidden_size if up_proj_in_features is None else up_proj_in_features,
                                 self.intermediate_size if up_proj_out_features is None else up_proj_out_features,
                                 bias=up_proj_bias)
        self.down_proj = nn.Linear(self.intermediate_size if down_proj_in_features is None else down_proj_in_features,
                                   self.hidden_size if down_proj_out_features is None else down_proj_out_features,
                                   bias=down_proj_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def update_config(self, config: MambaConfig, layer_idx):
        layer_idx = int(layer_idx)
        config.gate_proj_in_features_dict[layer_idx] = self.gate_proj.in_features
        config.gate_proj_out_features_dict[layer_idx] = self.gate_proj.out_features
        config.gate_proj_bias_dict[layer_idx] = self.gate_proj.bias is not None
        config.up_proj_in_features_dict[layer_idx] = self.up_proj.in_features
        config.up_proj_out_features_dict[layer_idx] = self.up_proj.out_features
        config.up_proj_bias_dict[layer_idx] = self.up_proj.bias is not None
        config.down_proj_in_features_dict[layer_idx] = self.down_proj.in_features
        config.down_proj_out_features_dict[layer_idx] = self.down_proj.out_features
        config.down_proj_bias_dict[layer_idx] = self.down_proj.bias is not None




class MambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MambaConfig,
        layer_idx: int,
        device=None,
        dtype=None,
        residual_in_fp32=True,
    ):
        super(MambaDecoderLayer, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        layer_idx = int(layer_idx)
        self.layer_idx = layer_idx
        if str(layer_idx) in config.d_xb_dict:
            layer_idx = str(layer_idx)
        if  layer_idx in config.d_xb_dict:
            self.mamba = Mamba2(
                d_model=config.d_model, d_xb=config.d_xb_dict[layer_idx], d_inner=config.d_inner_dict[layer_idx],
                d_state=config.d_state_dict[layer_idx], d_ssm=config.d_ssm_dict[layer_idx], nheads=config.nheads_dict[layer_idx],
                ngroups=config.ngroups_dict[layer_idx], headdim=config.headdim_dict[layer_idx], in_proj_out_features=config.in_proj_out_features_dict[layer_idx],
                out_proj_in_features=config.out_proj_in_features_dict[layer_idx], conv1d_dim=config.conv_dim_dict[layer_idx],
                in_proj_bias=config.in_proj_bias_dict[layer_idx], out_proj_bias=config.out_proj_bias_dict[layer_idx],
                layer_idx=layer_idx, expand=config.ssm_cfg['expand'], D_shape=config.D_dict[layer_idx],**factory_kwargs
            )
        else:
            self.mamba = Mamba2(
                d_model=config.d_model, d_xb=config.d_xb, d_inner=config.d_inner, layer_idx=layer_idx, **config.ssm_cfg, **factory_kwargs
            )
        if int(layer_idx) in config.gate_proj_in_features_dict:
            layer_idx = int(layer_idx)
        elif str(layer_idx) in config.gate_proj_in_features_dict:
            layer_idx = str(layer_idx)

        if layer_idx in config.gate_proj_in_features_dict:
            self.mlp = MLP(config=config, gate_proj_in_features=config.gate_proj_in_features_dict[layer_idx],
                           gate_proj_out_features=config.gate_proj_out_features_dict[layer_idx],
                           up_proj_in_features=config.up_proj_in_features_dict[layer_idx],
                           up_proj_out_features=config.up_proj_out_features_dict[layer_idx],
                           down_proj_in_features=config.down_proj_in_features_dict[layer_idx],
                           down_proj_out_features=config.down_proj_out_features_dict[layer_idx])
        else:
            self.mlp = MLP(config=config)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.residual_in_fp32 = True
        
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, hidden_states: Tensor, *args, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        # hidden_states = self.mamba(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # so here is just to be compatible with Transformer
        if kwargs is None:
            return (hidden_states, None, None)
        else:
            past_key_value = kwargs.get("past_key_value", None)
            if past_key_value is not None:
                dummy_keys = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                dummy_values = torch.ones(
                    1, 1, hidden_states.size(1), 1, device=hidden_states.device, dtype=hidden_states.dtype
                )
                # Update kv cache with dummy values
                past_key_value.update(dummy_keys, dummy_values, self.layer_idx)
            return (hidden_states, None, past_key_value)

    @property
    def mixer(self):
        return self.mamba

    def update_config(self, config: MambaConfig):
        self.mamba.update_config(config)
        self.mlp.update_config(config, self.layer_idx)