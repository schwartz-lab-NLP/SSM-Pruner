# Copyright 2023 Microsoft and the HuggingFace Inc. team. All rights reserved.
# transformers.models.phi.modeling_phi
from cgitb import reset
from importlib import import_module

import torch.nn as nn
from torch import Tensor
from transformers.models.llama.configuration_llama import LlamaConfig

from modules.modeling_llama import LlamaMLP, LlamaRMSNorm


class Block(nn.Module):
    def __init__(self, d_model, config, factory_kwargs, layer_idx, layer_mask_head_index=None, prune_dstates_ratio=None, **kwargs):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.layer_idx = layer_idx

        # Mixer
        MixerClass = import_module(config.CoreType).Mixer
        self.mixer = MixerClass(
            d_model=self.d_model,
            layer_idx=layer_idx,
            layer_mask_head_index=layer_mask_head_index,
            prune_dstates_ratio=prune_dstates_ratio,
            hf_config=config.core_input,
            **kwargs,
            **config.core_input,
            **factory_kwargs,
        )

        # MLP + LayerNorm + Dropout
        mlp_intermediate_size = config.mlp.intermediate_size if hasattr(config, "mlp") else self.d_model * 4
        self.mlp = LlamaMLP(
            LlamaConfig(
                hidden_size=self.d_model,
                intermediate_size=mlp_intermediate_size,
                hidden_act="silu",
            )
        )
        self.input_layernorm = LlamaRMSNorm(self.d_model, eps=config.block_input.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(self.d_model, eps=config.block_input.rms_norm_eps)


    def forward(
        self,
        hidden_states: Tensor,
        inference_params=None,
        run_mlp_component=True,
        return_mixer_matrix=False,
        return_mamba_outputs=False,
        position_ids=None,
        attention_mask=None,
        use_cache=False,
        layer_mask_head_index = None,
        return_attn_presoftmax=False,
        position_embeddings=None,
        **kwargs,
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        outputs = {}

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Apply Mixer
        mamba_outputs = self.mixer(
            hidden_states,
            return_mixer_matrix=return_mixer_matrix,
            inference_params=inference_params,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            layer_mask_head_index=layer_mask_head_index,
            return_attn_presoftmax=return_attn_presoftmax,
            position_embeddings=position_embeddings,
        )
        mamba_outputs["hidden_states"] = mamba_outputs["hidden_states"].to(
            residual.dtype
        )

        if not run_mlp_component:
            return mamba_outputs

        # store outputs
        if return_mamba_outputs:
            outputs["mamba_hidden_states"] = mamba_outputs["hidden_states"]
        if return_mixer_matrix:
            outputs["transfer_matrix"] = mamba_outputs["transfer_matrix"]

        hidden_states = residual + mamba_outputs["hidden_states"]
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs["hidden_states"] = hidden_states


        return outputs

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        if getattr(self.mixer, "allocate_inference_cache", None) is None:
            return
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )


    def prune_by_mask_head_index(self):
        self.mixer.prune_by_mask_head_index()


