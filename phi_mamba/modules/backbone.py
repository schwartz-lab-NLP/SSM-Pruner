from importlib import import_module

import torch.nn as nn

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from modules.modeling_llama import LlamaRotaryEmbedding, LlamaRMSNorm


class MixerModel(nn.Module):
    def __init__(
        self, input_size, config=None, device=None, dtype=None, mask_head_indexes=None, prune_dstates_ratio=None, final_layernorm_bias=True, **kwargs
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config
        n_layer = self.config.MixerModel.input.n_layer
        d_model = self.config.MixerModel.input.d_model
        self.mask_head_indexes = mask_head_indexes
        self.prune_dstates_ratio = prune_dstates_ratio
        self.embedding = nn.Embedding(input_size, d_model, **factory_kwargs)
        self.rotary_emb = None
        if hasattr(self.config.MixerModel.input,'rotary_emb') and self.config.MixerModel.input.rotary_emb:
            self.rotary_emb = LlamaRotaryEmbedding(config=self.config.MixerModel.input.rotary_emb_config, device=device)
        blocks = [
            self.config.__dict__[name]
            for name in self.config.__dict__.keys()
            if name.startswith("Block")
        ]
        self.layers = nn.ModuleList()
        for block_cfg in blocks:
            if hasattr(self.config, "sparse"):
                block_cfg.sparse = self.config.sparse
            n_layers = block_cfg.n_layers
            Block = import_module(block_cfg.BlockType).Block
            layers = nn.ModuleList(
                [
                    Block(
                        d_model=d_model,
                        config=block_cfg,
                        factory_kwargs=factory_kwargs,
                        layer_idx=i,
                        layer_mask_head_index=mask_head_indexes[i] if mask_head_indexes is not None else None,
                        prune_dstates_ratio=prune_dstates_ratio,
                        **kwargs,
                    ).to(device)
                    for i in range(len(self.layers), len(self.layers) + n_layers)
                ]
            )
            self.layers += layers
        assert len(self.layers) == n_layer

        # Initialize norm:
        norm_epsilon: float = 1e-5
        norm_cls = self.config.MixerModel.input.lm_head_prenorm
        if norm_cls == "layer":
            self.final_layernorm = nn.LayerNorm(d_model, eps=norm_epsilon, bias=final_layernorm_bias).to(device)
        elif norm_cls == "rms":
            self.final_layernorm = LlamaRMSNorm(d_model, eps=self.config.MixerModel.input.rms_norm_eps)
        else:
            raise Exception(f"Norm class {norm_cls} is not valid.")

        return

    @property
    def embed_tokens(self):
        return self.embedding

    def allocate_inference_cache(self, *args, **kwargs):
        return {
            i: layer.allocate_inference_cache(*args, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids,
        return_mixer_matrix=False,
        return_mamba_outputs=False,
        return_hidden_states=False,
        inference_params=None,
        position_ids=None,
        attention_mask=None,
        use_cache=False,
        mask_head_indexes=None,
        return_attn_presoftmax=False,
        **kwargs
    ):
        # compatibility wuth modeling phi
        return_hidden_states = return_hidden_states or kwargs.get("output_hidden_states", False)
        return_mixer_matrix = return_mixer_matrix or kwargs.get("output_attentions", False)
        return_mamba_outputs = return_mamba_outputs or kwargs.get("output_attention_results", False)

        # Start running the layers
        hidden_states = self.embedding(input_ids)

        if self.rotary_emb:
            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        # Initialize outputs
        outputs = {
            "last_hidden_state": None,
            "all_hidden_states": (hidden_states,) if return_hidden_states else (),
            "all_transfer_matrices": tuple(),
            "all_mamba_outputs": tuple(),
        }

        if attention_mask is not None:
            batch_size, seq_length = input_ids.shape[:2]
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                hidden_states,
                0 if inference_params is None else inference_params.seqlen_offset,
            )

        # Run the layers
        for ind, layer in enumerate(self.layers):
            if mask_head_indexes is not None:
                layer_mask_head_index = mask_head_indexes[ind]
            elif self.mask_head_indexes is not None:
                layer_mask_head_index = self.mask_head_indexes[ind]
            else:
                layer_mask_head_index = None
            layer_outputs = layer(
                hidden_states,
                return_mixer_matrix=return_mixer_matrix,
                return_mamba_outputs=return_mamba_outputs,
                inference_params=inference_params,
                position_ids=position_ids,
                attention_mask=attention_mask,
                mask_head_index=layer_mask_head_index,
                return_attn_presoftmax=return_attn_presoftmax,
                position_embeddings=position_embeddings,
            )
            # Record outputs
            hidden_states = layer_outputs["hidden_states"]
            if return_hidden_states:
                outputs["all_hidden_states"] += (hidden_states,)
            if return_mamba_outputs:
                outputs["all_mamba_outputs"] += (layer_outputs["mamba_hidden_states"],)
            if return_mixer_matrix:
                outputs["all_transfer_matrices"] += (layer_outputs["transfer_matrix"],)

        # Last layer, apply layer norm
        outputs["last_hidden_state"] = self.final_layernorm(hidden_states)
        return outputs


