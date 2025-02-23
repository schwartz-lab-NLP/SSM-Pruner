# Copyright (c) 2023, Albert Gu, Tri Dao.

import math
from functools import partial
import json
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from collections import namedtuple

import torch
import torch.nn as nn

from original_mamba.mamba_ssm.models.config_mamba import MambaConfig
from original_mamba.mamba_ssm.modules.mamba_simple import Mamba
from original_mamba.mamba_ssm.modules.mamba2 import Mamba2
from original_mamba.mamba_ssm.modules.mha import MHA
from original_mamba.mamba_ssm.modules.mlp import GatedMLP
from original_mamba.mamba_ssm.modules.block import Block
from original_mamba.mamba_ssm.utils.generation import GenerationMixin
from original_mamba.mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    **mixer_kwargs
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        **mixer_kwargs
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
        config=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.device = device
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        blocks = []
        for i in range(n_layer):
            mixer_kwargs = {}
            if config is not None:
                if len(config.d_inner_list) > 0:
                    mixer_kwargs["d_inner"] = config.d_inner_list[i]
                if len(config.headdim_list) > 0:
                    mixer_kwargs["headdim"] = config.headdim_list[i]
                if len(config.nheads_list) > 0:
                    mixer_kwargs["nheads"] = config.nheads_list[i]
                if len(config.ngroups_list) > 0:
                    mixer_kwargs["ngroups"] = config.ngroups_list[i]
                if len(config.d_state_list) > 0:
                    mixer_kwargs["d_state"] = config.d_state_list[i]
                if len(config.d_ssm_list) > 0:
                    mixer_kwargs["d_ssm"] = config.d_ssm_list[i]
                if len(config.in_proj_bias_list) > 0:
                    mixer_kwargs["in_proj_bias"] = config.in_proj_bias_list[i]
                if len(config.in_proj_out_features_list) > 0:
                    mixer_kwargs["in_proj_out_features"] = config.in_proj_out_features_list[i]
                if len(config.out_proj_bias_list) > 0:
                    mixer_kwargs["out_proj_bias"] = config.out_proj_bias_list[i]
                if len(config.out_proj_in_features_list) > 0:
                    mixer_kwargs["out_proj_in_features"] = config.out_proj_in_features_list[i]
                if len(config.conv_dim_list) > 0:
                    mixer_kwargs["conv1d_dim"] = config.conv_dim_list[i]

            blocks.append(
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                    **mixer_kwargs
                )
            )
        self.layers = nn.ModuleList(blocks)
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class MambaLMHeadModel(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.device = device
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        d_intermediate = config.d_intermediate
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            config=config,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)
        shift_labels = input_ids[:, 1:].contiguous()
        shift_logits = lm_logits[:, :-1].contiguous()
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "loss"])
        return CausalLMOutput(logits=lm_logits, loss=loss)


    @classmethod
    def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
        config_data = load_config_hf(pretrained_model_name)
        config = MambaConfig(**config_data)
        model = cls(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
        return model

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        os.makedirs(save_directory, exist_ok=True)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        print(self.config)
        # update config
        for layer in self.backbone.layers:
            layer.update_config(self.config)
        print(self.config)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=4)

    @property
    def model(self):
        return self.backbone

    def prune_dstates(self, ratio, method, exclude_layers=None):
        layers = self.backbone.layers if exclude_layers is None else [layer for i, layer in
                                                                      enumerate(self.backbone.layers) if
                                                                      i not in exclude_layers]
        for layer in layers:
            layer.mixer.prune_dstates(ratio, method=method)

    def prune_headdim(self, ratio, method, exclude_layers=None, exclude_out_proj=True, only_nullify=False):
        layers = self.backbone.layers if exclude_layers is None else [layer for i, layer in
                                                                      enumerate(self.backbone.layers) if
                                                                      i not in exclude_layers]
        for layer in layers:
            layer.mixer.prune_headdim(ratio, method=method, exclude_out_proj=exclude_out_proj, only_nullify=only_nullify)

    def prune_kq_heads(self, new_qk_heads: int, clustering_method='mean_pooling', exclude_layers=None):
        layers = self.backbone.layers if exclude_layers is None else [layer for i, layer in enumerate(self.backbone.layers) if i not in exclude_layers]
        for layer in layers:
            layer.mixer.prune_kq_heads(new_qk_heads, clustering_method)

    def split_in_proj(self):
        for layer in self.backbone.layers:
            layer.mixer.split_in_proj()

    import numpy as np
    import matplotlib.pyplot as plt
    import math

    def visualize_inproj(
            self,
            layer_idx: int = 0,
            *,
            # A list of integers defining the row sizes for each sub-matrix in the in_proj weight.
            split_sizes: list[int] = None,
            # Optional names for each sub-matrix (e.g. ["Q", "K", "V"]).
            split_names: list[str] = None,
            # A dictionary mapping sub-matrix index (starting at 0) to number of heads
            # for further splitting that sub-matrix.
            multi_head_splits: dict[int, int] = None,
            cmap: str = "viridis",
            log_scale: bool = False,
            # Optionally you can still pass vmin/vmax if you want to override auto-computation.
            vmin: float = None,
            vmax: float = None,
            show: bool = True,
            save_path: str = None,
            show_color_bar=False
    ):
        """
        Visualize the consolidated in-projection weight matrix from a given layer as a
        colored heatmap (or set of heatmaps) that expose its constituent sub-matrices.

        The in-projection weight is assumed to be stored in
            self.backbone.layers[layer_idx].mixer.in_proj.weight
        and is a consolidation (vertical concatenation) of several sub-matrices.

        Args:
            layer_idx (int): Index of the layer to visualize.
            split_sizes (list[int], optional): List of integers summing to the total number
                of rows in the in-projection weight. Each integer gives the number of rows for one
                sub-matrix. If not provided, the entire weight is visualized as one block.
            split_names (list[str], optional): List of names (labels) for each sub-matrix.
            multi_head_splits (dict[int, int], optional): Dictionary where keys are indices
                referring to a sub-matrix (from split_sizes) and values are the number
                of heads for that sub-matrix. The method will further split those sub-matrices evenly.
            cmap (str): Colormap to use (default "viridis").
            log_scale (bool): If True, use a logarithmic (symmetric) scaling to enhance differences.
            vmin (float, optional): Minimum value for colormap scaling (if None, computed from data).
            vmax (float, optional): Maximum value for colormap scaling (if None, computed from data).
            show (bool): Whether to display the plot.
            save_path (str, optional): If provided, save the figure to this path.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Get the desired layer and its in_proj weight.
        try:
            layer = self.backbone.layers[layer_idx]
        except IndexError:
            raise ValueError(f"Layer index {layer_idx} is out of range.")
        if not hasattr(layer.mixer, "in_proj"):
            raise ValueError(f"Layer {layer_idx}'s mixer does not have an attribute 'in_proj'.")
        weight = layer.mixer.in_proj.weight.detach().cpu().numpy()  # shape: [total_rows, input_dim]
        total_rows, input_dim = weight.shape

        # If split_sizes is provided, split the matrix accordingly.
        submatrices = []
        labels = []
        if split_sizes is not None:
            if sum(split_sizes) != total_rows:
                raise ValueError(
                    f"Sum of split_sizes {sum(split_sizes)} does not equal total rows {total_rows}."
                )
            indices = np.cumsum(split_sizes)[:-1]
            submatrices = np.split(weight, indices, axis=0)
            if split_names is not None:
                if len(split_names) != len(submatrices):
                    raise ValueError("Length of split_names must equal number of splits.")
                labels = split_names
            else:
                labels = [f"Split {i}" for i in range(len(submatrices))]
        else:
            submatrices = [weight]
            labels = ["InProj"]

        # Prepare list of (matrix, title) items to plot.
        matrices_to_plot = []
        titles = []
        multi_head_splits = multi_head_splits or {}

        for i, submat in enumerate(submatrices):
            if i in multi_head_splits:
                n_heads, head_to_plot_inds = multi_head_splits[i]
                head_to_plot_inds = head_to_plot_inds if isinstance(head_to_plot_inds, list) else [head_to_plot_inds]
                rows = submat.shape[0]
                if rows % n_heads != 0:
                    raise ValueError(
                        f"Submatrix {i} has {rows} rows which is not divisible by {n_heads} heads."
                    )
                head_size = rows // n_heads
                head_matrices = np.split(submat, n_heads, axis=0)
                for j, head_mat in enumerate(head_matrices):
                    if j in head_to_plot_inds:
                        matrices_to_plot.append(head_mat)
                        titles.append(f"{labels[i]} - Head {j}")

            matrices_to_plot.append(submat)
            titles.append(labels[i])

        # Compute a global min/max from all submatrices if not provided.
        all_vals = np.concatenate([mat.ravel() for mat in matrices_to_plot])
        if vmin is None:
            vmin = float(np.min(all_vals))
        if vmax is None:
            vmax = float(np.max(all_vals))

        # Optionally set up a normalization object.
        norm = None
        if log_scale:
            # Use SymLogNorm to handle negative and positive values.
            # linthresh is a small value that defines the range around zero that remains linear.
            norm = colors.SymLogNorm(linthresh=1e-3, vmin=vmin, vmax=vmax)
            # norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        # Otherwise, you could do per-matrix min-max scaling if desired,
        # but here we use the global vmin/vmax for consistency across subplots.

        # Determine grid size for subplots.
        n_plots = len(matrices_to_plot)
        n_cols = math.ceil(math.sqrt(n_plots))
        n_rows = math.ceil(n_plots / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_plots == 1:
            axes = np.array([[axes]])
        else:
            axes = np.array(axes).reshape(n_rows, n_cols)

        plot_idx = 0
        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes[r, c]
                if plot_idx < n_plots:
                    # Pass the normalization if one is set.
                    im = ax.imshow(
                        matrices_to_plot[plot_idx],
                        cmap=cmap,
                        norm=norm,
                        vmin=None if norm else vmin,
                        vmax=None if norm else vmax,
                    )
                    ax.set_title(titles[plot_idx])
                    if show_color_bar:
                        fig.colorbar(im, ax=ax)
                else:
                    ax.axis("off")
                plot_idx += 1

        fig.suptitle(f"InProj Weight Visualization (Layer {layer_idx})", fontsize=16)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_outproj(
            self,
            layer_idx: int = 0,
            cmap: str = "viridis",
            log_scale: bool = False,
            vmin: float = None,
            vmax: float = None,
            linthresh: float = 1e-3,
            show: bool = True,
            save_path: str = None,
            show_color_bar=False
    ):
        """
        Visualize the output projection (out_proj) weight matrix from a given layer.

        The method assumes that in the desired layer (accessed via
        self.backbone.layers[layer_idx].mixer), there is an attribute 'out_proj'
        (typically an nn.Linear) whose weight matrix is of shape [output_features, input_features].

        Args:
            layer_idx (int): Index of the layer whose out_proj weight is to be visualized.
            cmap (str): Colormap to use.
            log_scale (bool): If True, apply symmetric log scaling (useful when weight differences are small).
            vmin (float, optional): Global minimum value for color scaling. If None, computed from the data.
            vmax (float, optional): Global maximum value for color scaling. If None, computed from the data.
            linthresh (float): The threshold around zero for symmetric log scaling. A default of 1e-3 is a good starting point.
            show (bool): Whether to display the plot.
            save_path (str, optional): If provided, the figure is saved to this file.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Access the specified layer.
        try:
            layer = self.backbone.layers[layer_idx]
        except IndexError:
            raise ValueError(f"Layer index {layer_idx} is out of range.")
        if not hasattr(layer.mixer, "out_proj"):
            raise ValueError(f"Layer {layer_idx}'s mixer does not have an attribute 'out_proj'.")

        # Extract the weight matrix.
        weight = layer.mixer.out_proj.weight.detach().cpu().numpy()  # shape: [output_features, input_features]

        # Compute global min/max if not provided.
        if vmin is None:
            vmin = float(np.min(weight))
        if vmax is None:
            vmax = float(np.max(weight))

        # Set up normalization.
        norm = None
        if log_scale:
            norm = colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
            # norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

        # Create the figure and single axis.
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(weight, cmap=cmap, norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
        ax.set_title(f"OutProj Weight Visualization (Layer {layer_idx})")

        # Add one common colorbar.
        if show_color_bar:
            fig.colorbar(im, ax=ax, orientation="vertical")
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved out_proj visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    def visualize_conv1d_3d_single(
            self,
            layer_idx: int = 0,
            cmap: str = "viridis",
            log_scale: bool = False,
            vmin: float = None,
            vmax: float = None,
            linthresh: float = 1e-3,
            max_filters: int = None,
            highlight_filters: list[int] = None,
            base_color: str = "gray",
            highlight_color: str = "red",
            show: bool = True,
            save_path: str = None,
    ):
        """
        Visualize all conv1d filters from a given layer in a single 3D plot.

        For a depthwise conv1d layer (e.g. with parameters like
            Conv1d(6144, 6144, kernel_size=(4,), stride=(1,), padding=(3,), groups=6144)
        ) the weight tensor has shape (out_channels, 1, kernel_size). In that case, each filter
        is a 1×4 vector. This method plots each filter as a line in 3D where:

            - X-axis: Kernel positions (0 to kernel_size-1, displayed as integers).
            - Y-axis: Filter index (offset so filters do not overlap).
            - Z-axis: Weight values.

        By default, if `highlight_filters` is not provided (or empty), the method uses a colormap
        to determine the color of each filter (based on its average weight). However, if you pass
        a list of filter indices in `highlight_filters`, then all filters are drawn in a uniform color
        (specified by `base_color`), except for those indices which are drawn in `highlight_color`.

        Args:
            layer_idx (int): Index of the layer containing the conv1d layer.
            cmap (str): Colormap to use if not using uniform colors (default "viridis").
            log_scale (bool): If True, use symmetric log normalization for the dynamic coloring.
            vmin (float, optional): Global minimum for normalization. If None, computed from data.
            vmax (float, optional): Global maximum for normalization. If None, computed from data.
            linthresh (float): Threshold around zero for symmetric log scaling (default 1e-3).
            max_filters (int, optional): If provided, only visualize up to this many filters.
            highlight_filters (list[int], optional): List of filter indices to highlight.
            base_color (str): Color for non-highlighted filters (when using uniform coloring).
            highlight_color (str): Color for highlighted filters.
            show (bool): Whether to display the plot.
            save_path (str, optional): If provided, the figure is saved to this file.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Access the specified layer.
        try:
            layer = self.backbone.layers[layer_idx]
        except IndexError:
            raise ValueError(f"Layer index {layer_idx} is out of range.")
        if not hasattr(layer.mixer, "conv1d"):
            raise ValueError(f"Layer {layer_idx}'s mixer does not have an attribute 'conv1d'.")

        # Extract the conv1d weight: expected shape (out_channels, in_channels, kernel_size)
        conv_layer = layer.mixer.conv1d
        weight = conv_layer.weight.detach().cpu().numpy()
        out_channels, in_channels, kernel_size = weight.shape

        # Optionally restrict the number of filters visualized.
        if max_filters is not None and max_filters < out_channels:
            weight = weight[:max_filters]
            out_channels = max_filters

        # Decide on coloring mode:
        use_uniform = bool(highlight_filters and len(highlight_filters) > 0)

        if not use_uniform:
            # Compute global min and max over all weights if not provided.
            if vmin is None:
                vmin = float(np.min(weight))
            if vmax is None:
                vmax = float(np.max(weight))
            # Set up normalization and get colormap instance.
            if log_scale:
                norm = colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
                # norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
            else:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
            cmap_instance = plt.get_cmap(cmap)

        # Create a single 3D axis.
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Plot each filter as a 3D line.
        for i in range(out_channels):
            # For depthwise conv1d, each filter is of shape (1, kernel_size).
            x = np.arange(kernel_size)  # Kernel positions (integers)
            y = np.full_like(x, i)  # Use filter index as constant Y offset.
            z = weight[i, 0, :]  # Weight values for this filter.

            if use_uniform:
                color = highlight_color if i in highlight_filters else base_color
            else:
                avg_val = np.mean(z)
                color = cmap_instance(norm(avg_val))

            ax.plot(x, y, z, marker="o", color=color, lw=2)

        # Set x-axis ticks to integer kernel positions.
        ax.set_xticks(np.arange(kernel_size))
        ax.set_xlabel("Kernel Position")
        ax.set_ylabel("Filter Index")
        ax.set_zlabel("Weight Value")
        ax.set_title(f"3D Visualization of Conv1d Filters (Layer {layer_idx})")

        # Add a colorbar only if using dynamic (colormap) coloring.
        if not use_uniform:
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_instance)
            mappable.set_array(weight.ravel())
            fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=20)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved 3D conv1d visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


    def visualize_conv1d(
            self,
            layer_idx: int = 0,
            cmap: str = "viridis",
            log_scale: bool = False,
            vmin: float = None,
            vmax: float = None,
            linthresh: float = 1e-3,
            max_filters: int = 10,  # New parameter: maximum number of filters to visualize.
            show: bool = True,
            save_path: str = None,
    ):
        """
        Visualize the 1D convolution (conv1d) filters from a given layer.

        The method assumes that in the desired layer (accessed via
        self.backbone.layers[layer_idx].mixer), there is an attribute 'conv1d'
        (typically an nn.Conv1d) whose weight tensor is of shape
        [out_channels, in_channels, kernel_size]. Each filter (i.e. for each out_channel)
        is visualized as a 2D heatmap of shape [in_channels, kernel_size].

        Args:
            layer_idx (int): Index of the layer whose conv1d filters are to be visualized.
            cmap (str): Colormap to use.
            log_scale (bool): If True, apply symmetric log scaling.
            vmin (float, optional): Global minimum value for color scaling. If None, computed from the data.
            vmax (float, optional): Global maximum value for color scaling. If None, computed from the data.
            linthresh (float): The threshold around zero for symmetric log scaling.
            max_filters (int, optional): If provided, only visualize up to this many filters. Useful if the layer has many filters.
            show (bool): Whether to display the plot.
            save_path (str, optional): If provided, the figure is saved to this file.

        Returns:
            matplotlib.figure.Figure: The generated figure.
        """
        # Access the specified layer.
        try:
            layer = self.backbone.layers[layer_idx]
        except IndexError:
            raise ValueError(f"Layer index {layer_idx} is out of range.")
        if not hasattr(layer.mixer, "conv1d"):
            raise ValueError(f"Layer {layer_idx}'s mixer does not have an attribute 'conv1d'.")

        # Extract the conv1d weight.
        conv_layer = layer.mixer.conv1d
        weight = conv_layer.weight.detach().cpu().numpy()  # shape: [out_channels, in_channels, kernel_size]
        out_channels, in_channels, kernel_size = weight.shape

        # Optionally limit the number of filters to visualize.
        if max_filters is not None and max_filters < out_channels:
            # For simplicity, select the first max_filters filters.
            weight = weight[:max_filters]
            out_channels = max_filters

        # Compute global min/max if not provided.
        if vmin is None:
            vmin = float(np.min(weight))
        if vmax is None:
            vmax = float(np.max(weight))

        # Set up normalization.
        norm = None
        if log_scale:
            norm = colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
            # norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)

        # Determine grid layout: aim for a roughly square grid.
        n_filters = out_channels
        n_cols = math.ceil(math.sqrt(n_filters))
        n_rows = math.ceil(n_filters / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        # Ensure axes is a 1D array for uniform handling.
        if n_filters == 1:
            axes = [axes]
        else:
            axes = np.array(axes).ravel()

        # Plot each filter.
        im = None
        for i in range(n_filters):
            ax = axes[i]
            # Each filter is a 2D array of shape [in_channels, kernel_size].
            filter_mat = weight[i, :, :]
            im = ax.imshow(filter_mat, cmap=cmap, norm=norm, vmin=None if norm else vmin,
                           vmax=None if norm else vmax)
            ax.set_title(f"Filter {i}")
            ax.set_xlabel("Kernel Pos")
            ax.set_ylabel("In Channels")

        # Turn off any extra axes if the grid has more cells than filters.
        for j in range(n_filters, len(axes)):
            axes[j].axis("off")

        # Add one common colorbar.
        fig.colorbar(im, ax=axes.tolist(), orientation="vertical")
        fig.suptitle(f"Conv1D Filter Visualization (Layer {layer_idx})", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight")
            print(f"Saved conv1d visualization to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig