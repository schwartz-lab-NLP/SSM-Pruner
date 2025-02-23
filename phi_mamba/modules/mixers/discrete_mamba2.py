from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from numpy.ma.core import indices
from torch import Tensor, dtype

from utils.config import Config

not_available_lambda = lambda *a, **k: NotImplementedError("This function is not available")

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = not_available_lambda, not_available_lambda

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
except ImportError:
    mamba_chunk_scan_combined = not_available_lambda

from modules.mixers.discrete_mamba2_ref import materialize_mixer, materialize_mixer_multi_value_attention_model

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = not_available_lambda
from einops import repeat


class Mixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=64,
            n_qk_heads=32,
            n_v_heads=32,
            d_conv=4,
            expand=1,
            activation="identity",
            bias=False,
            conv_bias=True,
            chunk_size=128,
            layer_idx=None,
            device=None,
            dtype=None,
            layer_mask_head_index=None,
            prune_dstates_ratio=None,
            d_inner=None,
            in_proj=None,
            out_proj=None,
            in_proj_bias=None,
            out_proj_bias=None,
            conv_dim=None,
            **kwargs,  # Absorb kwarg for general module
    ):
        """
        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args.
        Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr"

        Other options are all experimental and should not need to be configured
        """
        self.just_pruned = False

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model) if d_inner is None else d_inner
        self.n_qk_heads = n_qk_heads
        self.n_v_heads = n_v_heads
        self.headdim = self.d_inner // self.n_v_heads
        try:
            assert self.n_v_heads == self.d_inner // self.headdim, f"n_v_heads: {self.n_v_heads}, d_inner: {self.d_inner}, headdim: {self.headdim}"
            assert self.d_inner % self.headdim == 0, f"d_inner: {self.d_inner}, headdim: {self.headdim}"
            assert self.n_v_heads % self.n_qk_heads == 0, f"n_v_heads: {self.n_v_heads}, n_qk_heads: {self.n_qk_heads}"
        except AssertionError as e:
            print(e)
            print("Warning: Pruning may have caused the above assertion error !!!!!!!!!!!!!!!!!!!!!!!!!")

        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.bias = bias
        self.kwargs = kwargs

        # for in-place zeroing out of heads
        self.layer_mask_head_index = layer_mask_head_index
        # for in-place zeroing out of dstate
        self.prune_dstates_ratio = prune_dstates_ratio

        # Projections
        if in_proj is None:
            self.in_proj = nn.Linear(
                self.d_model,
                2 * self.d_inner + 2 * self.n_qk_heads * self.d_state + self.n_v_heads,
                bias=bias,
                **factory_kwargs,
            )
        else:
            self.in_proj = nn.Linear(in_proj['in_proj_in_features'], in_proj['in_proj_out_features'],
                                     bias=bias or in_proj_bias, **factory_kwargs)
        self.z_bias = (
            nn.Parameter(torch.zeros(self.d_inner, device=device)) if not bias else 0
        )  # make sure z_bias always exists

        # Convolutional layer
        conv_dim = self.d_inner + 2 * self.n_qk_heads * self.d_state if conv_dim is None else conv_dim
        self.conv_bias = conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # Activation after conv
        if self.activation == "identity":
            self.act = nn.Identity()
        elif self.activation in ["silu", "swish"]:
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation {self.activation}")

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.n_v_heads, device=device))
        self.D._optim = {"weight_decay": 0.0}

        # out_proj
        if out_proj is None:
            self.out_proj = nn.Linear(
                self.d_inner, self.d_model, bias=bias, **factory_kwargs
            )
        else:
            self.out_proj = nn.Linear(out_proj['out_proj_in_features'], out_proj['out_proj_out_features'],
                                      bias=bias or out_proj_bias, **factory_kwargs)

        self.disable_conv = False

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor

    def forward(self, u, return_mixer_matrix=False, inference_params=None,
                attention_mask: Optional[torch.Tensor] = None, layer_mask_head_index=None,
                **kwargs):
        """
        u: (batch_size, seq_len, model_dim)
        Returns: same shape as u
        """
        outputs = {}
        # assert state is None
        batch, seqlen, dim = u.shape

        state = None
        if inference_params is not None:
            state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # States are updated inplace
                out, _ = self.step(u, state, layer_mask_head_index=layer_mask_head_index)
                return {"hidden_states": out}

        # Hacky way to initialize state during inference
        chunk_size = self.chunk_size if state is None else seqlen

        # Pad input to nearest multiple of chunklen
        padded_len = (1 + (seqlen - 1) // chunk_size) * chunk_size
        u = F.pad(u, (0, 0, 0, padded_len - seqlen))

        # Attention mask
        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            dtype = u.dtype
            u = (u * attention_mask[:, :, None]).to(dtype)

        # Project input
        xBCzA_log = self.in_proj(u)
        xBC, z, A_log = torch.split(
            xBCzA_log,
            [
                self.d_inner + 2 * self.n_qk_heads * self.d_state,
                self.d_inner,
                self.n_v_heads,
            ],
            dim=-1,
        )


        if state is not None:
            # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            xBC_t = rearrange(xBC[:, :seqlen, :], "batch sequence_len dim -> batch dim sequence_len")
            state["conv"].copy_(
                F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
            )  # Update state (B D W)

        # Convolutional layer
        xBC = self.convolutional_forward(xBC, padded_len)

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_qk_heads * self.d_state,
                self.n_qk_heads * self.d_state,
            ],
            dim=-1,
        )

        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            # tune out hidden states for pad tokens, see https://github.com/state-spaces/mamba/issues/66
            dtype = x.dtype
            try:
                x = (x * attention_mask[:, :, None]).to(dtype)
            except:
                print('attention_mask shape:', attention_mask.shape)
                print('x shape:', x.shape)
                raise

        x = rearrange(x, "batch seq_len (num_heads channels_per_head) -> batch seq_len num_heads channels_per_head",
                      num_heads=self.n_v_heads)
        B = rearrange(B, "batch seq_len (num_heads channels_per_head) -> batch seq_len num_heads channels_per_head",
                      num_heads=self.n_qk_heads)
        C = rearrange(C, "batch seq_len (num_heads channels_per_head) -> batch seq_len num_heads channels_per_head",
                      num_heads=self.n_qk_heads)

        if not self.just_pruned and (layer_mask_head_index is not None or self.layer_mask_head_index is not None):
            layer_mask_head_index = layer_mask_head_index if layer_mask_head_index is not None else self.layer_mask_head_index
            x, z, B, C, self.D, A_log = self.zero_out_head(x, z, B, C, layer_mask_head_index, D=self.D, A_log=A_log)

        if not self.just_pruned and self.prune_dstates_ratio is not None:
            # Prune the lowest percentage of d_state dimensions per head
            ratio = self.prune_dstates_ratio  # This should be a value between 0 and 1
            num_to_zero = int(self.d_state * ratio)
            zeroed_indices = []
            if num_to_zero > 0:
                for h in range(self.n_qk_heads):
                    # Compute magnitudes for B and C per d_state dimension
                    B_h = B[:, :, h, :]  # (batch, length, d_state)
                    C_h = C[:, :, h, :]  # (batch, length, d_state)
                    magnitudes = (B_h.abs() + C_h.abs()).sum(dim=(0, 1))  # (d_state,)

                    # Get indices to zero out
                    _, indices = torch.sort(magnitudes)  # ascending order
                    indices_to_zero = indices[:num_to_zero]
                    zeroed_indices.extend(indices_to_zero.tolist())
                    # Zero out B and C at those indices for head h
                    B[:, :, h, indices_to_zero] = 0
                    C[:, :, h, indices_to_zero] = 0

        # SSM forward
        # import pdb; pdb.set_trace()
        result = mamba_chunk_scan_combined(
            x=x / F.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=A_log,
            dt_softplus=True,
            A=-torch.ones(self.n_v_heads, device=A_log.device),
            B=B,
            C=C,
            chunk_size=chunk_size,
            # initial_states=(state["ssm"] if state is not None else None), # currently not supported by mamba_ssm.utils.generation
            return_final_states=(state is not None),
        )


        if state is not None:
            y, ssm_state = result
            # print('-'*100, ssm_state, type(ssm_state), ssm_state.shape if hasattr(ssm_state, 'shape') else None)
            state["ssm"].copy_(ssm_state)
        else:
            y = result

        Du = torch.einsum("h,blhp->blhp", self.D, x)
        y = rearrange(y + Du, "batch seq_len heads channels_per_head -> batch seq_len (heads channels_per_head)")

        # Norm and gate
        out = self.out_proj(y * F.silu(z + self.z_bias))
        outputs["hidden_states"] = out[:, :seqlen, :]

        if return_mixer_matrix:
            if self.n_qk_heads == self.n_v_heads:
                outputs["transfer_matrix"] = materialize_mixer(
                    A_log=A_log, B=B, C=C, D=self.D
                )[..., :seqlen, :seqlen]
            else:
                outputs["transfer_matrix"] = materialize_mixer_multi_value_attention_model(
                    A_log=A_log, B=B, C=C, D=self.D, n_qk_heads=self.n_qk_heads, n_v_heads=self.n_v_heads
                )[..., :seqlen, :seqlen]
        return outputs

    def step(self, u, state, layer_mask_head_index=None, **kwargs):
        """
        u: (batch_size, model_dim)
        state: dict of states
        Returns: same shape as u
        """
        # Project input
        xBCzA_log = self.in_proj(u.squeeze(1))
        xBC, z, A_log = torch.split(
            xBCzA_log,
            [
                self.d_inner + 2 * self.n_qk_heads * self.d_state,
                self.d_inner,
                self.n_v_heads,
            ],
            dim=-1,
        )

        xBC, conv_state = self.convolutional_step(xBC, state["conv"])
        state["conv"].copy_(conv_state)  # update state in place

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_qk_heads * self.d_state,
                self.n_qk_heads * self.d_state,
            ],
            dim=-1,
        )

        x = rearrange(x, "batch (num_heads dim_per_head) -> batch num_heads dim_per_head", num_heads=self.n_v_heads)
        B = rearrange(B, "batch (num_heads dim_per_head) -> batch num_heads dim_per_head", num_heads=self.n_qk_heads)
        C = rearrange(C, "batch (num_heads dim_per_head) -> batch num_heads dim_per_head", num_heads=self.n_qk_heads)

        if not self.just_pruned and (layer_mask_head_index is not None or self.layer_mask_head_index is not None):
            layer_mask_head_index = layer_mask_head_index if layer_mask_head_index is not None else self.layer_mask_head_index
            x, z, B, C, self.D, A_log = self.zero_out_head(x, z, B, C, layer_mask_head_index, D=self.D, A_log=A_log)

        if not self.just_pruned and self.prune_dstates_ratio is not None:
            # Prune the lowest percentage of d_state dimensions per head
            ratio = self.prune_dstates_ratio  # This should be a value between 0 and 1
            num_to_zero = int(self.d_state * ratio)
            zeroed_indices = []
            if num_to_zero > 0:
                for h in range(self.n_qk_heads):
                    # Compute magnitudes for B and C per d_state dimension
                    B_h = B[:, :, h, :]  # (batch, length, d_state)
                    C_h = C[:, :, h, :]  # (batch, length, d_state)
                    magnitudes = (B_h.abs() + C_h.abs()).sum(dim=(0, 1))  # (d_state,)

                    # Get indices to zero out
                    _, indices = torch.sort(magnitudes)  # ascending order
                    indices_to_zero = indices[:num_to_zero]
                    zeroed_indices.extend(indices_to_zero.tolist())
                    # Zero out B and C at those indices for head h
                    B[:, :, h, indices_to_zero] = 0
                    C[:, :, h, indices_to_zero] = 0

        state["ssm"] = state["ssm"].to(x.dtype)
        zeros = torch.zeros((self.n_v_heads, self.headdim), device=A_log.device).to(dtype=x.dtype)
        ones = torch.ones((self.n_v_heads, self.headdim, self.d_state), device=A_log.device).to(dtype=x.dtype)
        y = selective_state_update(
            x=x / F.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=repeat(A_log, "batch heads -> batch heads dim_per_head", dim_per_head=self.headdim),
            dt_softplus=True,
            A=-ones,
            B=B,
            C=C,
            state=state["ssm"],  # will be updated in place
            dt_bias=zeros,
            D=zeros,
        )

        y = y + self.D[:, None] * x
        y = rearrange(y, "batch heads dim_per_head -> batch (heads dim_per_head)")

        # Norm and gate
        out = self.out_proj(y * F.silu(z + self.z_bias))

        return out, state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.in_proj.weight.device
        # conv_state:
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_conv,
            self.conv1d.weight.shape[0],
            device=device,
            dtype=conv_dtype,
        ).transpose(1, 2)
        # ssm_state:
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.n_v_heads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return {"conv": conv_state, "ssm": ssm_state}

    def _get_states_from_cache(
            self, inference_params, batch_size, initialize_states=False
    ):
        """
        conv_state: (batch_size, self.d_conv, conv1d.weight.shape[0])
        ssm_state: (batch_size, n_qk_heads, headdim, d_state)
        """
        assert self.layer_idx is not None
        # Allocate memory if not exists
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[
                self.layer_idx
            ] = self.allocate_inference_cache(
                batch_size, inference_params.max_seqlen, dtype=torch.float32
            )
        # Get states
        states = inference_params.key_value_memory_dict[self.layer_idx]
        if initialize_states:
            states["conv"].zero_()
            states["ssm"].zero_()
        return states

    def convolutional_forward(self, xBC, padded_len):
        if self.disable_conv:
            return xBC
        if causal_conv1d_fn is None or self.activation not in [
            "silu",
            "swish",
            "identity",
        ]:
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2))[..., :padded_len].transpose(1, 2)
            )
        else:
            # print('-'*100,'xBC shape:', xBC.shape, padded_len)
            xBC = xBC.contiguous()
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "out_channels 1 kernel_size -> out_channels kernel_size"),
                self.conv1d.bias,
                activation=None if self.activation == "identity" else self.activation,
            ).transpose(1, 2)

        return xBC

    def convolutional_step(self, xBC, conv_state):
        if self.disable_conv:
            return xBC, conv_state
        # Convolutional layer
        conv_state = conv_state.to(xBC.dtype)
        if causal_conv1d_update:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "out_channels 1 kernel_size -> out_channels kernel_size"),
                self.conv1d.bias,
                self.activation if self.activation != "identity" else None,
            )
        else:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "out_channels 1 kernel_size -> out_channels kernel_size"),
                dim=-1
            )  # (B D)
            if self.conv_bias:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(xBC.dtype)  # Some activations change dtype

        return xBC, conv_state

    ################################### Pruning methods ########################################

    @torch.no_grad()
    def prune_dstate_and_dinner(self, ratio):
        """
        Prune the model by reducing d_inner to d_inner_new and d_state according to the given ratio.
        Adjusts in_proj, out_proj, conv1d, z_bias, and related parameters accordingly.
        """

        # Calculate the number of units to prune in d_inner

        # Prune d_inner units
        self.prune_dinner(ratio)

        # Prune d_state dimensions
        if ratio > 0:
            self.prune_dstates(ratio, method='taylor_second')

        # Update the flag
        self.just_pruned = True

    @torch.no_grad()
    def prune_dinner(self, ratio):
        """
        Prune the least important units from d_inner, adjusting in_proj, out_proj, conv1d, and z_bias.
        Importance is estimated based on the magnitude of in_proj and out_proj weights.
        """
        if ratio == 0:
            return

        num_to_prune = int(self.d_inner * ratio)
        d_inner_new = self.d_inner - num_to_prune
        assert d_inner_new > 0, "d_inner must be positive after pruning"

        # Compute importance scores for each unit in d_inner

        importance_scores_out, importance_scores_x, importance_scores_z = self.dinner_taylor_importance_estimator()

        # Get indices of units to prune
        _, indices_x = torch.sort(importance_scores_x)  # Ascending order
        _, indices_z = torch.sort(importance_scores_z)  # Ascending order
        _, indices_out = torch.sort(importance_scores_out)  # Ascending order
        indices_to_prune_x = indices_x[:num_to_prune].tolist()
        indices_to_prune_z = indices_z[:num_to_prune].tolist()
        indices_to_prune_out = indices_out[:num_to_prune].tolist()
        indices_to_keep_z_bias = indices_z[num_to_prune:].tolist()

        # Build mask for in_proj weights and biases
        mask_in_proj = torch.ones(self.in_proj.out_features, dtype=torch.bool, device=self.in_proj.weight.device)

        z_start = self.d_inner + 2 * self.n_qk_heads * self.d_state
        # Indices to prune from x in in_proj
        for idx in indices_to_prune_x:
            mask_in_proj[idx] = False  # x indices
        for idx in indices_to_prune_z:
            mask_in_proj[z_start + idx] = False  # z indices

        # Update in_proj weights and biases
        # preserve acc_grad if existing
        tmp_in_proj_acc_grad = None
        if hasattr(self.in_proj.weight, 'acc_grad'):
            tmp_in_proj_acc_grad = self.in_proj.weight.acc_grad[mask_in_proj]

        self.in_proj.weight = nn.Parameter(self.in_proj.weight[mask_in_proj, :])

        if tmp_in_proj_acc_grad is not None:
            self.in_proj.weight.acc_grad = tmp_in_proj_acc_grad

        if self.in_proj.bias is not None:
            self.in_proj.bias = nn.Parameter(self.in_proj.bias[mask_in_proj])

        # Adjust in_proj output features
        self.in_proj.out_features = self.in_proj.weight.shape[0]

        # Update conv1d
        conv_dim_old = self.conv1d.weight.shape[0]
        conv_mask = torch.ones(conv_dim_old, dtype=torch.bool, device=self.conv1d.weight.device)
        # Indices to prune in conv1d (x indices)
        for idx in indices_to_prune_x + indices_to_prune_z:
            conv_mask[idx] = False

        # Create new conv1d layer
        conv_dim_new = conv_mask.sum().item()
        factory_kwargs = {"device": self.conv1d.weight.device, "dtype": self.conv1d.weight.dtype}
        new_conv = nn.Conv1d(
            in_channels=conv_dim_new,
            out_channels=conv_dim_new,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=conv_dim_new,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )

        # Transfer weights and biases
        new_conv.weight.data.copy_(self.conv1d.weight.data[conv_mask, :, :])
        if self.conv1d.bias is not None:
            new_conv.bias.data.copy_(self.conv1d.bias.data[conv_mask])

        self.conv1d = new_conv

        # Update z_bias
        self.z_bias = nn.Parameter(self.z_bias[indices_to_keep_z_bias])

        # Update out_proj
        mask_out_proj = torch.ones(self.out_proj.in_features, dtype=torch.bool, device=self.out_proj.weight.device)
        for idx in indices_to_prune_out:
            mask_out_proj[idx] = False

        #preserve acc_grad if existing
        tmp_out_proj_acc_grad = None
        if hasattr(self.out_proj.weight, 'acc_grad'):
            tmp_out_proj_acc_grad = self.out_proj.weight.acc_grad[mask_out_proj]
        self.out_proj.weight = nn.Parameter(self.out_proj.weight[:, mask_out_proj])
        if tmp_out_proj_acc_grad is not None:
            self.out_proj.weight.acc_grad = tmp_out_proj_acc_grad
        if self.out_proj.bias is not None:
            self.out_proj.bias = nn.Parameter(self.out_proj.bias)

        # Adjust out_proj input features
        self.out_proj.in_features = self.out_proj.weight.shape[1]

        # Update d_inner
        self.d_inner = d_inner_new

        # Update headdim
        self.headdim = self.d_inner // self.n_v_heads
        assert self.d_inner % self.n_v_heads == 0, "d_inner must be divisible by n_v_heads"

    def dinnner_importance_estimate_magnitude(self):
        # For x in in_proj.weight[:d_inner, :]
        x_weights = self.in_proj.weight[:self.d_inner, :]  # Shape: (d_inner, d_model)
        # For z in in_proj.weight[z_indices, :]
        xBC_end = self.d_inner + 2 * self.n_qk_heads * self.d_state
        z_start = xBC_end
        z_end = z_start + self.d_inner
        z_weights = self.in_proj.weight[z_start:z_end, :]  # Shape: (d_inner, d_model)

        # For out_proj.weight of shape (d_model, d_inner)
        out_proj_weights = self.out_proj.weight  # Shape: (d_model, d_inner)
        # Sum the magnitudes
        importance_scores_x = (
            x_weights.abs().sum(dim=1)
        )  # Shape: (d_inner,)
        importance_scores_z = (
            z_weights.abs().sum(dim=1)
        )  # Shape: (d_inner,)
        importance_scores_out = (
            out_proj_weights.abs().sum(dim=0)
        )  # Shape: (d_inner,)
        return importance_scores_out, importance_scores_x, importance_scores_z

    def dinnner_importance_estimate_magnitude_summed(self):
        # Sum the magnitudes
        # For x in in_proj.weight[:d_inner, :]
        x_weights = self.in_proj.weight[:self.d_inner, :]  # Shape: (d_inner, d_model)
        # For z in in_proj.weight[z_indices, :]
        xBC_end = self.d_inner + 2 * self.n_qk_heads * self.d_state
        z_start = xBC_end
        z_end = z_start + self.d_inner
        z_weights = self.in_proj.weight[z_start:z_end, :]  # Shape: (d_inner, d_model)

        # For out_proj.weight of shape (d_model, d_inner)
        out_proj_weights = self.out_proj.weight  # Shape: (d_model, d_inner)

        importance_scores_x = (
            x_weights.abs().sum(dim=1)
        )  # Shape: (d_inner,)
        importance_scores_z = (
            z_weights.abs().sum(dim=1)
        )  # Shape: (d_inner,)
        importance_scores_out = (
            out_proj_weights.abs().sum(dim=0)
        )  # Shape: (d_inner,)
        summed = importance_scores_x + importance_scores_z + importance_scores_out
        return summed, summed, summed

    def dinner_taylor_importance_estimator(self):
        """
        Compute the importance of each d_inner unit based on the Taylor expansion of the output.
        """
        # For x in in_proj.weight[:d_inner, :]
        x_weights = self.in_proj.weight[:self.d_inner, :]  # Shape: (d_inner, d_model)
        x_acc = self.in_proj.weight.acc_grad[:self.d_inner, :]
        # For z in in_proj.weight[z_indices, :]
        xBC_end = self.d_inner + 2 * self.n_qk_heads * self.d_state
        z_start = xBC_end
        z_end = z_start + self.d_inner

        z_weights = self.in_proj.weight[z_start:z_end, :]  # Shape: (d_inner, d_model)
        z_acc = self.in_proj.weight.acc_grad[z_start:z_end, :]

        # For out_proj.weight of shape (d_model, d_inner)
        out_proj_weights = self.out_proj.weight  # Shape: (d_model, d_inner)
        out_proj_acc = self.out_proj.weight.acc_grad

        # Compute the salience of each d_inner unit
        x_salience = x_weights * x_acc * x_weights
        z_salience = z_weights * z_acc * z_weights
        out_proj_salience = out_proj_weights * out_proj_acc * out_proj_weights

        x_taylor_imp = x_salience.abs().sum(dim=1)  # (d_inner,)
        z_taylor_imp = z_salience.abs().sum(dim=1)  # (d_inner,)
        out_proj_taylor_imp = out_proj_salience.abs().sum(dim=0)  # (d_inner,)
        taylor_imp = x_taylor_imp + z_taylor_imp + out_proj_taylor_imp
        _, indices = torch.sort(taylor_imp)  # ascending order
        return indices, indices, indices

    def outproj_importance_estimator(self):
        """
        Compute the importance of each tensor inside the heads starting from x_head_start and z_head_start based on the Taylor importance metric
        """
        x_weights = self.out_proj.weight
        x_acc = self.out_proj.weight.acc_grad
        x_salience = x_weights * x_acc * x_weights

        x_taylor_imp = x_salience.abs().sum(dim=1)  # (dinner,)

        # sort
        _, indices = torch.sort(x_taylor_imp)  # ascending order
        return indices

    def headdim_taylor_second_indp_XZ_importance_estimator(self, x_head_start, z_head_start):
        """
        Compute the importance of each tensor inside the heads starting from x_head_start and z_head_start based on the Taylor importance metric
        """
        x_head_weights = self.in_proj.weight[x_head_start:x_head_start + self.headdim, :]
        x_head_acc = self.in_proj.weight.acc_grad[x_head_start:x_head_start + self.headdim, :]
        z_head_weights = self.in_proj.weight[z_head_start:z_head_start + self.headdim, :]
        z_head_acc = self.in_proj.weight.acc_grad[z_head_start:z_head_start + self.headdim, :]

        x_head_salience = x_head_weights * x_head_acc * x_head_weights
        z_head_salience = z_head_weights * z_head_acc * z_head_weights

        x_head_taylor_imp = x_head_salience.abs().sum(dim=1)  # (headdim,)
        z_head_taylor_imp = z_head_salience.abs().sum(dim=1)  # (headdim,)

        # sort
        _, indices_x = torch.sort(x_head_taylor_imp)  # ascending order
        _, indices_z = torch.sort(z_head_taylor_imp)  # ascending order
        return indices_x, indices_z

    def dstates_taylor_second_indp_BC_importance_estimator(self, *args):
        B_taylor_imp, C_taylor_imp = self.dstate_taylor_importance_aux(2, *args)
        taylor_imp = B_taylor_imp + C_taylor_imp
        _, indices_B = _, indices_C = torch.sort(taylor_imp)  # ascending order
        return indices_B, indices_C

    def dstates_taylor_first_indp_BC_importance_estimator(self, *args):
        B_taylor_imp, C_taylor_imp = self.dstate_taylor_importance_aux(1, *args)
        taylor_imp = B_taylor_imp + C_taylor_imp
        _, indices_B = _, indices_C = torch.sort(taylor_imp)  # ascending order
        return indices_B, indices_C

    def dstates_taylor_first_by_B_importance_estimator(self, *args):
        B_taylor_imp, _ = self.dstate_taylor_importance_aux(1, *args)
        taylor_imp = B_taylor_imp
        _, indices_B = _, indices_C = torch.sort(taylor_imp)  # ascending order
        return indices_B, indices_C

    def dstates_taylor_first_by_C_importance_estimator(self, *args):
        _, C_taylor_imp = self.dstate_taylor_importance_aux(1, *args)
        taylor_imp = C_taylor_imp
        _, indices_B = _, indices_C = torch.sort(taylor_imp)  # ascending order
        return indices_B, indices_C

    def dstates_taylor_second_by_B_importance_estimator(self, *args):
        B_taylor_imp, _ = self.dstate_taylor_importance_aux(2, *args)
        taylor_imp = B_taylor_imp
        _, indices_B = _, indices_C = torch.sort(taylor_imp)  # ascending order
        return indices_B, indices_C

    def dstates_taylor_second_by_C_importance_estimator(self, *args):
        _, C_taylor_imp = self.dstate_taylor_importance_aux(2, *args)
        taylor_imp = C_taylor_imp
        _, indices_B = _, indices_C = torch.sort(taylor_imp)  # ascending order
        return indices_B, indices_C

    def dstate_taylor_importance_aux(self, order, *args):
        idx_B_start = args[2]
        idx_C_start = args[3]
        B_weights_acc_hess = C_weights_acc_hess = None
        B_weights = self.in_proj.weight[idx_B_start:idx_B_start + self.d_state, :]
        B_weights_acc_grad = self.in_proj.weight.acc_grad[idx_B_start:idx_B_start + self.d_state, :]
        if hasattr(self.in_proj.weight, 'acc_hess'):
            B_weights_acc_hess = self.in_proj.weight.acc_hess[idx_B_start:idx_B_start + self.d_state, :]
        B_salience = (B_weights * B_weights_acc_grad)

        C_weights = self.in_proj.weight[idx_C_start:idx_C_start + self.d_state, :]
        C_weights_acc_grad = self.in_proj.weight.acc_grad[idx_C_start:idx_C_start + self.d_state, :]

        if hasattr(self.in_proj.weight, 'acc_hess'):
            C_weights_acc_hess = self.in_proj.weight.acc_hess[idx_C_start:idx_C_start + self.d_state, :]
        C_salience = (C_weights * C_weights_acc_grad)

        if order == 2:
            if not hasattr(self.in_proj.weight, 'acc_hess'):
                B_salience = B_salience * B_weights
                C_salience = C_salience * C_weights
            else:
                B_salience = B_salience + 0.5 * B_weights_acc_hess * (B_weights ** 2)
                C_salience = C_salience + 0.5 * C_weights_acc_hess * (C_weights ** 2)

        B_taylor_imp = B_salience.abs().mean(dim=1)  # (d_state,)
        C_taylor_imp = C_salience.abs().mean(dim=1)  # (d_state,)
        return B_taylor_imp, C_taylor_imp

    def BC_heads_taylor_importance_estimator(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the importance of each B and C head based on the Taylor expansion of the output.
        """
        in_proj_indices = self.get_in_proj_indices()
        B_proj_start, B_proj_end = in_proj_indices['B']
        C_proj_start, C_proj_end = in_proj_indices['C']
        B_proj = self.in_proj.weight[B_proj_start:B_proj_end, :]
        B_acc = self.in_proj.weight.acc_grad[B_proj_start:B_proj_end, :]
        C_proj = self.in_proj.weight[C_proj_start:C_proj_end, :]
        C_acc = self.in_proj.weight.acc_grad[C_proj_start:C_proj_end, :]

        print(B_proj.shape)
        B_proj = rearrange(B_proj, "(n_qk_heads d_state) d_model -> n_qk_heads d_model d_state",
                           n_qk_heads=self.n_qk_heads)
        B_acc = rearrange(B_acc, "(n_qk_heads d_state) d_model -> n_qk_heads d_model d_state",
                          n_qk_heads=self.n_qk_heads)
        print(B_proj.shape)
        C_proj = rearrange(C_proj, "(n_qk_heads d_state) d_model -> n_qk_heads d_model d_state",
                           n_qk_heads=self.n_qk_heads)
        C_acc = rearrange(C_acc, "(n_qk_heads d_state) d_model -> n_qk_heads d_model d_state",
                          n_qk_heads=self.n_qk_heads)

        B_salience = B_proj * B_acc * B_proj
        # B_taylor_imp = B_salience.abs().sum(dim=(1, 2))  # (n_qk_heads,)
        B_taylor_imp = B_salience.abs().mean(dim=2).sum(dim=1)  # (n_qk_heads,)

        C_salience = C_proj * C_acc * C_proj
        # C_taylor_imp = C_salience.abs().sum(dim=(1, 2))  # (n_qk_heads,)
        C_taylor_imp = C_salience.abs().mean(dim=2).sum(dim=1)  # (n_qk_heads,)

        BC_heads_taylor_imp = B_taylor_imp + C_taylor_imp

        values_BC_heads, indices_BC_heads = torch.sort(BC_heads_taylor_imp, dim=0, descending=False)
        print('indices_BC_heads:', indices_BC_heads)
        print('values_BC_heads:', values_BC_heads)
        return values_BC_heads, indices_BC_heads

    def dstates_magnitude_importance_estimator_summed(self, B_weights, C_weights, *args):
        """
        Compute the importance of each d_state dimension based on the magnitudes of the B and C weights.
        """
        magnitudes = (B_weights.abs() + C_weights.abs()).sum(dim=1)  # (d_state,)

        # Get indices to prune
        _, indices = torch.sort(magnitudes)  # ascending order
        return indices, indices

    def dstates_random_importance_estimator(self, B_weights, *args):
        """
        Assign random importance of each d_state dimension
        """
        d_state = B_weights.shape[0]
        indices = torch.randperm(d_state)
        return indices, indices

    def dstates_magnitude_importance_estimator(self, B_weights, C_weights, *args):
        """
        Compute the importance of each d_state dimension based on the magnitudes of the B and C weights.
        """
        ## use L2 norm
        magnitudes_B = B_weights.abs().pow(2).sum(dim=1)  # (d_state,)
        magnitudes_C = C_weights.abs().pow(2).sum(dim=1)  # (d_state,)

        # Get indices to prune
        _, indices_B = torch.sort(magnitudes_B)  # ascending order
        _, indices_C = torch.sort(magnitudes_C)
        return indices_B, indices_C

    def _mean_pool_dstates(self, num_to_prune) -> \
            tuple[Tensor, Tensor, Any | None, Any | None, Tensor, Tensor, Tensor | None, Tensor | None]:
        """
        mean pool the consecutive dstates from BC heads by grouping them in self.dstates // num_to_prune groups
        """
        assert self.d_state % num_to_prune == 0, f"d_state must be divisible by num_to_prune - {self.d_state} % {num_to_prune} != 0"
        group_size = self.d_state // num_to_prune
        B_weight = self.in_proj.weight[self.d_inner:self.d_inner + self.n_qk_heads * self.d_state, :]
        C_weight = self.in_proj.weight[
                   self.d_inner + self.n_qk_heads * self.d_state:self.d_inner + 2 * self.n_qk_heads * self.d_state, :]
        B_weight = rearrange(B_weight, "(n_qk_heads d_state) d_model -> n_qk_heads d_model d_state",
                             n_qk_heads=self.n_qk_heads)
        C_weight = rearrange(C_weight, "(n_qk_heads d_state) d_model -> n_qk_heads d_model d_state",
                             n_qk_heads=self.n_qk_heads)
        B_weight = rearrange(B_weight,
                             "n_qk_heads d_model (new_d_state group_size) -> n_qk_heads d_model new_d_state group_size",
                             group_size=group_size)
        C_weight = rearrange(C_weight,
                             "n_qk_heads d_model (new_d_state group_size) -> n_qk_heads d_model new_d_state group_size",
                             group_size=group_size)
        B_weight = B_weight.mean(dim=-1)  # (n_qk_heads, d_model, d_state)
        C_weight = C_weight.mean(dim=-1)  # (n_qk_heads, d_model, d_state)
        B_weight = rearrange(B_weight, "n_qk_heads d_model new_d_state -> (n_qk_heads new_d_state) d_model")
        C_weight = rearrange(C_weight, "n_qk_heads d_model new_d_state -> (n_qk_heads new_d_state) d_model")

        B_bias, C_bias = None, None
        if self.in_proj.bias is not None:
            B_bias = self.in_proj.bias[self.d_inner:self.d_inner + self.n_qk_heads * self.d_state]
            C_bias = self.in_proj.bias[
                     self.d_inner + self.n_qk_heads * self.d_state:self.d_inner + 2 * self.n_qk_heads * self.d_state]
            B_bias = rearrange(B_bias, "(n_qk_heads d_state) -> n_qk_heads d_state")
            C_bias = rearrange(C_bias, "(n_qk_heads d_state) -> n_qk_heads d_state")
            B_bias = rearrange(B_bias, "n_qk_heads (new_d_state group_size) -> n_qk_heads new_d_state group_size",
                               group_size=group_size)
            C_bias = rearrange(C_bias, "n_qk_heads (new_d_state group_size) -> n_qk_heads new_d_state group_size",
                               group_size=group_size)
            B_bias = B_bias.mean(dim=-1)
            C_bias = C_bias.mean(dim=-1)
            B_bias = rearrange(B_bias, "n_qk_heads new_d_state -> (n_qk_heads new_d_state)")
            C_bias = rearrange(C_bias, "n_qk_heads new_d_state -> (n_qk_heads new_d_state)")

        conv_weight_B = self.conv1d.weight[self.d_inner:self.d_inner + self.n_qk_heads * self.d_state, :]
        conv_weight_C = self.conv1d.weight[
                        self.d_inner + self.n_qk_heads * self.d_state:self.d_inner + 2 * self.n_qk_heads * self.d_state,
                        :]
        conv_weight_B = rearrange(conv_weight_B,
                                  "(n_qk_heads d_state) 1 kernel_size -> n_qk_heads 1 kernel_size d_state",
                                  n_qk_heads=self.n_qk_heads)
        conv_weight_C = rearrange(conv_weight_C,
                                  "(n_qk_heads d_state) 1 kernel_size -> n_qk_heads 1 kernel_size d_state",
                                  n_qk_heads=self.n_qk_heads)
        conv_weight_B = rearrange(conv_weight_B,
                                  "n_qk_heads 1 kernel_size (new_d_state group_size) -> n_qk_heads 1 kernel_size new_d_state group_size",
                                  group_size=group_size)
        conv_weight_C = rearrange(conv_weight_C,
                                  "n_qk_heads 1 kernel_size (new_d_state group_size) -> n_qk_heads 1 kernel_size new_d_state group_size",
                                  group_size=group_size)
        conv_weight_B = conv_weight_B.mean(dim=-1)
        conv_weight_C = conv_weight_C.mean(dim=-1)
        conv_weight_B = rearrange(conv_weight_B,
                                  "n_qk_heads 1 kernel_size new_d_state -> (n_qk_heads new_d_state) 1 kernel_size")
        conv_weight_C = rearrange(conv_weight_C,
                                  "n_qk_heads 1 kernel_size new_d_state -> (n_qk_heads new_d_state) 1 kernel_size")
        conv_bias_B, conv_bias_C = None, None
        if self.conv1d.bias is not None:
            conv_bias_B = self.conv1d.bias[self.d_inner:self.d_inner + self.n_qk_heads * self.d_state]
            conv_bias_C = self.conv1d.bias[
                          self.d_inner + self.n_qk_heads * self.d_state:self.d_inner + 2 * self.n_qk_heads * self.d_state]
            conv_bias_B = rearrange(conv_bias_B, "(n_qk_heads d_state) -> n_qk_heads d_state",
                                    n_qk_heads=self.n_qk_heads)
            conv_bias_C = rearrange(conv_bias_C, "(n_qk_heads d_state) -> n_qk_heads d_state",
                                    n_qk_heads=self.n_qk_heads)
            conv_bias_B = rearrange(conv_bias_B,
                                    "n_qk_heads (new_d_state group_size) -> n_qk_heads new_d_state group_size",
                                    group_size=group_size)
            conv_bias_C = rearrange(conv_bias_C,
                                    "n_qk_heads (new_d_state group_size) -> n_qk_heads new_d_state group_size",
                                    group_size=group_size)
            conv_bias_B = conv_bias_B.mean(dim=-1)
            conv_bias_C = conv_bias_C.mean(dim=-1)
            conv_bias_B = rearrange(conv_bias_B, "n_qk_heads new_d_state -> (n_qk_heads new_d_state)")
            conv_bias_C = rearrange(conv_bias_C, "n_qk_heads new_d_state -> (n_qk_heads new_d_state)")

        return B_weight, C_weight, B_bias, C_bias, conv_weight_B, conv_weight_C, conv_bias_B, conv_bias_C

    @torch.no_grad()
    def prune_dstates(self, ratio, method="taylor_second", return_plot=False):
        """
        Prune the lowest magnitude d_state dimensions per head by removing their parameters according to the given ratio.
        The same number of elements are removed from each head, and d_state is reduced accordingly.
        """
        # importance_estimator = self.dstates_magnitude_importance_estimator_summed
        if method == "taylor_first_B":
            importance_estimator = self.dstates_taylor_first_by_B_importance_estimator
        elif method == "taylor_first_C":
            importance_estimator = self.dstates_taylor_first_by_C_importance_estimator
        elif method == "taylor_second_B":
            importance_estimator = self.dstates_taylor_second_by_B_importance_estimator
        elif method == "taylor_second_C":
            importance_estimator = self.dstates_taylor_second_by_C_importance_estimator
        elif method == "taylor_first":
            importance_estimator = self.dstates_taylor_first_indp_BC_importance_estimator
        elif method == "taylor_second":
            importance_estimator = self.dstates_taylor_second_indp_BC_importance_estimator
        elif method == 'magnitude':
            importance_estimator = self.dstates_magnitude_importance_estimator_summed
        elif method == 'random':
            importance_estimator = self.dstates_random_importance_estimator
        elif method == 'mean_pooling':
            importance_estimator = self.dstates_random_importance_estimator

        num_to_prune = int(self.d_state * ratio)
        if method == 'mean_pooling':
            assert self.d_state % num_to_prune == 0, f"d_state must be divisible by num_to_prune - {self.d_state} % {num_to_prune} != 0"

        if num_to_prune == 0:
            return  # Nothing to prune

        indices_to_prune = []
        indices_to_prune_conv = []
        plots = []
        for h in range(self.n_qk_heads):
            # Indices in in_proj corresponding to B and C for head h
            idx_B_start = self.d_inner + h * self.d_state
            idx_C_start = self.d_inner + self.n_qk_heads * self.d_state + h * self.d_state

            # Get B and C weights for head h
            B_weights = self.in_proj.weight[idx_B_start:idx_B_start + self.d_state, :]  # (d_state, d_model)
            C_weights = self.in_proj.weight[idx_C_start:idx_C_start + self.d_state, :]  # (d_state, d_model)

            indices_B, indices_C = importance_estimator(B_weights, C_weights, idx_B_start, idx_C_start)
            indices_prune_head_B = indices_B[:num_to_prune].tolist()
            indices_prune_head_C = indices_C[:num_to_prune].tolist()

            # Map indices to global indices in in_proj
            # For B
            indices_to_prune.extend([idx_B_start + i for i in indices_prune_head_B])
            # For C
            indices_to_prune.extend([idx_C_start + i for i in indices_prune_head_C])

            # Now, for conv1d channels
            idx_B_conv = self.d_inner + h * self.d_state
            idx_C_conv = self.d_inner + self.n_qk_heads * self.d_state + h * self.d_state

            indices_to_prune_conv.extend([idx_B_conv + i for i in indices_prune_head_B])
            indices_to_prune_conv.extend([idx_C_conv + i for i in indices_prune_head_C])

        # Build mask for in_proj.weight and in_proj.bias
        mask_in_proj = torch.ones(self.in_proj.out_features, dtype=torch.bool, device=self.in_proj.weight.device)
        for idx in indices_to_prune:
            mask_in_proj[idx] = False

        if return_plot:
            plots.append(mask_in_proj)

        if method != 'mean_pooling':
            pruned_in_proj_weight = self.in_proj.weight[mask_in_proj, :]
            if self.in_proj.bias is not None:
                pruned_in_proj_bias = self.in_proj.bias[mask_in_proj]
        else:
            mean_pooled_BC_dstates = self._mean_pool_dstates(num_to_prune)
            B_weights, C_weights, B_bias, C_bias, conv_weight_B, conv_weight_C, conv_bias_B, conv_bias_C = mean_pooled_BC_dstates
            pruned_in_proj_weight = self.in_proj.weight[mask_in_proj, :]
            if self.in_proj.bias is not None:
                pruned_in_proj_bias = self.in_proj.bias[mask_in_proj, :]
            new_d_state = self.d_state - num_to_prune
            pruned_in_proj_weight[self.d_inner:self.d_inner + self.n_qk_heads * new_d_state, :] = B_weights
            pruned_in_proj_weight[
            self.d_inner + self.n_qk_heads * new_d_state:self.d_inner + 2 * self.n_qk_heads * new_d_state,
            :] = C_weights
        # Update in_proj weights and biases
        self.in_proj.weight = nn.Parameter(pruned_in_proj_weight)
        if self.in_proj.bias is not None:
            self.in_proj.bias = nn.Parameter(pruned_in_proj_bias)

        # Adjust self.d_state
        self.d_state -= num_to_prune

        # Adjust output features of in_proj
        self.in_proj.out_features = self.in_proj.weight.shape[0]

        # Build mask for conv1d channels
        conv_dim_old = self.conv1d.weight.shape[0]
        conv_mask = torch.ones(conv_dim_old, dtype=torch.bool, device=self.conv1d.weight.device)
        for idx in indices_to_prune_conv:
            conv_mask[idx] = False

        # Create new conv1d layer with adjusted dimensions
        conv_dim_new = conv_mask.sum().item()
        factory_kwargs = {"device": self.conv1d.weight.device, "dtype": self.conv1d.weight.dtype}
        new_conv = nn.Conv1d(
            in_channels=conv_dim_new,
            out_channels=conv_dim_new,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=conv_dim_new,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )

        pruned_conv_weights = self.conv1d.weight[conv_mask, :, :]
        if method == 'mean_pooling':
            # copy B and C weights to the new conv1d layer
            pruned_conv_weights[self.d_inner:self.d_inner + self.n_qk_heads * self.d_state, :] = conv_weight_B
            pruned_conv_weights[
            self.d_inner + self.n_qk_heads * self.d_state:self.d_inner + 2 * self.n_qk_heads * self.d_state,
            :] = conv_weight_C
        # Transfer weights and biases from the old conv1d layer
        new_conv.weight.data.copy_(pruned_conv_weights)
        if self.conv1d.bias is not None:
            pruned_conv_bias = self.conv1d.bias.data[conv_mask]
            if method == 'mean_pooling':
                pruned_conv_bias[self.d_inner:self.d_inner + self.n_qk_heads * self.d_state] = conv_bias_B
                pruned_conv_bias[
                self.d_inner + self.n_qk_heads * self.d_state:self.d_inner + 2 * self.n_qk_heads * self.d_state] = conv_bias_C
            new_conv.bias.data.copy_(pruned_conv_bias)

        self.conv1d = new_conv

        # Update the flag
        self.just_pruned = True
        # print(f"Pruned {num_to_prune} d_state dimensions per head")

    @torch.no_grad()
    def prune_by_mask_head_index(self):
        self.prune_heads(v_head_indices=self.layer_mask_head_index, qk_head_indices=self.layer_mask_head_index)



    def zero_out_head(self, x, z, B, C, head_index: Tensor, D=None, A_log=None):
        """
        Zero out the specified head in the parameters.

        Arguments:
            x: (batch_size, seq_len, n_heads, d_state) [or step shape: (batch_size, n_heads, d_state)]
            z: (batch_size, seq_len, n_heads*headdim)
            A_log: (batch_size, seq_len, n_heads)
            B: (batch_size, seq_len, n_heads, d_state)
            C: (batch_size, seq_len, n_heads, d_state)
            D: (n_heads,) or None
            head_index: Integer index of the head to zero out
        Returns:
            Modified A_log, B, C, D with the specified head zeroed out
        """
        # Zero out A_log for the head
        if A_log is not None:
            A_log[:, :, head_index] = 0

        # Zero out B and C for the head
        B[:, :, head_index, :] = 0
        C[:, :, head_index, :] = 0

        # Zero out D for the head if D is not None
        if D is not None:
            D[head_index] = 0

        x[:, :, head_index, :] = 0
        z = rearrange(z, "batch seq_len (num_heads channels_per_head) -> batch seq_len num_heads channels_per_head",
                      num_heads=self.n_v_heads)
        z[:, :, head_index, :] = 0
        z = rearrange(z, "batch seq_len num_heads channels_per_head -> batch seq_len (num_heads channels_per_head)")

        return x, z, B, C, D, A_log

    def get_in_proj_indices(self):
        """
        Returns a dictionary containing the start and end indices for each of the
        X, B, C, Z, and A_log parts in the in_proj weights.
        """
        idx = 0

        # Sizes of each component
        x_size = self.d_inner
        B_size = self.n_qk_heads * self.d_state
        C_size = self.n_qk_heads * self.d_state
        z_size = self.d_inner
        A_log_size = self.n_v_heads

        # Calculate indices for X
        x_start = idx
        x_end = x_start + x_size

        # Calculate indices for B
        B_start = x_end
        B_end = B_start + B_size

        # Calculate indices for C
        C_start = B_end
        C_end = C_start + C_size

        # Calculate indices for Z
        z_start = C_end
        z_end = z_start + z_size

        # Calculate indices for A_log
        A_log_start = z_end
        A_log_end = A_log_start + A_log_size

        x_conv_start = x_start
        x_conv_end = x_end
        B_conv_start = B_start
        B_conv_end = B_end
        C_conv_start = C_start
        C_conv_end = C_end

        indices = {
            'X': (x_start, x_end),
            'x': (x_start, x_end),
            'B': (B_start, B_end),
            'b': (B_start, B_end),
            'C': (C_start, C_end),
            'c': (C_start, C_end),
            'Z': (z_start, z_end),
            'z': (z_start, z_end),
            'A_log': (A_log_start, A_log_end),
            'a_log': (A_log_start, A_log_end),
            'x_conv': (x_conv_start, x_conv_end),
            'B_conv': (B_conv_start, B_conv_end),
            'C_conv': (C_conv_start, C_conv_end)
        }

        return indices

    def _random_cluster_kq_heads(self, new_qk_head: int):
        """
        randomly cluster the kq heads into gva_heads number of groups
        """
        assert self.n_v_heads % new_qk_head == 0, "n_v_heads must be divisible by gva_heads"
        random_indices = torch.randperm(self.n_qk_heads)[:(self.n_qk_heads - new_qk_head)].tolist()
        self._resize_in_proj_BC(new_qk_head, head_indices=random_indices)

    def _mean_pooling_cluster_kq_heads(self, new_qk_head: int):
        """
        average the kq heads into gva_heads number of groups
        """
        assert self.n_v_heads % new_qk_head == 0, "n_v_heads must be divisible by gva_heads"
        in_proj_indices = self.get_in_proj_indices()
        B_start, B_end = in_proj_indices['B']
        C_start, C_end = in_proj_indices['C']
        B_proj = self.in_proj.weight[B_start:B_end, :]
        C_proj = self.in_proj.weight[C_start:C_end, :]
        print(B_proj.shape)
        B_proj = rearrange(B_proj, "(n_qk_heads d_state) d_model -> n_qk_heads d_model d_state",
                           n_qk_heads=self.n_qk_heads)
        print(B_proj.shape)
        C_proj = rearrange(C_proj, "(n_qk_heads d_state) d_model -> n_qk_heads d_model d_state",
                           n_qk_heads=self.n_qk_heads)

        # mean_pool to new_qk_head heads by averaging every consecutive (self.n_v_heads // new_qk_head) heads
        group = self.n_v_heads // new_qk_head
        B_proj = rearrange(B_proj, "(new_qk_heads group) d_model d_state -> new_qk_heads group d_model d_state",
                           group=group)
        print(B_proj.shape)
        C_proj = rearrange(C_proj, "(new_qk_heads group) d_model d_state -> new_qk_heads group d_model d_state",
                           group=group)
        B_proj = reduce(B_proj, "new_qk_heads group d_model d_state -> new_qk_heads d_model d_state", "mean")
        print(B_proj.shape)
        C_proj = reduce(C_proj, "new_qk_heads group d_model d_state -> new_qk_heads d_model d_state", "mean")

        # revert to original form
        B_proj = rearrange(B_proj, "new_qk_heads d_model d_state -> (new_qk_heads d_state) d_model")
        print(B_proj.shape)
        C_proj = rearrange(C_proj, "new_qk_heads d_model d_state -> (new_qk_heads d_state) d_model")

        # update in_proj weight size by resizing B and C in in_proj then update them with the new ones
        self._resize_in_proj_BC(new_qk_head)
        in_proj_indices = self.get_in_proj_indices()
        B_start, B_end = in_proj_indices['B']
        C_start, C_end = in_proj_indices['C']

        self.in_proj.weight[B_start:B_end, :] = B_proj
        self.in_proj.weight[C_start:C_end, :] = C_proj

    def _kmeans_cluster_kq_heads(self, new_qk_head: int, num_iters=30, tol=1e-8):
        """
        Cluster the n_qk_heads into `new_qk_head` groups with a simple K-means.
        We apply the same cluster membership to both B and C.
        Then each group is replaced by the mean vector of its members.

        Arguments:
          new_qk_head: number of new QK heads after clustering
          num_iters: maximum number of K-means iterations
          tol: early stopping threshold for cluster-center changes
        """
        device = self.in_proj.weight.device
        assert self.n_v_heads % new_qk_head == 0, "n_v_heads must be divisible by new_qk_head"

        # --- 1) Gather the B and C weights for all existing heads ---
        in_proj_indices = self.get_in_proj_indices()
        B_start, B_end = in_proj_indices['B']
        C_start, C_end = in_proj_indices['C']

        B_proj = self.in_proj.weight[B_start:B_end, :].detach().clone()
        C_proj = self.in_proj.weight[C_start:C_end, :].detach().clone()

        # shape: (n_qk_heads, d_model, d_state)
        B_proj = rearrange(B_proj, "(n_qk_heads d_state) d_model -> n_qk_heads d_model d_state",
                           n_qk_heads=self.n_qk_heads).to(device)
        C_proj = rearrange(C_proj, "(n_qk_heads d_state) d_model -> n_qk_heads d_model d_state",
                           n_qk_heads=self.n_qk_heads).to(device)

        # We combine B and C for consistent clustering
        # shape each head as a single vector of size 2 * d_model * d_state
        n_qk_heads, d_model, d_state = B_proj.shape
        BC_cat = torch.cat([
            B_proj.reshape(n_qk_heads, -1),  # shape: (n_qk_heads, d_model*d_state)
            C_proj.reshape(n_qk_heads, -1)  # shape: (n_qk_heads, d_model*d_state)
        ], dim=1)  # shape: (n_qk_heads, 2*d_model*d_state)

        # --- 2) K-means cluster on BC_cat => new_qk_head clusters ---
        # simple K-means without external libs:
        k = new_qk_head
        # random init: pick k random heads as initial cluster centers
        init_indices = torch.randperm(n_qk_heads, device=device)[:k]
        centers = BC_cat[init_indices].clone()  # shape: (k, 2*d_model*d_state)

        # run iterative k-means
        for _ in range(num_iters):
            # dist => shape (n_qk_heads, k)
            print(BC_cat.shape, centers.shape)
            print(BC_cat.unsqueeze(1).shape, centers.unsqueeze(0).shape)

            dist = (BC_cat.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=2)
            assignments = dist.argmin(dim=1)  # shape: (n_qk_heads,)

            # update centers
            new_centers = []
            for cluster_id in range(k):
                members = BC_cat[assignments == cluster_id]
                if members.shape[0] == 0:
                    # corner-case: if no members, just keep old center
                    new_centers.append(centers[cluster_id])
                else:
                    new_centers.append(members.mean(dim=0))
            new_centers = torch.stack(new_centers, dim=0)

            move = (centers - new_centers).pow(2).sum().sqrt()
            centers = new_centers
            if move < tol:
                print(f"K-means converged after {_} iterations with move {move}")
                break

        # final assignments
        dist = (BC_cat.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(dim=2)
        assignments = dist.argmin(dim=1)  # (n_qk_heads,)

        # --- 3) For each cluster, compute the mean B, C separately, then store in new B/C arrays ---
        # shape: (k, d_model, d_state) for B, same for C
        new_B = torch.zeros(k, d_model, d_state, device=device, dtype=B_proj.dtype)
        new_C = torch.zeros(k, d_model, d_state, device=device, dtype=C_proj.dtype)

        for cluster_id in range(k):
            member_indices = (assignments == cluster_id).nonzero(as_tuple=True)[0]
            if len(member_indices) == 0:
                # If no members, default to zero or keep the old center
                continue
            # average B
            cluster_B = B_proj[member_indices]  # shape: (#members, d_model, d_state)
            cluster_B_mean = cluster_B.mean(dim=0)
            new_B[cluster_id] = cluster_B_mean

            # average C
            cluster_C = C_proj[member_indices]  # shape: (#members, d_model, d_state)
            cluster_C_mean = cluster_C.mean(dim=0)
            new_C[cluster_id] = cluster_C_mean

        # flatten back to shape: (k*d_state, d_model)
        new_B_2d = rearrange(new_B, "k d_model d_state -> (k d_state) d_model")
        new_C_2d = rearrange(new_C, "k d_model d_state -> (k d_state) d_model")

        # --- 4) Now we have new_B_2d, new_C_2d for k heads. Resize in_proj and put them back ---
        self._resize_in_proj_BC(new_qk_head)  # adjusts self.n_qk_heads to new_qk_head
        in_proj_indices = self.get_in_proj_indices()
        B_start, B_end = in_proj_indices['B']
        C_start, C_end = in_proj_indices['C']

        self.in_proj.weight[B_start:B_end, :] = new_B_2d
        self.in_proj.weight[C_start:C_end, :] = new_C_2d

    def _taylor_cluster_kq_heads(self, new_qk_head: int):
        """
        similar to. mean pooling with the differemce of pruning least important heads according to  BC_heads_taylor_importance_estimator
        """
        assert self.n_v_heads % new_qk_head == 0, "n_v_heads must be divisible by new_qk_head"
        # Get importance scores for each B and C head
        BC_heads_scores, BC_heads_indices = self.BC_heads_taylor_importance_estimator()

        # Get the least important heads
        BC_heads_indices = BC_heads_indices[:self.n_qk_heads - new_qk_head]

        # pruning the least important heads
        self._resize_in_proj_BC(new_qk_head, head_indices=BC_heads_indices.tolist())

    def _taylor_cluster_mean_pooling_kq_heads(self, new_qk_head: int):
        """
        1) Sort heads by ascending importance
        2) Partition into new_qk_head groups
        3) Mean-pool the B, C parameters across each group
        4) Resize in_proj/conv1d to new_qk_head heads
        5) Copy the averaged parameters back
        """
        assert self.n_v_heads % new_qk_head == 0, "n_v_heads must be divisible by new_qk_head"

        # --- 1) Obtain per-head importance and sorted indices (least -> most important) ---
        BC_heads_scores, BC_heads_indices = self.BC_heads_taylor_importance_estimator()
        # BC_heads_scores: [n_qk_heads]
        # BC_heads_indices: [n_qk_heads], sorted from least to most important

        old_n_qk_heads = self.n_qk_heads
        assert old_n_qk_heads >= new_qk_head, "Cannot cluster to more heads than currently exist"

        # --- 2) Gather B, C from in_proj for all heads in shape (n_qk_heads, d_model, d_state) ---
        in_proj_indices = self.get_in_proj_indices()
        B_start, B_end = in_proj_indices['B']
        C_start, C_end = in_proj_indices['C']

        # shape => [n_qk_heads * d_state, d_model]
        B_flat = self.in_proj.weight[B_start:B_end, :]
        C_flat = self.in_proj.weight[C_start:C_end, :]

        # reshape => [n_qk_heads, d_model, d_state]
        B_proj = rearrange(B_flat, "(heads ds) dm -> heads dm ds", heads=old_n_qk_heads)
        C_proj = rearrange(C_flat, "(heads ds) dm -> heads dm ds", heads=old_n_qk_heads)

        # --- 3) Reorder heads in ascending importance and chunk them into groups ---
        B_sorted = B_proj[BC_heads_indices, ...]  # shape => [n_qk_heads, d_model, d_state]
        C_sorted = C_proj[BC_heads_indices, ...]

        group_size = old_n_qk_heads // new_qk_head
        # e.g. if old_n_qk_heads=8, new_qk_head=4 => group_size=2

        # shape => [new_qk_head, group_size, d_model, d_state]
        B_grouped = B_sorted.view(new_qk_head, group_size, B_sorted.size(1), B_sorted.size(2))
        C_grouped = C_sorted.view(new_qk_head, group_size, C_sorted.size(1), C_sorted.size(2))

        # mean across group_size dimension => [new_qk_head, d_model, d_state]
        B_new = B_grouped.mean(dim=1)
        C_new = C_grouped.mean(dim=1)

        # flatten back to => [(new_qk_head*d_state), d_model]
        B_new_flat = rearrange(B_new, "heads dm ds -> (heads ds) dm")
        C_new_flat = rearrange(C_new, "heads dm ds -> (heads ds) dm")

        # --- 4) Resize model from old_n_qk_heads -> new_qk_head ---
        #     We'll let head_indices=None, so that the code re-initializes or
        #     discards the old B,C blocks and sets new channels
        self._resize_in_proj_BC(new_qk_head, head_indices=None)

        # --- 5) Insert averaged B, C back into the re-sized in_proj ---
        in_proj_indices = self.get_in_proj_indices()  # re-check after resize
        B_start, B_end = in_proj_indices['B']
        C_start, C_end = in_proj_indices['C']

        # Overwrite with the merged heads
        self.in_proj.weight[B_start:B_end, :] = B_new_flat
        self.in_proj.weight[C_start:C_end, :] = C_new_flat

        # Done. The model now has new_qk_head heads that are "merged" via mean pooling
        self.n_qk_heads = new_qk_head

    def _resize_in_proj_BC(self, new_n_qk_heads: int, head_indices=None):
        mask_dim = self.in_proj.out_features

        B_start = self.d_inner
        C_start = B_start + self.n_qk_heads * self.d_state
        if head_indices is not None:
            B_indices = []
            C_indices = []
            for idx in head_indices:
                # Indices for B
                B_head_start = B_start + idx * self.d_state
                B_head_end = B_head_start + self.d_state
                B_indices.extend(range(B_head_start, B_head_end))

                # Indices for C
                C_head_start = C_start + idx * self.d_state
                C_head_end = C_head_start + self.d_state
                C_indices.extend(range(C_head_start, C_head_end))
        else:
            B_indices = list(range(B_start, B_start + (self.n_qk_heads - new_n_qk_heads) * self.d_state))
            C_indices = list(range(C_start, C_start + (self.n_qk_heads - new_n_qk_heads) * self.d_state))

        mask = torch.ones(mask_dim, dtype=torch.bool, device=self.in_proj.weight.device)
        for idx in B_indices + C_indices:
            mask[idx] = False
        self.in_proj.weight = nn.Parameter(self.in_proj.weight[mask, :])
        if self.in_proj.bias is not None:
            self.in_proj.bias = nn.Parameter(self.in_proj.bias[mask])

        self.in_proj.out_features = self.in_proj.weight.shape[0]

        old_n_qk_heads = self.n_qk_heads
        self.n_qk_heads = new_n_qk_heads

        conv_dim_new = (self.d_inner + 2 * new_n_qk_heads * self.d_state)
        # Update conv1d
        conv_dim_old = self.conv1d.weight.shape[0]
        conv_mask = torch.ones(conv_dim_old, dtype=torch.bool, device=self.conv1d.weight.device)
        if head_indices is not None:
            # Indices to prune in conv1d (x indices)
            for idx in B_indices + C_indices:
                conv_mask[idx] = False
        else:

            group_size = old_n_qk_heads // new_n_qk_heads

            old_conv_dim = self.conv1d.in_channels

            # shape: [old_conv_dim, 1, kernel_size]
            old_weight = self.conv1d.weight
            kernel_size = old_weight.shape[2]

            # Separate out x-channels and BC-channels
            x_size = self.d_inner  # first part is for x
            bc_size = 2 * old_n_qk_heads * self.d_state  # second part is B + C
            assert x_size + bc_size == old_conv_dim, f"Mismatch in old_conv_dim {old_conv_dim} vs. x_size {x_size} + bc_size {bc_size}"

            # 2.1) Keep x-channels as-is (they aren't changing heads)
            x_weight = old_weight[:x_size, :, :]  # shape: (d_inner, 1, kernel_size)

            # 2.2) Group BC channels for averaging
            bc_weight = old_weight[x_size: x_size + bc_size, :, :]  # shape: (bc_size, 1, kernel_size)

            # Reshape bc_weight into (old_n_qk_heads, 2*d_state, 1, kernel_size)
            bc_weight = bc_weight.view(old_n_qk_heads, 2 * self.d_state, 1, kernel_size)
            # shape: (old_n_qk_heads, 2*d_state, 1, kernel_size)

            # Group heads: (new_qk_heads, group_size, 2*d_state, 1, kernel_size)
            bc_weight = rearrange(
                bc_weight,
                "(new_qk_heads group) c_in in_ch ks -> new_qk_heads group c_in in_ch ks",
                new_qk_heads=new_n_qk_heads,
                group=group_size
            )
            # Average along group dimension
            bc_weight = bc_weight.mean(dim=1)
            # Now shape: (new_qk_heads, 2*d_state, 1, kernel_size)

            # Flatten back: (new_qk_heads * 2*d_state, 1, kernel_size)
            bc_weight = bc_weight.view(-1, 1, kernel_size)

            # Concatenate x_weight + bc_weight => new full weight
            new_weight = torch.cat([x_weight, bc_weight], dim=0)  # shape: (new_conv_dim, 1, kernel_size)

        factory_kwargs = {"device": self.conv1d.weight.device, "dtype": self.conv1d.weight.dtype}
        new_conv = nn.Conv1d(
            in_channels=conv_dim_new,
            out_channels=conv_dim_new,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=conv_dim_new,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )

        if head_indices is not None:
            new_conv.weight.data.copy_(self.conv1d.weight.data[conv_mask, :, :])
            if self.conv1d.bias is not None:
                new_conv.bias.data.copy_(self.conv1d.bias.data[conv_mask])
        else:
            new_conv.weight.data.copy_(new_weight)
            if self.conv1d.bias is not None:
                # Similarly, we do not want to just "keep" or "prune" the bias
                # We want to do the same grouping approach. So let's separate x-bias & bc-bias
                old_bias = self.conv1d.bias  # shape: (old_conv_dim,)
                x_bias = old_bias[:x_size]  # shape: (d_inner,)
                bc_bias = old_bias[x_size: x_size + bc_size]  # (bc_size,)

                # Reshape: (old_n_qk_heads, 2*d_state)
                bc_bias = bc_bias.view(old_n_qk_heads, 2 * self.d_state)
                bc_bias = rearrange(
                    bc_bias,
                    "(new_qk_heads group) c_in -> new_qk_heads group c_in",
                    new_qk_heads=new_n_qk_heads,
                    group=group_size
                )
                # Mean over group => (new_qk_heads, 2*d_state)
                bc_bias = bc_bias.mean(dim=1)
                # Flatten => (new_qk_heads * 2*d_state,)
                bc_bias = bc_bias.view(-1)

                # Concat x-bias + bc-bias
                new_bias = torch.cat([x_bias, bc_bias], dim=0)
                new_conv.bias.copy_(new_bias)

        self.conv1d = new_conv

    @torch.no_grad()
    def prune_kq_heads(self, new_qk_head: int, clustering_method='mean_pooling'):
        """
        attention K Q  corresponds to B C matrices here.
        Group kq heads into number of groups that n_v_heads divides.
        1) extract the kq heads from B and C
        then group them into gva_heads number of groups using mean_pooling_cluster_kq_heads
        2) update the self.in_proj weights and dimensions
        3) update the self.conv1d weights and dimensions
        """
        if clustering_method == 'mean_pooling':
            self._mean_pooling_cluster_kq_heads(new_qk_head)
        elif clustering_method == 'random':
            self._random_cluster_kq_heads(new_qk_head)
        elif clustering_method == 'kmeans':
            self._kmeans_cluster_kq_heads(new_qk_head)
        elif clustering_method == 'taylor':
            self._taylor_cluster_kq_heads(new_qk_head)
        elif clustering_method == 'taylor_mean_pooling':
            self._taylor_cluster_mean_pooling_kq_heads(new_qk_head)
        else:
            raise NotImplementedError


    @torch.no_grad()
    def prune_v_head_internal(self, ratio: float, method: str = 'taylor_second'):
        """
                Prune the lowest magnitude d_state dimensions per head by removing their parameters according to the given ratio.
                The same number of elements are removed from each head, and d_state is reduced accordingly.
                """
        importance_estimator = self.headdim_taylor_second_indp_XZ_importance_estimator

        num_to_prune = int(self.headdim * ratio)

        if num_to_prune == 0:
            return  # Nothing to prune

        indices_to_prune = []
        indices_to_prune_conv = []
        in_proj_indices = self.get_in_proj_indices()
        X_proj_start, X_proj_end = in_proj_indices['X']
        Z_proj_start, Z_proj_end = in_proj_indices['Z']
        for h in range(self.n_v_heads):
            # Indices in in_proj corresponding to B and C for head h
            idx_X_start = X_proj_start + h * self.headdim
            idx_Z_start = Z_proj_start + h * self.headdim

            indices_X, indices_Z = importance_estimator(idx_X_start, idx_Z_start)
            indices_prune_head_X = indices_X[:num_to_prune]
            indices_prune_head_Z = indices_Z[:num_to_prune]

            # Map indices to global indices in in_proj
            # For X
            indices_to_prune.extend([idx_X_start + i for i in indices_prune_head_X])
            # For Z
            indices_to_prune.extend([idx_Z_start + i for i in indices_prune_head_Z])

            # Now, for conv1d channels
            idx_X_conv = h * self.headdim

            indices_to_prune_conv.extend([idx_X_conv + i for i in indices_prune_head_X])

        # Build mask for in_proj.weight and in_proj.bias
        mask_in_proj = torch.ones(self.in_proj.out_features, dtype=torch.bool, device=self.in_proj.weight.device)
        for idx in indices_to_prune:
            mask_in_proj[idx] = False

        pruned_in_proj_weight = self.in_proj.weight[mask_in_proj, :]
        if self.in_proj.bias is not None:
            pruned_in_proj_bias = self.in_proj.bias[mask_in_proj]
        # Update in_proj weights and biases
        self.in_proj.weight = nn.Parameter(pruned_in_proj_weight)
        if self.in_proj.bias is not None:
            self.in_proj.bias = nn.Parameter(pruned_in_proj_bias)

        self.z_bias = nn.Parameter(self.z_bias[mask_in_proj[in_proj_indices['Z'][0]:in_proj_indices['Z'][1]]])
        # Adjust output features of in_proj
        self.in_proj.out_features = self.in_proj.weight.shape[0]

        # Build mask for conv1d channels
        conv_dim_old = self.conv1d.weight.shape[0]
        conv_mask = torch.ones(conv_dim_old, dtype=torch.bool, device=self.conv1d.weight.device)
        for idx in indices_to_prune_conv:
            conv_mask[idx] = False

        # Create new conv1d layer with adjusted dimensions
        conv_dim_new = conv_mask.sum().item()
        factory_kwargs = {"device": self.conv1d.weight.device, "dtype": self.conv1d.weight.dtype}
        new_conv = nn.Conv1d(
            in_channels=conv_dim_new,
            out_channels=conv_dim_new,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=conv_dim_new,
            padding=self.d_conv - 1,
            **factory_kwargs,
        )

        pruned_conv_weights = self.conv1d.weight[conv_mask, :, :]

        # Transfer weights and biases from the old conv1d layer
        new_conv.weight.data.copy_(pruned_conv_weights)
        if self.conv1d.bias is not None:
            pruned_conv_bias = self.conv1d.bias.data[conv_mask]
            new_conv.bias.data.copy_(pruned_conv_bias)

        self.conv1d = new_conv


        # prune outproj
        outproj_indices = self.outproj_importance_estimator()

        outproj_indices = outproj_indices[:(num_to_prune*self.n_v_heads)]

        mask_out_proj = torch.ones(self.out_proj.in_features, dtype=torch.bool, device=self.out_proj.weight.device)
        for idx in outproj_indices:
            mask_out_proj[idx] = False

        pruned_out_proj_weight = self.out_proj.weight[:, mask_out_proj]
        if self.out_proj.bias is not None:
            pruned_out_proj_bias = self.out_proj.bias[mask_out_proj]

        self.out_proj.weight = nn.Parameter(pruned_out_proj_weight)
        if self.out_proj.bias is not None:
            self.out_proj.bias = nn.Parameter(pruned_out_proj_bias)

        self.out_proj.in_features = self.out_proj.weight.shape[1]



        # Adjust
        self.headdim -= num_to_prune
        self.d_inner = self.headdim * self.n_v_heads
        assert  self.d_inner % self.n_v_heads == 0, f"d_inner must be divisible by n_v_heads, got {self.d_inner} % {self.n_v_heads}"

        # Update the flag
        self.just_pruned = True
        # print(f"Pruned {num_to_prune} d_state dimensions per head")
