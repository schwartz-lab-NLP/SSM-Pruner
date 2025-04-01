# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

from original_mamba.mamba_ssm.models.config_mamba import MambaConfig

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from original_mamba.mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from original_mamba.mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin


class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
        d_inner=None,
        nheads=None,

        in_proj_bias=None,
        in_proj_out_features=None,
        out_proj_bias=None,
        out_proj_in_features=None,
        conv1d_dim=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = (self.expand * self.d_model) // self.world_size if d_inner is None else d_inner
        try:
            assert self.d_inner * self.world_size == self.expand * self.d_model, f"d_inner must be divisible by world_size, got {self.d_inner} and {self.expand * self.d_model}"
        except AssertionError as e:
            print(e)

        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim if nheads is None else nheads
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.conv_bias = conv_bias

        ###################################### added by tGhattas  ##############################################
        # in case in_proj_is_split is True we'll have self.z, self.x, self.B, self.C, self.dt
        self.in_proj_is_split = False
        self.z = self.x = self.B = self.C = self.dt = None

        self.repeat_y = False
        ###################################################################################################

        # Order: [z, x, B, C, dt]
        if in_proj_out_features is not None:
            d_in_proj = in_proj_out_features
        else:
            d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if not self.in_proj_is_split :
            if self.process_group is None:
                self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias or in_proj_bias, **factory_kwargs)
            else:
                self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                    process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                    **factory_kwargs)
        else:
            self.in_proj = None

        if conv1d_dim is not None:
            conv_dim = conv1d_dim
        else:
            conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        out_proj_in_features = out_proj_in_features if out_proj_in_features is not None else self.d_inner
        if self.process_group is None:
            self.out_proj = nn.Linear(out_proj_in_features, self.d_model, bias=bias or out_proj_bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(out_proj_in_features * self.world_size, self.d_model, bias=bias or out_proj_bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None, **kwargs):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        if not self.in_proj_is_split:
            zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        else:
            zu = self.z(u)
            xu = self.x(u)
            Bu = self.B(u)
            Cu = self.C(u)
            dtu = self.dt(u)
            zxbcdt = torch.cat([zu, xu, Bu, Cu, dtu], dim=-1)

        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        if self.use_mem_eff_path and inference_params is None:

            out = mamba_split_conv1d_scan_combined(
                zxbcdt,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.dt_bias,
                A,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                chunk_size=self.chunk_size,
                seq_idx=seq_idx,
                activation=self.activation,
                rmsnorm_weight=self.norm.weight if self.rmsnorm else None,
                rmsnorm_eps=self.norm.eps if self.rmsnorm else 1e-6,
                outproj_weight=self.out_proj.weight,
                outproj_bias=self.out_proj.bias,
                headdim=None if self.D_has_hdim else self.headdim,
                ngroups=self.ngroups,
                norm_before_gate=self.norm_before_gate,
                **dt_limit_kwargs,
            )
            if seqlen_og is not None:
                out = rearrange(out, "b l d -> (b l) d")
            if self.process_group is not None:
                reduce_fn = reduce_scatter if self.sequence_parallel else all_reduce
                out = reduce_fn(out, self.process_group)
        else:
            d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
            z0, x0, z, xBC, dt = torch.split(
                zxbcdt,
                [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
                dim=-1
            )
            if conv_state is not None:
                if cu_seqlens is None:
                    # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                    # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                    xBC_t = rearrange(xBC, "b l d -> b d l")
                    conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
                else:
                    assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                    assert batch == 1, "varlen inference only supports batch dimension 1"
                    conv_varlen_states = causal_conv1d_varlen_states(
                        xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                    )
                    conv_state.copy_(conv_varlen_states)
            assert self.activation in ["silu", "swish"]
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
                xBC = self.act(
                    self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)]
                )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
            else:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2),
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            ) # (batch, seqlen, nheads, headdim)
            if ssm_state is not None:
                y, last_state, *rest = y
                if cu_seqlens is None:
                    ssm_state.copy_(last_state)
                else:
                    varlen_states = rest[0]
                    ssm_state.copy_(varlen_states)

            y = rearrange(y, "b l h p -> b l (h p)")
            if self.repeat_y:
                y = torch.repeat_interleave(y, self.out_proj.in_features // y.shape[-1], dim=-1)
                z = torch.repeat_interleave(z, self.out_proj.in_features // z.shape[-1], dim=-1)
            if self.rmsnorm:
                y = self.norm(y, z)
            if d_mlp > 0:
                y = torch.cat([F.silu(z0) * x0, y], dim=-1)
            if seqlen_og is not None:
                y = rearrange(y, "b l d -> (b l) d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        if not self.in_proj_is_split:
            zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        else:
            zu = self.z(hidden_states.squeeze(1))
            xu = self.x(hidden_states.squeeze(1))
            Bu = self.B(hidden_states.squeeze(1))
            Cu = self.C(hidden_states.squeeze(1))
            dtu = self.dt(hidden_states.squeeze(1))
            zxbcdt = torch.cat([zu, xu, Bu, Cu, dtu], dim=-1)

        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1
        )

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        A = -torch.exp(self.A_log.float())  # (nheads,)

        # SSM step
        if selective_state_update is None:
            assert self.ngroups == 1, "Only support ngroups=1 for this inference code path"
            # Discretize A and B
            dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
            dA = torch.exp(dt * A)  # (batch, nheads)
            x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
            ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
            y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
            y = y + rearrange(self.D.to(dtype), "h -> h 1") * x
            y = rearrange(y, "b h p -> b (h p)")
            if not self.rmsnorm:
                y = y * self.act(z)  # (B D)
        else:
            A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.headdim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
            D = repeat(self.D, "h -> h p", p=self.headdim)
            B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
            C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
            x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
            if not self.rmsnorm:
                z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
            y = selective_state_update(
                ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
                dt_bias=dt_bias, dt_softplus=True
            )
            y = rearrange(y, "b h p -> b (h p)")
        if self.rmsnorm:
            print("-"*100, f"y shape {y.shape}, z shape {z.shape}")
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


####################################################################################################################
# Pruning Methods added by Tamer Ghattas
####################################################################################################################
    @torch.no_grad()
    def split_in_proj(self):
        self.z = nn.Linear(self.d_model, self.d_inner, bias=self.in_proj.bias is not None, device=self.in_proj.weight.device)
        self.x = nn.Linear(self.d_model, self.d_inner, bias=self.in_proj.bias is not None, device=self.in_proj.weight.device)
        self.B = nn.Linear(self.d_model, self.ngroups * self.d_state, bias=self.in_proj.bias is not None, device=self.in_proj.weight.device)
        self.C = nn.Linear(self.d_model, self.ngroups * self.d_state, bias=self.in_proj.bias is not None, device=self.in_proj.weight.device)
        self.dt = nn.Linear(self.d_model, self.nheads, bias=self.in_proj.bias is not None, device=self.in_proj.weight.device)

        split = torch.split(self.in_proj.weight, [self.d_inner, self.d_inner, self.ngroups*self.d_state, self.ngroups*self.d_state, self.nheads], dim=0)
        if self.in_proj.bias is not None:
            bias_split = torch.split(self.in_proj.bias, [self.d_inner, self.d_inner, self.ngroups*self.d_state, self.ngroups*self.d_state, self.nheads], dim=0) if self.in_proj.bias is not None else None

        self.z.weight.data = nn.Parameter(split[0])
        self.x.weight.data = nn.Parameter(split[1])
        self.B.weight.data = nn.Parameter(split[2])
        self.C.weight.data = nn.Parameter(split[3])
        self.dt.weight.data = nn.Parameter(split[4])
        if self.in_proj.bias is not None:
            self.z.bias.data = nn.Parameter(bias_split[0])
            self.x.bias.data = nn.Parameter(bias_split[1])
            self.B.bias.data = nn.Parameter(bias_split[2])
            self.C.bias.data = nn.Parameter(bias_split[3])
            self.dt.bias.data = nn.Parameter(bias_split[4])
        self.in_proj = None
        self.in_proj_is_split = True


    def dstate_taylor_importance_aux(self, order, *args):
        idx_B_start = args[0]
        idx_C_start = args[1]

        B_weights = self.in_proj.weight[idx_B_start:idx_B_start + self.d_state, :]
        B_weights_acc_grad = self.in_proj.weight.acc_grad[idx_B_start:idx_B_start + self.d_state, :]

        C_weights = self.in_proj.weight[idx_C_start:idx_C_start + self.d_state, :]
        C_weights_acc_grad = self.in_proj.weight.acc_grad[idx_C_start:idx_C_start + self.d_state, :]

        if hasattr(self.in_proj.weight, 'acc_hess'):
            B_weights_acc_hess = self.in_proj.weight.acc_hess[idx_B_start:idx_B_start + self.d_state, :]
            C_weights_acc_hess = self.in_proj.weight.acc_hess[idx_C_start:idx_C_start + self.d_state, :]
        B_salience = (B_weights * B_weights_acc_grad)
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

    def _mean_pool_dstates(self, num_to_prune):
        """
        mean pool the consecutive dstates from BC heads by grouping them in self.dstates // num_to_prune groups
        """
        assert self.d_state % num_to_prune == 0, f"d_state must be divisible by num_to_prune - {self.d_state} % {num_to_prune} != 0"
        group_size = self.d_state // num_to_prune
        C_start, C_end = self.get_in_proj_indices()['C']
        C_weight = self.in_proj.weight[C_start:C_end, :]
        C_weight = rearrange(C_weight, "(ngroups d_state) d_model -> ngroups d_model d_state",
                            ngroups=self.ngroups)
        C_weight = rearrange(C_weight, "ngroups d_model (new_d_state group_size) -> ngroups d_model new_d_state group_size",
                            group_size=group_size)
        C_weight = C_weight.mean(dim=-1) # (ngroups, d_model, d_state)
        C_weight = rearrange(C_weight, "ngroups d_model new_d_state -> (ngroups new_d_state) d_model")

        C_bias = None
        if self.in_proj.bias is not None:
            C_bias = self.in_proj.bias[self.d_inner + self.ngroups * self.d_state:self.d_inner + 2 * self.ngroups * self.d_state]
            C_bias = rearrange(C_bias, "(ngroups d_state) -> ngroups d_state", group_size=group_size)
            C_bias = rearrange(C_bias, "ngroups (new_d_state group_size) -> ngroups new_d_state group_size",
                            group_size=group_size)
            C_bias = C_bias.mean(dim=-1)
            C_bias = rearrange(C_bias, "ngroups new_d_state -> (ngroups new_d_state)")

        conv_weight_C = self.conv1d.weight[2*self.d_xb:2*self.d_xb + self.ngroups * self.d_state,:]
        conv_weight_C = rearrange(conv_weight_C, "(ngroups d_state) 1 kernel_size -> ngroups 1 kernel_size d_state",
                                    ngroups=self.ngroups)
        conv_weight_C = rearrange(conv_weight_C, "ngroups 1 kernel_size (new_d_state group_size) -> ngroups 1 kernel_size new_d_state group_size",
                                    group_size=group_size)
        conv_weight_C = conv_weight_C.mean(dim=-1)
        conv_weight_C = rearrange(conv_weight_C, "ngroups 1 kernel_size new_d_state -> (ngroups new_d_state) 1 kernel_size")
        conv_bias_C = None
        if self.conv1d.bias is not None:
            conv_bias_C = self.conv1d.bias[2*self.d_xb:2*self.d_xb + self.ngroups * self.d_state,:]
            conv_bias_C = rearrange(conv_bias_C, "(ngroups d_state) -> ngroups d_state", ngroups=self.ngroups,
                                    group_size=group_size)
            conv_bias_C = rearrange(conv_bias_C, "ngroups (new_d_state group_size) -> ngroups new_d_state group_size",
                                    group_size=group_size)
            conv_bias_C = conv_bias_C.mean(dim=-1)
            conv_bias_C = rearrange(conv_bias_C, "ngroups new_d_state -> (ngroups new_d_state)")

        return C_weight, C_bias, conv_weight_C, conv_bias_C

    def dstates_taylor_first_indp_BC_importance_estimator(self, *args):
        B_taylor_imp, C_taylor_imp = self.dstate_taylor_importance_aux(1, *args)
        taylor_imp = B_taylor_imp + C_taylor_imp
        _, indices_B = _, indices_C = torch.sort(taylor_imp)  # ascending order
        return indices_B, indices_C

    def dstates_taylor_second_indp_BC_importance_estimator(self, *args):
        B_taylor_imp, C_taylor_imp = self.dstate_taylor_importance_aux(2, *args)
        taylor_imp = B_taylor_imp + C_taylor_imp
        _, indices_B = _, indices_C = torch.sort(taylor_imp)  # ascending order
        return indices_B, indices_C

    def dstates_taylor_first_by_B_importance_estimator(self, *args):
        B_taylor_imp, _ = self.dstate_taylor_importance_aux(1, *args)
        _, indices_B = _, indices_C = torch.sort(B_taylor_imp)
        return indices_B, indices_C

    def dstates_taylor_second_by_B_importance_estimator(self, *args):
        B_taylor_imp, _ = self.dstate_taylor_importance_aux(2, *args)
        _, indices_B = _, indices_C = torch.sort(B_taylor_imp)
        return indices_B, indices_C

    def dstates_taylor_first_by_C_importance_estimator(self, *args):
        _, C_taylor_imp = self.dstate_taylor_importance_aux(1, *args)
        _, indices_B = _, indices_C = torch.sort(C_taylor_imp)  # ascending order
        return indices_B, indices_C

    def dstates_taylor_second_by_C_importance_estimator(self, *args):
        _, C_taylor_imp = self.dstate_taylor_importance_aux(2, *args)
        _, indices_B = _, indices_C = torch.sort(C_taylor_imp)  # ascending order
        return indices_B, indices_C

    def dstates_magnitude_importance_estimator(self, *args):
        """
        Compute the importance of each d_state dimension based on the magnitudes of the B and C weights.
        """
        zxBCdt_indices = self.get_in_proj_indices()
        B_start, B_end = zxBCdt_indices['B']
        C_start, C_end = zxBCdt_indices['C']
        B_weights = self.in_proj.weight[B_start:B_end, :]
        C_weights = self.in_proj.weight[C_start:C_end, :]
        ## use L2 norm
        magnitudes_B = B_weights.abs().pow(2).sum(dim=1)  # (d_state,)
        magnitudes_C = C_weights.abs().pow(2).sum(dim=1)  # (d_state,)

        # Get indices to prune
        _, indices_B = torch.sort(magnitudes_B)  # ascending order
        _, indices_C = torch.sort(magnitudes_C)
        return indices_B, indices_C

    def dstates_random_importance_estimator(self, *args):
        """
        Assign random importance of each d_state dimension
        """
        indices = torch.randperm(self.d_state)
        return indices, indices

    @torch.no_grad()
    def prune_dstates(self, ratio, method="taylor_second", return_plot=False):
        """
        Prune the lowest magnitude d_state dimensions per head by removing their parameters according to the given ratio.
        The same number of elements are removed from each head, and d_state is reduced accordingly.
        """
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
            importance_estimator = self.dstates_magnitude_importance_estimator
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
        zxBCdt_indices = self.get_in_proj_indices()
        for h in range(self.ngroups):
            # Indices in in_proj corresponding to B and C for head h
            idx_B_start = zxBCdt_indices['B'][0] + h * self.d_state
            idx_C_start = zxBCdt_indices['C'][0] + h * self.d_state


            indices_B, indices_C = importance_estimator(idx_B_start, idx_C_start)
            indices_prune_head_B = indices_B[:num_to_prune].tolist()
            indices_prune_head_C = indices_C[:num_to_prune].tolist()

            # Map indices to global indices in in_proj
            # For B
            indices_to_prune.extend([idx_B_start + i for i in indices_prune_head_B])
            # For C
            indices_to_prune.extend([idx_C_start + i for i in indices_prune_head_C])

            # Now, for conv1d channels
            idx_B_conv = self.d_ssm + h * self.d_state
            idx_C_conv = self.d_ssm + self.ngroups * self.d_state + h * self.d_state

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
            pruned_in_proj_weight[self.d_inner:self.d_inner + self.ngroups * new_d_state, :] = B_weights
            pruned_in_proj_weight[self.d_inner + self.ngroups * new_d_state:self.d_inner + 2 * self.ngroups * new_d_state, :] = C_weights
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
            pruned_conv_weights[self.d_ssm:self.d_ssm + self.ngroups * self.d_state, :] = conv_weight_B
            pruned_conv_weights[self.d_ssm + self.ngroups * self.d_state:self.d_ssm + 2 * self.ngroups * self.d_state, :] = conv_weight_C
        # Transfer weights and biases from the old conv1d layer
        new_conv.weight.data.copy_(pruned_conv_weights)
        if self.conv1d.bias is not None:
            pruned_conv_bias = self.conv1d.bias.data[conv_mask]
            if method == 'mean_pooling':
                pruned_conv_bias[self.d_ssm:self.d_ssm + self.ngroups * self.d_state] = conv_bias_B
                pruned_conv_bias[self.d_ssm + self.ngroups * self.d_state:self.d_ssm + 2 * self.ngroups * self.d_state] = conv_bias_C
            new_conv.bias.data.copy_(pruned_conv_bias)

        self.conv1d = new_conv

        # Update the flag
        self.just_pruned = True

####################################################################################################################
# dinner prune methods
####################################################################################################################
    # random estimators
    def headdim_random_importance_estimator(self, *args):
        """
        Assign random importance of each headdim dimension
        """
        indices = torch.randperm(self.headdim)
        return indices

    def headdim_wanda_importance_estimator(self, *args, mask=None):
        """
        Assign random importance of each headdim dimension
        """
        idx_z_start = args[0]
        idx_x_start = args[1]

        z_weights = self.in_proj.weight[idx_z_start:idx_z_start + self.headdim, :]
        x_weights = self.in_proj.weight[idx_x_start:idx_x_start + self.headdim, :]

        z_salience = (z_weights * mask[idx_z_start:idx_z_start + self.headdim, :])
        x_salience = (x_weights * mask[idx_x_start:idx_x_start + self.headdim, :])

        z_taylor_imp = z_salience.abs().mean(dim=1)  # (headdim,)
        x_taylor_imp = x_salience.abs().mean(dim=1)  # (headdim,)

        _, indices_z = torch.sort(z_taylor_imp)
        _, indices_x = torch.sort(x_taylor_imp)
        return indices_z, indices_x


    def out_proj_random_importance_estimator(self, *args):
        """
        Assign random importance of each d_model dimension
        """
        indices = torch.randperm(self.d_inner)
        return indices
    # Taylor estimators
    def zx_headdim_taylor_importance_aux(self, order, *args):
        idx_z_start = args[0]
        idx_x_start = args[1]

        z_weights = self.in_proj.weight[idx_z_start:idx_z_start + self.headdim, :]
        z_weights_acc_grad = self.in_proj.weight.acc_grad[idx_z_start:idx_z_start + self.headdim, :]

        x_weights = self.in_proj.weight[idx_x_start:idx_x_start + self.headdim, :]
        x_weights_acc_grad = self.in_proj.weight.acc_grad[idx_x_start:idx_x_start + self.headdim, :]

        if hasattr(self.in_proj.weight, 'acc_hess'):
            z_weights_acc_hess = self.in_proj.weight.acc_hess[idx_z_start:idx_z_start + self.headdim, :]
            x_weights_acc_hess = self.in_proj.weight.acc_hess[idx_x_start:idx_x_start + self.headdim, :]
        z_salience = (z_weights * z_weights_acc_grad)
        x_salience = (x_weights * x_weights_acc_grad)

        if order == 2:
            if not hasattr(self.in_proj.weight, 'acc_hess'):
                z_salience = z_salience * z_weights
                x_salience = x_salience * x_weights
            else:
                z_salience = z_salience + 0.5 * z_weights_acc_hess * (z_weights ** 2)
                x_salience = x_salience + 0.5 * x_weights_acc_hess * (x_weights ** 2)

        z_taylor_imp = z_salience.abs().mean(dim=1)  # (headdim,)
        x_taylor_imp = x_salience.abs().mean(dim=1)  # (headdim,)
        return z_taylor_imp, x_taylor_imp

    def headdim_taylor_first_importance_estimator(self, *args):
        z_taylor_imp, x_taylor_imp = self.zx_headdim_taylor_importance_aux(1, *args)
        _, indices_z = torch.sort(z_taylor_imp)
        _, indices_x = torch.sort(x_taylor_imp)
        return indices_z, indices_x

    def headdim_taylor_second_importance_estimator(self, *args):
        z_taylor_imp, x_taylor_imp = self.zx_headdim_taylor_importance_aux(2, *args)
        _, indices_z = torch.sort(z_taylor_imp)
        _, indices_x = torch.sort(x_taylor_imp)
        return indices_z, indices_x

    def headdim_taylor_second_importance_estimator_agg(self, *args):
        z_taylor_imp, x_taylor_imp = self.zx_headdim_taylor_importance_aux(2, *args)
        aggregated_taylor_imp = z_taylor_imp + x_taylor_imp
        _, indices_zx = torch.sort(aggregated_taylor_imp)
        return indices_zx, indices_zx

    def headdim_taylor_first_by_z_importance_estimator(self, *args):
        z_taylor_imp, _ = self.zx_headdim_taylor_importance_aux(1, *args)
        _, indices_z = torch.sort(z_taylor_imp)
        return indices_z, indices_z

    def headdim_taylor_second_by_z_importance_estimator(self, *args):
        z_taylor_imp, _ = self.zx_headdim_taylor_importance_aux(2, *args)
        _, indices_z = torch.sort(z_taylor_imp)
        return indices_z, indices_z

    def headdim_taylor_first_by_x_importance_estimator(self, *args):
        _, x_taylor_imp = self.zx_headdim_taylor_importance_aux(1, *args)
        _, indices_x = torch.sort(x_taylor_imp)
        return indices_x, indices_x

    def headdim_taylor_second_by_x_importance_estimator(self, *args):
        _, x_taylor_imp = self.zx_headdim_taylor_importance_aux(2, *args)
        _, indices_x = torch.sort(x_taylor_imp)
        return indices_x, indices_x

    def outproj_taylor_importance_aux(self, order, *args):
        out_proj_weights = self.out_proj.weight
        out_proj_weights_acc_grad = self.out_proj.weight.acc_grad

        if hasattr(self.out_proj.weight, 'acc_hess'):
            out_proj_weights_acc_hess = self.out_proj.weight.acc_hess

        out_proj_salience = out_proj_weights * out_proj_weights_acc_grad

        if order == 2:
            if not hasattr(self.out_proj.weight, 'acc_hess'):
                out_proj_salience = out_proj_salience * out_proj_weights
            else:
                out_proj_salience = out_proj_salience + 0.5 * out_proj_weights_acc_hess * (out_proj_weights ** 2)

        out_proj_taylor_imp = out_proj_salience.abs().mean(dim=1) # (d_inner,)
        return out_proj_taylor_imp

    def outproj_taylor_first_importance_estimator(self, *args):
        out_proj_taylor_imp = self.outproj_taylor_importance_aux(1, *args)
        _, indices = torch.sort(out_proj_taylor_imp)
        return indices

    def outproj_taylor_second_importance_estimator(self, *args):
        out_proj_taylor_imp = self.outproj_taylor_importance_aux(2, *args)
        _, indices = torch.sort(out_proj_taylor_imp)
        return indices

    def D_taylor_importance_aux(self, order, *args):
        D_weights = self.D
        d_head_start = args[0]

        D_weights = D_weights[d_head_start:d_head_start + self.headdim]
        D_weights_acc_grad = self.D.acc_grad[d_head_start:d_head_start + self.headdim]
        if hasattr(self.D, 'acc_hess'):
            D_weights_acc_hess = self.D.acc_hess[d_head_start:d_head_start + self.headdim]
        D_salience = D_weights * D_weights_acc_grad

        if order == 2:
            if not hasattr(self.D, 'acc_hess'):
                D_salience = D_salience * D_weights
            else:
                D_salience = D_salience + 0.5 * D_weights_acc_hess * (D_weights ** 2)

        D_taylor_imp = D_salience.abs()
        return D_taylor_imp


    def D_taylor_first_importance_estimator(self, *args):
        D_taylor_imp = self.D_taylor_importance_aux(1, *args)
        _, indices = torch.sort(D_taylor_imp)
        return indices

    def D_taylor_second_importance_estimator(self, *args):
        D_taylor_imp = self.D_taylor_importance_aux(2, *args)
        _, indices = torch.sort(D_taylor_imp)
        return indices

    def norm_importance_estimator_aux(self, order, *args):
        norm_weights = self.norm.weight
        norm_head_start = args[0]

        norm_weights = norm_weights[norm_head_start:norm_head_start + self.headdim]

        norm_weights_acc_grad = self.norm.weight.acc_grad[norm_head_start:norm_head_start + self.headdim]

        if hasattr(self.norm.weight, 'acc_hess'):
            norm_weights_acc_hess = self.norm.weight.acc_hess[norm_head_start:norm_head_start + self.headdim]
        norm_salience = norm_weights * norm_weights_acc_grad
        if order == 2:
            if not hasattr(self.norm.weight, 'acc_hess'):
                norm_salience = norm_salience * norm_weights
            else:
                norm_salience = norm_salience + 0.5 * norm_weights_acc_hess * (norm_weights ** 2)
        norm_taylor_imp = norm_salience.abs()
        return norm_taylor_imp

    def norm_taylor_first_importance_estimator(self, *args):
        norm_taylor_imp = self.norm_importance_estimator_aux(1, *args)
        _, indices = torch.sort(norm_taylor_imp)
        return indices

    def norm_taylor_second_importance_estimator(self, *args):
        norm_taylor_imp = self.norm_importance_estimator_aux(2, *args)
        _, indices = torch.sort(norm_taylor_imp)
        return indices

    def _mean_pool_headdim(self, num_to_prune):
        """
        mean pool the consecutive headdim from zx` heads by grouping them in self.headdim // num_to_prune groups
        """
        assert self.headdim % num_to_prune == 0, f"headdim must be divisible by num_to_prune - {self.headdim} % {num_to_prune} != 0"
        group_size = self.headdim // num_to_prune
        z_start, z_end = self.get_in_proj_indices()['z']
        z_weight = self.in_proj.weight[z_start:z_end, :]
        z_weight = rearrange(z_weight, "(nheads headdim) d_model -> nheads d_model headdim",
                            nheads=self.nheads)
        z_weight = rearrange(z_weight, "nheads d_model (new_headdim group_size) -> nheads d_model new_headdim group_size",
                            group_size=group_size)
        z_weight = z_weight.mean(dim=-1)

        x_start, x_end = self.get_in_proj_indices()['x']
        x_weight = self.in_proj.weight[x_start:x_end, :]
        x_weight = rearrange(x_weight, "(nheads headdim) d_model -> nheads d_model headdim",
                            nheads=self.nheads)
        x_weight = rearrange(x_weight, "nheads d_model (new_headdim group_size) -> nheads d_model new_headdim group_size",
                            group_size=group_size)
        x_weight = x_weight.mean(dim=-1)

        z_bias = x_bias = None
        if self.in_proj.bias is not None:
            z_bias = self.in_proj.bias[:self.nheads * self.headdim]
            z_bias = rearrange(z_bias, "(nheads headdim) -> nheads headdim", headdim=group_size)
            z_bias = z_bias.mean(dim=-1)

            x_bias = self.in_proj.bias[self.nheads * self.headdim:self.nheads * 2 * self.headdim]
            x_bias = rearrange(x_bias, "(nheads headdim) -> nheads headdim", headdim=group_size)
            x_bias = x_bias.mean(dim=-1)
        x_end_conv = self.d_inner
        conv_weight_x = self.conv1d.weight[:x_end_conv, :]
        conv_weight_x = rearrange(conv_weight_x, "(headdim nheads) 1 kernel_size -> 1 kernel_size nheads headdim",
                                  nheads=self.nheads)
        conv_weight_x = rearrange(conv_weight_x, "1 kernel_size nheads (group_size new_headdim) -> 1 kernel_size nheads new_headdim group_size",
                                  group_size=group_size)
        conv_weight_x = conv_weight_x.mean(dim=-1)
        conv_weight_x = rearrange(conv_weight_x, "1 kernel_size  nheads new_headdim -> (nheads new_headdim) 1 kernel_size")
        conv_bias_x = None
        if self.conv1d.bias is not None:
            conv_bias_x = self.conv1d.bias[:x_end_conv]
            conv_bias_x = rearrange(conv_bias_x, "(headdim nheads group_size) -> nheads headdim group_size",
                                    group_size=group_size, nheads=self.nheads)
            conv_bias_x = conv_bias_x.mean(dim=-1)
            conv_bias_x = rearrange(conv_bias_x, "nheads headdim -> (nheads headdim)")

        # revert to original shape
        z_weight = rearrange(z_weight, "nheads d_model new_headdim -> (nheads new_headdim) d_model")
        x_weight = rearrange(x_weight, "nheads d_model new_headdim -> (nheads new_headdim) d_model")
        return z_weight, z_bias, x_weight, x_bias, conv_weight_x, conv_bias_x


    def _mean_pool_nheads(self, num_to_prune):
        """
        mean pool the consecutive nheads from zx heads by grouping them in self.nheads // num_to_prune groups
        """
        assert self.nheads % (self.nheads-num_to_prune) == 0, f"nheads must be divisible by num_to_prune - {self.nheads} % {self.nheads - num_to_prune} != 0"
        group_size = self.nheads // (self.nheads-num_to_prune)
        z_start, z_end = self.get_in_proj_indices()['z']
        z_weight = self.in_proj.weight[z_start:z_end, :]
        z_weight = rearrange(z_weight, "(nheads headdim) d_model -> nheads d_model headdim",
                            headdim=self.headdim)
        z_weight = rearrange(z_weight, "nheads d_model (group_size new_nheads) -> nheads d_model new_nheads group_size",
                            group_size=group_size)
        z_weight = z_weight.mean(dim=-1)
        z_weight = rearrange(z_weight, "nheads d_model new_nheads -> (nheads new_nheads) d_model")

        x_start, x_end = self.get_in_proj_indices()['x']
        x_weight = self.in_proj.weight[x_start:x_end, :]
        x_weight = rearrange(x_weight, "(nheads headdim) d_model -> nheads d_model headdim",
                            headdim=self.headdim)
        x_weight = rearrange(x_weight, "nheads d_model (group_size new_nheads) -> nheads d_model new_nheads group_size",
                            group_size=group_size)
        x_weight = x_weight.mean(dim=-1)
        x_weight = rearrange(x_weight, "nheads d_model new_nheads -> (nheads new_nheads) d_model")

        z_bias = x_bias = None
        if self.in_proj.bias is not None:
            z_bias = self.in_proj.bias[:self.nheads * self.headdim]
            z_bias = rearrange(z_bias, "(nheads headdim) -> nheads headdim", headdim=self.headdim)
            z_bias = rearrange(z_bias, "nheads (group_size new_nheads) -> nheads new_nheads group_size",
                            group_size=group_size)
            z_bias = z_bias.mean(dim=-1)
            z_bias = rearrange(z_bias, "nheads new_nheads -> (nheads new_nheads)")

            x_bias = self.in_proj.bias[self.nheads * self.headdim:self.nheads * 2 * self.headdim]
            x_bias = rearrange(x_bias, "(nheads headdim) -> nheads headdim", headdim=self.headdim)
            x_bias = rearrange(x_bias, "nheads (group_size new_nheads) -> nheads new_nheads group_size",
                            group_size=group_size)
            x_bias = x_bias.mean(dim=-1)
            x_bias = rearrange(x_bias, "nheads new_nheads -> (nheads new_nheads)")

        x_end_conv = self.d_inner
        conv_weight_x = self.conv1d.weight[:x_end_conv, :]
        conv_weight_x = rearrange(conv_weight_x, "(headdim nheads) 1 kernel_size -> 1 kernel_size headdim nheads",
                                    headdim=self.headdim)
        conv_weight_x = rearrange(conv_weight_x, "1 kernel_size headdim (group_size new_nheads) -> 1 kernel_size headdim new_nheads group_size",
                                    group_size=group_size)
        conv_weight_x = conv_weight_x.mean(dim=-1)
        conv_weight_x = rearrange(conv_weight_x, "1 kernel_size  headdim new_nheads -> (headdim new_nheads) 1 kernel_size")
        conv_bias_x = None
        if self.conv1d.bias is not None:
            conv_bias_x = self.conv1d.bias[:x_end_conv]
            conv_bias_x = rearrange(conv_bias_x, "(headdim nheads group_size) -> headdim nheads group_size",
                                    group_size=group_size, headdim=self.headdim)
            conv_bias_x = conv_bias_x.mean(dim=-1)
            conv_bias_x = rearrange(conv_bias_x, "headdim nheads -> (headdim nheads)")


        out_proj_weight = self.out_proj.weight # (d_model, d_inner)
        out_proj_weight = rearrange(out_proj_weight, "d_model (nheads headdim) -> d_model nheads  headdim",
                            headdim=self.headdim)
        out_proj_weight = rearrange(out_proj_weight, "d_model nheads (group_size new_nheads) -> d_model nheads new_nheads group_size",
                            group_size=group_size)
        out_proj_weight = out_proj_weight.mean(dim=-1)
        out_proj_weight = rearrange(out_proj_weight,"d_model nheads new_nheads ->  d_model (nheads new_nheads)")
        out_proj_bias = None
        if self.out_proj.bias is not None:
            out_proj_bias = self.out_proj.bias
            out_proj_bias = rearrange(out_proj_bias, "(nheads headdim) -> nheads headdim", headdim=self.headdim)
            out_proj_bias = rearrange(out_proj_bias, "nheads (group_size new_nheads) -> nheads new_nheads group_size",
                            group_size=group_size)
            out_proj_bias = out_proj_bias.mean(dim=-1)
            out_proj_bias = rearrange(out_proj_bias, "nheads new_nheads -> (nheads new_nheads)")

        D_weight = None
        if self.D_has_hdim:
            D_weight = rearrange(self.D, "(nheads headdim) -> nheads headdim", headdim=self.headdim)
            D_weight = rearrange(D_weight, "nheads (group_size new_nheads) -> nheads new_nheads group_size",
                                group_size=group_size)
            D_weight = D_weight.mean(dim=-1)
            D_weight = rearrange(D_weight, "nheads new_nheads -> (nheads new_nheads)")

        return z_weight, z_bias, x_weight, x_bias, conv_weight_x, conv_bias_x, out_proj_weight, out_proj_bias, D_weight




    @torch.no_grad()
    def prune_headdim(self, ratio, method="random", return_plot=False, exclude_out_proj=False, only_nullify=False, W_mask=None):
        """
        Prune the headdim in dinner sized elements
        """
        if method == 'random':
            in_proj_importance_estimator = self.headdim_random_importance_estimator
            out_proj_importance_estimator = self.out_proj_random_importance_estimator
            D_importance_estimator = self.headdim_random_importance_estimator
            norm_importance_estimator = self.headdim_random_importance_estimator
        elif method == 'taylor_first':
            in_proj_importance_estimator = self.headdim_taylor_first_importance_estimator
            out_proj_importance_estimator = self.outproj_taylor_first_importance_estimator
            D_importance_estimator = self.D_taylor_first_importance_estimator
            norm_importance_estimator = self.norm_taylor_first_importance_estimator
        elif method == 'taylor_second':
            in_proj_importance_estimator = self.headdim_taylor_second_importance_estimator
            out_proj_importance_estimator = self.outproj_taylor_second_importance_estimator
            D_importance_estimator = self.D_taylor_second_importance_estimator
            norm_importance_estimator = self.norm_taylor_second_importance_estimator
        elif method == 'taylor_second_agg':
            in_proj_importance_estimator = self.headdim_taylor_second_importance_estimator_agg
            out_proj_importance_estimator = self.outproj_taylor_second_importance_estimator
            D_importance_estimator = self.D_taylor_second_importance_estimator
            norm_importance_estimator = self.norm_taylor_second_importance_estimator
        elif method == 'mean_pooling':
            in_proj_importance_estimator = self.headdim_random_importance_estimator
            out_proj_importance_estimator = self.out_proj_random_importance_estimator
            D_importance_estimator = self.headdim_random_importance_estimator
            norm_importance_estimator = self.headdim_random_importance_estimator
        elif method == 'mean_pooling_heads_taylor_outproj':
            in_proj_importance_estimator = self.headdim_random_importance_estimator
            out_proj_importance_estimator = self.outproj_taylor_second_importance_estimator
            D_importance_estimator = self.D_taylor_second_importance_estimator
            norm_importance_estimator = self.norm_taylor_second_importance_estimator
        elif method == 'mean_pooling_heads':
            in_proj_importance_estimator = self.headdim_random_importance_estimator
            out_proj_importance_estimator = self.out_proj_random_importance_estimator
            D_importance_estimator = self.headdim_random_importance_estimator
            norm_importance_estimator = self.headdim_random_importance_estimator
        elif method == 'wanda':
            assert W_mask is not None, "mask must be provided for wanda"
            in_proj_importance_estimator = self.headdim_wanda_importance_estimator
            out_proj_importance_estimator = self.out_proj_random_importance_estimator
            D_importance_estimator = self.headdim_random_importance_estimator
        else:
            raise ValueError(f"Unknown method: {method}")

        headdim_num_to_prune = int(self.headdim * ratio)
        nheads_num_to_prune = int(self.nheads * ratio)
        d_inner_num_to_prune = int(self.d_inner * ratio)
        if headdim_num_to_prune == 0:
            return

        # assert self.d_ssm % (self.headdim - headdim_num_to_prune) == 0, f"d_ssm must be divisible by new headdim - {self.d_ssm} % {(self.headdim - headdim_num_to_prune)} != 0"

        zxBCdt_indices = self.get_in_proj_indices()
        in_proj_x_indices_to_prune = torch.empty(0, dtype=torch.long, device='cpu')
        in_proj_z_indices_to_prune = torch.empty(0, dtype=torch.long, device='cpu')
        for h in range(self.nheads):
            x_head_start = zxBCdt_indices['x'][0] + h * self.headdim
            z_head_start = zxBCdt_indices['z'][0] + h * self.headdim
            res_imp = in_proj_importance_estimator(z_head_start, x_head_start) if W_mask is None else in_proj_importance_estimator(z_head_start, x_head_start, mask=W_mask)
            if len(res_imp) == 2:
                z_indices, x_indices = res_imp
            else:
                x_indices = z_indices = res_imp
            z_indices = z_head_start + z_indices[:headdim_num_to_prune]
            x_indices = x_head_start + x_indices[:headdim_num_to_prune]
            in_proj_z_indices_to_prune = in_proj_z_indices_to_prune.to(z_indices.device)
            in_proj_x_indices_to_prune = in_proj_x_indices_to_prune.to(x_indices.device)
            in_proj_z_indices_to_prune = torch.cat((in_proj_z_indices_to_prune, z_indices))
            in_proj_x_indices_to_prune = torch.cat((in_proj_x_indices_to_prune, x_indices))
        if not exclude_out_proj:
            out_proj_indices = out_proj_importance_estimator()[:d_inner_num_to_prune]
        elif not only_nullify:
                self.repeat_y = True
                self.use_mem_eff_path = False

            # Build mask for in_proj.weight and in_proj.bias
        mask_in_proj = torch.ones(self.in_proj.out_features, dtype=torch.bool, device=self.in_proj.weight.device) if not only_nullify else (
            torch.zeros(self.in_proj.out_features, dtype=torch.bool, device=self.in_proj.weight.device))
        mask_in_proj_flags = torch.cat((in_proj_x_indices_to_prune, in_proj_z_indices_to_prune))
        assert (mask_in_proj_flags < self.in_proj.out_features).all().item(), "Indices to prune out of bounds"
        mask_in_proj[mask_in_proj_flags] = False if not only_nullify else True

        plots = []
        if return_plot:
            plots.append(mask_in_proj)

        if method == 'mean_pooling':
            res = self._mean_pool_headdim(headdim_num_to_prune)
            z_weights, z_bias, x_weights, x_bias, conv_weight_x, conv_bias_x = res
        elif method == 'mean_pooling_heads':
            res = self._mean_pool_nheads(nheads_num_to_prune)
            z_weights, z_bias, x_weights, x_bias, conv_weight_x, conv_bias_x,  out_proj_weights,  out_proj_bias, D_weight = res

        if only_nullify:
            self.in_proj.weight[mask_in_proj, :] = 0
            if self.in_proj.bias is not None:
                self.in_proj.bias[mask_in_proj] = 0
        else:
            pruned_in_proj_weight = self.in_proj.weight[mask_in_proj, :]
            if self.in_proj.bias is not None:
                pruned_in_proj_bias = self.in_proj.bias[mask_in_proj]
            # Update in_proj weights and biases
            self.in_proj.weight = nn.Parameter(pruned_in_proj_weight)
            if self.in_proj.bias is not None:
                self.in_proj.bias = nn.Parameter(pruned_in_proj_bias)
            # Adjust output features of in_proj
            self.in_proj.out_features = self.in_proj.weight.shape[0]


        if not only_nullify:
            # prune conv1d
            conv_dim_old = self.conv1d.weight.shape[0]
            conv_mask = torch.ones(conv_dim_old, dtype=torch.bool, device=self.conv1d.weight.device)
            conv_mask_flags = in_proj_x_indices_to_prune - zxBCdt_indices['x'][0]
            assert (conv_mask_flags < (zxBCdt_indices['x'][1] - zxBCdt_indices['x'][0])).all().item(), "Conv indices to prune out of bounds"
            conv_mask[conv_mask_flags] = False
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
                new_x_dim = (self.headdim - headdim_num_to_prune) * self.nheads
                pruned_conv_weights[:new_x_dim, :] = conv_weight_x
            elif method == 'mean_pooling_heads':
                new_x_dim = (self.nheads - nheads_num_to_prune) * self.headdim
                pruned_conv_weights[:new_x_dim, :] = conv_weight_x
            new_conv.weight.data.copy_(pruned_conv_weights)
            if self.conv1d.bias is not None:
                pruned_conv_bias = self.conv1d.bias.data[conv_mask]
                if method == 'mean_pooling':
                    pruned_conv_bias[:new_x_dim] = conv_bias_x
                elif method == 'mean_pooling_heads':
                    pruned_conv_bias[:new_x_dim] = conv_bias_x
                new_conv.bias.data.copy_(pruned_conv_bias)
            self.conv1d = new_conv

        if not exclude_out_proj:
            # prune out_proj
            mask_out_proj = torch.ones(self.out_proj.in_features, dtype=torch.bool, device=self.out_proj.weight.device)
            mask_out_proj[out_proj_indices] = False
            if method == 'mean_pooling_heads':
                pruned_out_proj_weight = out_proj_weights
            else:
                pruned_out_proj_weight = self.out_proj.weight[:, mask_out_proj]
            if self.out_proj.bias is not None:
                pruned_out_proj_bias =  out_proj_bias if method == 'mean_pooling_heads' else self.out_proj.bias[mask_out_proj]
            # Update out_proj weights and biases
            self.out_proj.weight = nn.Parameter(pruned_out_proj_weight)
            if self.out_proj.bias is not None:
                self.out_proj.bias = nn.Parameter(pruned_out_proj_bias)
            # Adjust output features of out_proj
            self.out_proj.in_features = self.out_proj.weight.shape[1]

            # prune D
            if self.D_has_hdim:
                # meanpool D  heads

                if method == 'mean_pooling_heads':
                    self.D = nn.Parameter(D_weight)
                else:
                    D_mask = torch.ones(self.D.shape[0], dtype=torch.bool, device=self.D.device)
                    D_indices_to_prune = torch.empty(0, dtype=torch.long, device='cpu')
                    for D_head in range(self.nheads):
                        D_head_start = D_head * self.headdim
                        D_head_indices_to_prune = D_importance_estimator(D_head_start)
                        D_head_indices_to_prune = D_head_start + D_head_indices_to_prune[:headdim_num_to_prune]
                        D_indices_to_prune = D_indices_to_prune.to(D_head_indices_to_prune.device)
                        D_indices_to_prune = torch.cat((D_indices_to_prune, D_head_indices_to_prune))

                    D_mask[D_indices_to_prune] = False
                    self.D = self.D[D_mask]



        # prune rmsnorm
        if self.rmsnorm:
            norm_indices_to_prune = torch.empty(0, dtype=torch.long, device='cpu')
            for h in range(self.nheads):
                norm_head_start = h * self.headdim
                norm_head_indices_to_prune = norm_importance_estimator(norm_head_start)[:headdim_num_to_prune]
                norm_head_indices_to_prune = norm_head_start + norm_head_indices_to_prune
                norm_indices_to_prune = norm_indices_to_prune.to(norm_head_indices_to_prune.device)
                norm_indices_to_prune = torch.cat((norm_indices_to_prune, norm_head_indices_to_prune))
            mask = torch.ones(self.d_ssm, dtype=torch.bool, device=self.norm.weight.device)
            mask[norm_indices_to_prune] = False
            self.norm.weight = nn.Parameter(self.norm.weight[mask])
            new_norm_dim = self.norm.weight.shape[0]
            self.norm = RMSNormGated(new_norm_dim, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=new_norm_dim // self.ngroups, **factory_kwargs)



        # Adjust self.headdim
        if not only_nullify:
            self.headdim -= headdim_num_to_prune
            self.d_inner = self.d_ssm = self.headdim * self.nheads









####################################################################################################################

    def get_in_proj_indices(self):
        """
        Returns a dictionary of (start, end) row indices for each logical part
        of self.in_proj.weight, corresponding to [z, x, B, C, dt] and also
        the sub-splits [X, B, C] within xBC.

        NOTE: This is adapted for the default (repeat_kv_before_conv=False) layout:
            Order: [z, x, B, C, dt]
            d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
                       = self.in_proj.out_features
        Adjust if your model config differs.
        """
        assert not self.in_proj_is_split, "This method is for non-split in_proj only"

        d_in_proj = self.in_proj.out_features
        indices = {
            'z': (0, self.d_inner),
            'x': (self.d_inner, 2 * self.d_inner),
            'B': (2 * self.d_inner, 2 * self.d_inner + self.ngroups * self.d_state),
            'C': (2 * self.d_inner + self.ngroups * self.d_state, 2 * self.d_inner + 2 * self.ngroups * self.d_state),
            'dt': (2 * self.d_inner + 2 * self.ngroups * self.d_state, d_in_proj),
            'x_conv':(0, self.d_inner),
            'B_conv':(self.d_inner, self.d_inner + self.ngroups * self.d_state),
            'C_conv':(self.d_inner + self.ngroups * self.d_state,  self.d_inner + 2*self.ngroups * self.d_state),

        }
        return indices

####################################################################################################################
### Pruning KQ / BC heads
####################################################################################################################
    def _resize_in_proj_BC(self, new_ngroups: int, head_indices=None):
        mask_dim = self.in_proj.out_features
        zxBCdt_indices = self.get_in_proj_indices()
        B_start = zxBCdt_indices['B'][0]
        C_start = zxBCdt_indices['C'][0]
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
            B_indices = list(range(B_start, B_start + (self.ngroups - new_ngroups) * self.d_state))
            C_indices = list(range(C_start, C_start + (self.ngroups - new_ngroups) * self.d_state))

        mask = torch.ones(mask_dim, dtype=torch.bool, device=self.in_proj.weight.device)
        for idx in B_indices + C_indices:
            mask[idx] = False
        self.in_proj.weight = nn.Parameter(self.in_proj.weight[mask, :])
        if self.in_proj.bias is not None:
            self.in_proj.bias = nn.Parameter(self.in_proj.bias[mask])

        self.in_proj.out_features = self.in_proj.weight.shape[0]

        old_ngroups = self.ngroups
        self.ngroups = new_ngroups

        conv_dim_new = (self.d_inner + 2 * new_ngroups * self.d_state)
        # Update conv1d
        conv_dim_old = self.conv1d.weight.shape[0]
        conv_mask = torch.ones(conv_dim_old, dtype=torch.bool, device=self.conv1d.weight.device)
        if head_indices is not None:
            # Indices to prune in conv1d (x indices)
            for idx in B_indices + C_indices:
                conv_mask[idx] = False
        else:

            group_size = old_ngroups // new_ngroups

            old_conv_dim = self.conv1d.in_channels

            # shape: [old_conv_dim, 1, kernel_size]
            old_weight = self.conv1d.weight
            kernel_size = old_weight.shape[2]

            # Separate out x-channels and BC-channels
            x_size = self.d_inner  # first part is for x
            bc_size = 2 * old_ngroups * self.d_state  # second part is B + C
            assert x_size + bc_size == old_conv_dim, f"Mismatch in old_conv_dim {old_conv_dim} vs. x_size {x_size} + bc_size {bc_size}"

            # 2.1) Keep x-channels as-is (they aren't changing heads)
            x_weight = old_weight[:x_size, :, :]  # shape: (d_inner, 1, kernel_size)

            # 2.2) Group BC channels for averaging
            bc_weight = old_weight[x_size: x_size + bc_size, :, :]  # shape: (bc_size, 1, kernel_size)

            # Reshape bc_weight into (old_ngroups, 2*d_state, 1, kernel_size)
            bc_weight = bc_weight.view(old_ngroups, 2 * self.d_state, 1, kernel_size)
            # shape: (old_ngroups, 2*d_state, 1, kernel_size)

            # Group heads: (new_qk_heads, group_size, 2*d_state, 1, kernel_size)
            bc_weight = rearrange(
                bc_weight,
                "(new_qk_heads group) c_in in_ch ks -> new_qk_heads group c_in in_ch ks",
                new_qk_heads=new_ngroups,
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

                # Reshape: (old_ngroups, 2*d_state)
                bc_bias = bc_bias.view(old_ngroups, 2 * self.d_state)
                bc_bias = rearrange(
                    bc_bias,
                    "(new_qk_heads group) c_in -> new_qk_heads group c_in",
                    new_qk_heads=new_ngroups,
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

    def _mean_pooling_cluster_kq_heads(self, new_qk_head: int):
        """
        average the kq heads into gva_heads number of groups
        """
        assert self.nheads % new_qk_head == 0, "nheads must be divisible by new_qk_head"
        in_proj_indices = self.get_in_proj_indices()
        B_start, B_end = in_proj_indices['B']
        C_start, C_end = in_proj_indices['C']
        B_proj = self.in_proj.weight[B_start:B_end, :]
        C_proj = self.in_proj.weight[C_start:C_end, :]
        print(B_proj.shape)
        B_proj = rearrange(B_proj, "(ngroups d_state) d_model -> ngroups d_model d_state",
                           ngroups=self.ngroups)
        print(B_proj.shape)
        C_proj = rearrange(C_proj, "(ngroups d_state) d_model -> ngroups d_model d_state",
                           ngroups=self.ngroups)

        # mean_pool to new_qk_head heads by averaging every consecutive (self.nheads // new_qk_head) heads
        group = self.nheads // new_qk_head

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

    @torch.no_grad()
    def prune_kq_heads(self, new_qk_head: int, clustering_method='mean_pooling'):
        """
        attention K Q  corresponds to B C matrices here.
        Group kq heads into number of groups that nheads divides.
        1) extract the kq heads from B and C
        then group them into gva_heads number of groups using mean_pooling_cluster_kq_heads
        2) update the self.in_proj weights and dimensions
        3) update the self.conv1d weights and dimensions
        """
        if clustering_method == 'mean_pooling':
            self._mean_pooling_cluster_kq_heads(new_qk_head)
        else:
            raise NotImplementedError(f"Clustering method {clustering_method} is not implemented")


    def update_config(self, config: MambaConfig):
        config.d_inner_list.append(self.d_inner)
        config.in_proj_bias_list.append(self.in_proj.bias is not None)
        config.in_proj_out_features_list.append(self.in_proj.out_features)
        config.out_proj_bias_list.append(self.out_proj.bias is not None)
        config.out_proj_in_features_list.append(self.out_proj.in_features)
        config.conv_dim_list.append(self.conv1d.weight.shape[0])
        config.headdim_list.append(self.headdim)
        config.nheads_list.append(self.nheads)
        config.ngroups_list.append(self.ngroups)
        config.d_state_list.append(self.d_state)
        config.d_ssm_list.append(self.d_ssm)


####################################################################################################################