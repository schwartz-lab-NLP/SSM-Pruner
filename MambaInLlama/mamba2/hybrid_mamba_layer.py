# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from MambaInLlama.mamba2.hybrid_mamba_config import MambaConfig

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

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1 or n_rep == 0:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_xb,
        d_inner=None,
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
        repeat_kv_before_conv=False,
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

        nheads=None,
        in_proj_bias=None,
        in_proj_out_features=None,
        out_proj_bias=None,
        out_proj_in_features=None,
        conv1d_dim=None,
        D_shape=None,
        seqlen=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.seqlen = seqlen
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = d_inner if d_inner is not None else (self.expand * self.d_model) // self.world_size
        # assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        self.ngroups = ngroups // self.world_size
        try:
            assert ngroups % self.world_size == 0
            assert self.d_ssm % self.headdim == 0
        except AssertionError as e:
            print(f"WARNING: ngroups % world_size != 0 or d_ssm % headdim != 0, {ngroups % self.world_size} != 0 or {self.d_ssm % self.headdim} != 0")
            print(e)
        self.nheads = self.d_ssm // self.headdim if nheads is None else nheads
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.d_xb = d_xb
        self.repeat_group = self.d_inner // self.d_xb
        self.repeat_kv_before_conv = repeat_kv_before_conv
        self.conv_bias = conv_bias

        try:
            assert self.d_inner == self.ngroups * self.d_state
            assert self.d_inner == self.d_ssm
        except AssertionError as e:
            print(f"WARNING: d_inner != ngroups * d_state or d_inner != d_ssm, {self.d_inner} != {self.ngroups * self.d_state} or {self.d_inner} != {self.d_ssm}")
            print(e)
        
        self.nheads = self.ngroups if nheads is None else self.nheads
        self.headdim = self.d_state if headdim is None else self.headdim

        # Order: [z, x, B, C, dt]
        # [hidden_dim, hidden_dim, d_state]
        if in_proj_out_features is not None:
            d_in_proj = in_proj_out_features
        else:
            d_in_proj = self.d_inner + self.d_xb + self.d_xb + self.d_inner + self.nheads
        # d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias or in_proj_bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias or in_proj_bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        # conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

        if self.repeat_kv_before_conv:
            conv_dim = self.d_inner + self.d_inner + self.d_inner if conv1d_dim is None else conv1d_dim
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        else:
            conv_dim = self.d_inner + self.d_xb + self.d_xb if conv1d_dim is None else conv1d_dim
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
        if D_shape is not None:
            self.D = nn.Parameter(torch.ones(D_shape, device=device))
        else:
            self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        d_out_proj = self.d_inner if out_proj_in_features is None else out_proj_in_features
        if self.process_group is None:
            self.out_proj = nn.Linear(d_out_proj, self.d_model, bias=bias or out_proj_bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(d_out_proj * self.world_size, self.d_model, bias=bias or out_proj_bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
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

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        # [z, x, B, C, dt]
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.d_xb - self.nheads) // 2

        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.d_xb, self.nheads],
            dim=-1
        )

        if self.repeat_kv_before_conv:
            x, B, C = torch.split(xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1)
            # minic the GQA
            x = rearrange(x, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            # x shape: (bsz, n_group, l, dim)
            B = rearrange(B, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            B = repeat_kv(B, self.repeat_group)
            # combine x, B, C
            x = rearrange(x, "b g l p -> b l (g p)")
            B = rearrange(B, "b g l p -> b l (g p)")
            xBC = torch.cat((x, B, C), dim=-1)

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
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, -(self.dconv - 1):]
            )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
        else:
            # print('conv1d shape:  pre xBC', rearrange(self.conv1d.weight, "d 1 w -> d w").shape)
            try:
                xBC = causal_conv1d_fn(
                    xBC.transpose(1, 2).contiguous(),
                    rearrange(self.conv1d.weight, "d 1 w -> d w").contiguous(),
                    bias=self.conv1d.bias.contiguous(),
                    activation=self.activation,
                    seq_idx=seq_idx,
                ).transpose(1, 2)
            except Exception as e:
                import pdb; pdb.set_trace()
        
        if self.repeat_kv_before_conv:
            x, B, C = torch.split(xBC, [self.ngroups * self.d_state, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

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
            )

        else:
            # self.d_xb + self.d_xb + self.d_inner
            x, B, C = torch.split(xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1)
            
            # minic the GQA
            x = rearrange(x, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            # x shape: (bsz, n_group, l, dim)
            
            B = rearrange(B, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            B = repeat_kv(B, self.repeat_group)
            if  rearrange(x, "b g l p -> b l g p").shape[2] != dt.shape[-1]:
                import pdb; pdb.set_trace()
            y = mamba_chunk_scan_combined(
                # rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                rearrange(x, "b g l p -> b l g p"),
                dt,
                A,
                # rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(B, "b g l n -> b l g n"),
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
            )

        if ssm_state is not None:
            y, last_state, *rest = y
            if cu_seqlens is None:
                ssm_state.copy_(last_state)
            else:
                varlen_states = rest[0]
                ssm_state.copy_(varlen_states)
        y = rearrange(y, "b l h p -> b l (h p)")
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
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.d_xb - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.d_xb, self.nheads],
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

        A = -torch.exp(self.A_log.float())  # (nheads,)

        x, B, C = torch.split(xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1)
        
        # minic the GQA
        x = rearrange(x, "b (xb_group dstate) -> b xb_group dstate", dstate=self.d_state)
        x_reshaped = torch.repeat_interleave(x, dim=1, repeats=self.repeat_group)

        B = rearrange(B, "b (xb_group dstate) -> b xb_group dstate", dstate=self.d_state)
        B = torch.repeat_interleave(B, dim=1, repeats=self.repeat_group)
        
        # SSM step
        assert selective_state_update is not None
            
        A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
        dt = repeat(dt, "b h -> b h p", p=self.headdim)
        dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
        D = repeat(self.D, "h -> h p", p=self.headdim)
        # B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
        # x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        if not self.rmsnorm:
            z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
        y = selective_state_update(
            ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
            dt_bias=dt_bias, dt_softplus=True
        )
        y = rearrange(y, "b h p -> b (h p)")

        if self.rmsnorm:
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

    def wanda_importance(self, *args):
        x_head_start = args[0]
        b_head_start = args[1]
        X_head_weights = self.in_proj.weight[x_head_start:x_head_start + self.headdim, :]
        B_head_weights = self.in_proj.weight[b_head_start:b_head_start + self.headdim, :]

        X_head_wanda = torch.abs(X_head_weights.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))


    def headdim_importance_aux(self, order, *args):
        x_head_start = args[0]
        b_head_start = args[1]

        X_head_weights = self.in_proj.weight[x_head_start:x_head_start + self.headdim, :]
        B_head_weights = self.in_proj.weight[b_head_start:b_head_start + self.headdim, :]
        X_head_weights_acc_grad = self.in_proj.weight.acc_grad[x_head_start:x_head_start + self.headdim, :]
        B_head_weights_acc_grad = self.in_proj.weight.acc_grad[b_head_start:b_head_start + self.headdim, :]

        if hasattr(self.in_proj.weight, 'acc_hess'):
            X_head_weights_acc_hess = self.in_proj.weight.acc_hess[x_head_start:x_head_start + self.headdim, :]
            B_head_weights_acc_hess = self.in_proj.weight.acc_hess[b_head_start:b_head_start + self.headdim, :]

        X_head_salience = (X_head_weights * X_head_weights_acc_grad)
        B_head_salience = (B_head_weights * B_head_weights_acc_grad)

        if order == 2:
            if not hasattr(self.in_proj.weight, 'acc_hess'):
                X_head_salience = X_head_salience * X_head_weights
                B_head_salience = B_head_salience * B_head_weights
            else:
                X_head_salience = X_head_salience + 0.5 * X_head_weights_acc_hess * (X_head_weights ** 2)
                B_head_salience = B_head_salience + 0.5 * B_head_weights_acc_hess * (B_head_weights ** 2)

        X_head_taylor_imp = X_head_salience.abs().mean(dim=1)  # (headdim,)
        B_head_taylor_imp = B_head_salience.abs().mean(dim=1) # (headdim,)
        assert  X_head_taylor_imp.shape == B_head_taylor_imp.shape == (self.headdim,), f"X_head_taylor_imp.shape: {X_head_taylor_imp.shape}, B_head_taylor_imp.shape: {B_head_taylor_imp.shape}"

        return X_head_taylor_imp, B_head_taylor_imp

    def outproj_taylor_importance_aux(self, order, *args):
        o_weights = self.out_proj.weight
        o_weights_acc_grad = self.out_proj.weight.acc_grad
        outproj_salience = (o_weights * o_weights_acc_grad)
        if order == 2:

            if not hasattr(self.out_proj.weight, 'acc_hess'):
                outproj_salience = outproj_salience * o_weights
            else:
                outproj_salience = outproj_salience + 0.5 * outproj_salience * (o_weights ** 2)

        outproj_taylor_imp = outproj_salience.abs().mean(dim=1)  # (d_model,)
        return outproj_taylor_imp


    def dstate_taylor_importance_aux(self, order, *args):
        idx_C_start = args[0]
        C_weights = self.in_proj.weight[idx_C_start:idx_C_start + self.d_state, :]
        C_weights_acc_grad = self.in_proj.weight.acc_grad[idx_C_start:idx_C_start + self.d_state, :]

        if hasattr(self.in_proj.weight, 'acc_hess'):
            C_weights_acc_hess = self.in_proj.weight.acc_hess[idx_C_start:idx_C_start + self.d_state, :]
        C_salience = (C_weights * C_weights_acc_grad)

        if order == 2:
            if not hasattr(self.in_proj.weight, 'acc_hess'):
                C_salience = C_salience * C_weights
            else:
                C_salience = C_salience + 0.5 * C_weights_acc_hess * (C_weights ** 2)

        C_taylor_imp = C_salience.abs().mean(dim=1)  # (d_state,)
        return C_taylor_imp

    def _mean_pool_dstates(self, num_to_prune):
        """
        mean pool the consecutive dstates from BC heads by grouping them in self.dstates // num_to_prune groups
        """
        assert self.d_state % num_to_prune == 0, f"d_state must be divisible by num_to_prune - {self.d_state} % {num_to_prune} != 0"
        group_size = self.d_state // num_to_prune
        C_start, C_end = self.get_in_proj_indices()['C']
        C_weight = self.in_proj.weight[C_start:C_end, :]
        C_weight = rearrange(C_weight, "(ngroups d_state) d_model -> ngroups d_model d_state",
                            n_qk_heads=self.n_qk_heads)
        C_weight = rearrange(C_weight, "ngroups d_model (new_d_state group_size) -> ngroups d_model new_d_state group_size",
                            group_size=group_size)
        C_weight = C_weight.mean(dim=-1) # (n_qk_heads, d_model, d_state)
        C_weight = rearrange(C_weight, "ngroups d_model new_d_state -> (ngroups new_d_state) d_model")

        C_bias = None
        if self.in_proj.bias is not None:
            C_bias = self.in_proj.bias[self.d_inner + self.n_qk_heads * self.d_state:self.d_inner + 2 * self.n_qk_heads * self.d_state]
            C_bias = rearrange(C_bias, "(ngroups d_state) -> ngroups d_state", group_size=group_size)
            C_bias = rearrange(C_bias, "ngroups (new_d_state group_size) -> ngroups new_d_state group_size",
                            group_size=group_size)
            C_bias = C_bias.mean(dim=-1)
            C_bias = rearrange(C_bias, "ngroups new_d_state -> (ngroups new_d_state)")

        conv_weight_C = self.conv1d.weight[2*self.d_xb:2*self.d_xb + self.ngroups * self.d_state,:]
        conv_weight_C = rearrange(conv_weight_C, "(ngroups d_state) 1 kernel_size -> ngroups 1 kernel_size d_state",
                                    n_qk_heads=self.n_qk_heads)
        conv_weight_C = rearrange(conv_weight_C, "ngroups 1 kernel_size (new_d_state group_size) -> ngroups 1 kernel_size new_d_state group_size",
                                    group_size=group_size)
        conv_weight_C = conv_weight_C.mean(dim=-1)
        conv_weight_C = rearrange(conv_weight_C, "ngroups 1 kernel_size new_d_state -> (ngroups new_d_state) 1 kernel_size")
        conv_bias_C = None
        if self.conv1d.bias is not None:
            conv_bias_C = self.conv1d.bias[2*self.d_xb:2*self.d_xb + self.ngroups * self.d_state,:]
            conv_bias_C = rearrange(conv_bias_C, "(ngroups d_state) -> ngroups d_state", n_qk_heads=self.n_qk_heads,
                                    group_size=group_size)
            conv_bias_C = rearrange(conv_bias_C, "ngroups (new_d_state group_size) -> ngroups new_d_state group_size",
                                    group_size=group_size)
            conv_bias_C = conv_bias_C.mean(dim=-1)
            conv_bias_C = rearrange(conv_bias_C, "ngroups new_d_state -> (ngroups new_d_state)")

        return C_weight, C_bias, conv_weight_C, conv_bias_C

    def dstates_taylor_first_C_importance_estimator(self, *args):
        C_taylor_imp = self.dstate_taylor_importance_aux(1, *args)
        _, indices_C = torch.sort(C_taylor_imp)  # ascending order
        return indices_C

    def dstates_taylor_second_C_importance_estimator(self, *args):
        C_taylor_imp = self.dstate_taylor_importance_aux(2, *args)
        _, indices_C = torch.sort(C_taylor_imp)  # ascending order
        return indices_C

    def dstates_random_importance_estimator(self, *args):
        """
        Assign random importance of each d_state dimension
        """
        indices = torch.randperm(self.d_state)
        return indices

    def xb_random_importance_estimator(self, *args):
        """
        Assign random importance of each headdim dimension
        """
        indices = torch.randperm(self.headdim)
        return indices, indices

    def xb_taylor_first_importance_estimator(self, *args):
        """
        Assign taylor estimation importance of each headdim dimension
        """
        indices_x, indices_b = self.headdim_importance_aux(1, *args)
        agg_imp = indices_x + indices_b
        _, indices_x  = _, indices_b = torch.sort(agg_imp)
        return indices_x, indices_b

    def xb_taylor_second_importance_estimator_agg(self, *args):
        """
        Assign taylor estimation importance of each headdim dimension
        """
        indices_x, indices_b = self.headdim_importance_aux(2, *args)
        agg_imp = indices_x + indices_b
        _, indices_x  = _, indices_b = torch.sort(agg_imp)
        return indices_x, indices_b

    def xb_taylor_second_importance_estimator(self, *args):
        """
        Assign taylor estimation importance of each headdim dimension
        """
        indices_x, indices_b = self.headdim_importance_aux(2, *args)

        _, indices_x  = torch.sort(indices_x)
        _, indices_b = torch.sort(indices_b)
        return indices_x, indices_b


    def xb_wanda_importance_estimator(self, *args):
        """
        Assign wanda estimation importance of each headdim dimension
        """
        indices_x, indices_b = self.headdim_importance_aux(2, *args)

        _, indices_x  = torch.sort(indices_x)
        _, indices_b = torch.sort(indices_b)
        return indices_x, indices_b


    def outproj_random_importance_estimator(self, *args):
        """
        Assign random importance of each d_state dimension
        """
        indices = torch.randperm(self.d_inner)
        return indices

    def outproj_taylor_first_importance_estimator(self, *args):
        """
        Assign taylor estimation importance of each d_state dimension
        """
        indices = self.outproj_taylor_importance_aux(1, *args)
        _, indices = torch.sort(indices)
        return indices

    def outproj_taylor_second_importance_estimator(self, *args):

        indices = self.outproj_taylor_importance_aux(2, *args)
        _, indices = torch.sort(indices)
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

    @torch.no_grad()
    def prune_dstates(self, ratio, method="taylor_second", return_plot=False, prune_xb=False):
        """
        Prune the lowest magnitude d_state dimensions per head by removing their parameters according to the given ratio.
        The same number of elements are removed from each head, and d_state is reduced accordingly.
        """
        if method == 'random':
            dstates_importance_estimator = self.dstates_random_importance_estimator
            xb_importance_estimator = self.xb_random_importance_estimator
            out_proj_importance_estimator = self.outproj_random_importance_estimator
        elif method == "taylor_second":
            dstates_importance_estimator = self.dstates_taylor_second_C_importance_estimator
            out_proj_importance_estimator = self.outproj_taylor_second_importance_estimator
            xb_importance_estimator = self.xb_taylor_second_importance_estimator
        # elif method == 'magnitude':
        #     importance_estimator = self.dstates_magnitude_importance_estimator_summed
        elif method == 'mean_pooling':
            dstates_importance_estimator = self.dstates_random_importance_estimator

        d_xb_num_to_prune = int(self.d_xb * ratio) if prune_xb else 0
        dstate_num_to_prune = int(self.d_state * ratio)
        if method == 'mean_pooling':
            assert self.d_state % dstate_num_to_prune == 0, f"d_state must be divisible by num_to_prune - {self.d_state} % {dstate_num_to_prune} != 0"

        if dstate_num_to_prune == 0:
            return  # Nothing to prune

        indices_to_prune = []
        indices_to_prune_conv = []
        plots = []
        indices = self.get_in_proj_indices()
        C_start = indices['C'][0]
        norm_indices_to_prune = []
        for h in range(self.ngroups):
            # Indices in in_proj corresponding to B and C for head h
            idx_C_start = C_start + h * self.d_state

            indices_C = dstates_importance_estimator(idx_C_start)
            indices_prune_head_C = indices_C[:dstate_num_to_prune].tolist()
            norm_indices_to_prune.extend([(h * self.d_state + i) for i in indices_prune_head_C])
            # Map indices to global indices in in_proj
            # For C
            indices_to_prune.extend([idx_C_start + i for i in indices_prune_head_C])

            # for conv1d channels
            idx_C_conv = 2 * self.d_xb + h * self.d_state

            indices_to_prune_conv.extend([idx_C_conv + i for i in indices_prune_head_C])

        if prune_xb:
            xb_heads = self.d_xb // self.headdim
            headdim_to_prune = int(self.headdim * ratio)
            for h in range(xb_heads):
                x_head_start = indices['x'][0] + h * self.headdim
                b_head_start = indices['B'][0] + h * self.headdim
                indices_x_conv, indices_b_conv = indices_x_inproj, indices_b_inproj = xb_importance_estimator(
                    x_head_start, b_head_start)

                indices_x_inproj = indices_x_inproj[:headdim_to_prune]
                indices_b_inproj = indices_b_inproj[:headdim_to_prune]
                indices_x_conv = indices_x_conv[:headdim_to_prune]
                indices_b_conv = indices_b_conv[:headdim_to_prune]

                indices_x_inproj = indices_x_inproj + x_head_start
                indices_b_inproj = indices_b_inproj + b_head_start

                indices_to_prune.extend(indices_x_inproj.tolist())
                indices_to_prune.extend(indices_b_inproj.tolist())

                old_x_conv_start = 0
                old_b_conv_start = self.d_xb if not self.repeat_kv_before_conv else self.d_inner
                if self.repeat_kv_before_conv:
                    for h in range(self.repeat_group):
                        indices_x_conv_tmp = indices_x_conv + old_x_conv_start + h * self.d_xb
                        indices_to_prune_conv.extend(indices_x_conv_tmp.tolist())

                        indices_b_conv_tmp = indices_b_conv + old_b_conv_start + h * self.d_xb
                        indices_to_prune_conv.extend(indices_b_conv_tmp.tolist())
                else:
                    indices_x_conv = indices_x_conv + old_x_conv_start + h * self.headdim
                    indices_b_conv = indices_b_conv + old_b_conv_start + h * self.headdim
                    indices_to_prune_conv.extend(indices_x_conv.tolist())
                    indices_to_prune_conv.extend(indices_b_conv.tolist())

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
            mean_pooled_BC_dstates = self._mean_pool_dstates(dstate_num_to_prune)
            C_weights, C_bias, conv_weight_C, conv_bias_C = mean_pooled_BC_dstates
            pruned_in_proj_weight = self.in_proj.weight[mask_in_proj, :]
            if self.in_proj.bias is not None:
                pruned_in_proj_bias = self.in_proj.bias[mask_in_proj, :]
            new_d_state = self.d_state - dstate_num_to_prune
            pruned_in_proj_weight[C_start:C_start + self.ngroups * new_d_state, :] = C_weights
        # Update in_proj weights and biases
        self.in_proj.weight = nn.Parameter(pruned_in_proj_weight)
        if self.in_proj.bias is not None:
            self.in_proj.bias = nn.Parameter(pruned_in_proj_bias)
        # assert self.repeat_group ==  self.d_inner // self.d_xb, f"repeat_group: {self.repeat_group}, d_inner: {self.d_inner}, d_xb: {self.d_xb}"

        # adjust dt
        # self.dt_bias = nn.Parameter(self.dt_bias[:self.nheads])

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
            # copy C weights to the new conv1d layer
            pruned_conv_weights[2 * self.d_xb:2 * self.d_xb + self.ngroups * self.d_state, :] = conv_weight_C
        # Transfer weights and biases from the old conv1d layer
        new_conv.weight.data.copy_(pruned_conv_weights)
        if self.conv1d.bias is not None:
            pruned_conv_bias = self.conv1d.bias.data[conv_mask]
            if method == 'mean_pooling':
                pruned_conv_bias[2 * self.d_xb:2 * self.d_xb + self.ngroups * self.d_state] = conv_bias_C
            new_conv.bias.data.copy_(pruned_conv_bias)

        print('old  conv1d shape----', self.conv1d.weight.shape)
        print('new conv1d shape----', new_conv.weight.shape)
        self.conv1d = new_conv


        if self.rmsnorm:
            mask = torch.ones(self.d_ssm, dtype=torch.bool, device=self.norm.weight.device)
            for idx in norm_indices_to_prune:
                mask[idx] = False
            new_norm_dim = self.norm.weight.shape[0]
            new_norm = RMSNormGated(new_norm_dim, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=new_norm_dim // self.ngroups, **factory_kwargs)
            new_norm.weight = nn.Parameter(self.norm.weight[mask])
            self.norm = new_norm

        # prune out_proj

        out_proj_mask = torch.ones(self.out_proj.out_features, dtype=torch.bool, device=self.out_proj.weight.device)
        indices_out_proj = out_proj_importance_estimator()
        indices_out_proj = indices_out_proj[:self.d_inner]
        for idx in indices_out_proj:
            out_proj_mask[idx] = False

        pruned_out_proj_weight = self.out_proj.weight[:, out_proj_mask]
        self.out_proj.weight = nn.Parameter(pruned_out_proj_weight)
        self.out_proj.in_features = self.out_proj.weight.shape[1]

        # Adjust output features of in_proj
        self.in_proj.out_features = self.in_proj.weight.shape[0]

        # Adjust self.d_state
        self.d_state -= dstate_num_to_prune
        self.d_inner = self.ngroups * self.d_state
        self.d_ssm = self.d_inner

        if prune_xb:
            self.headdim = self.d_state
        else:
            assert self.d_inner % self.d_xb == 0, f"d_inner must be divisible by d_xb - {self.d_inner} % {self.d_xb} != 0"
            self.repeat_group = self.d_inner // self.d_xb
        self.d_xb -= d_xb_num_to_prune

        # Update the flag
        self.just_pruned = True

    @torch.no_grad()
    def prune_xb(self,  ratio, method='taylor_second'):


        if method == 'random':
            xb_importance_estimator = self.xb_random_importance_estimator
        elif method == "taylor_first":
            xb_importance_estimator = self.xb_taylor_first_importance_estimator
        elif method == "taylor_second":
            xb_importance_estimator = self.xb_taylor_second_importance_estimator
        elif method ==  "mean_pooling_heads":
            xb_importance_estimator = self.xb_random_importance_estimator

        headdim_to_prune = int(self.headdim * ratio)

        if headdim_to_prune == 0:
            return

        indices_to_prune = []
        indices_to_prune_conv = []
        plots = []

        in_proj_indices = self.get_in_proj_indices()
        xb_heads = self.d_xb // self.headdim

        for h in range(xb_heads):
            x_head_start = in_proj_indices['x'][0] + h * self.headdim
            b_head_start = in_proj_indices['B'][0] + h * self.headdim
            indices_x_conv, indices_b_conv = indices_x_inproj, indices_b_inproj = xb_importance_estimator(x_head_start, b_head_start)

            indices_x_inproj = indices_x_inproj[:headdim_to_prune]
            indices_b_inproj = indices_b_inproj[:headdim_to_prune]
            indices_x_conv = indices_x_conv[:headdim_to_prune]
            indices_b_conv = indices_b_conv[:headdim_to_prune]

            indices_x_inproj = indices_x_inproj +  x_head_start
            indices_b_inproj = indices_b_inproj + b_head_start

            indices_to_prune.extend(indices_x_inproj.tolist())
            indices_to_prune.extend(indices_b_inproj.tolist())

            old_x_conv_start = 0
            old_b_conv_start = self.d_xb if not self.repeat_kv_before_conv else self.d_inner
            if self.repeat_kv_before_conv:
                for h in range(self.repeat_group):
                    indices_x_conv_tmp = indices_x_conv + old_x_conv_start + h * self.d_xb
                    indices_to_prune_conv.extend(indices_x_conv_tmp.tolist())

                    indices_b_conv_tmp = indices_b_conv + old_b_conv_start + h * self.d_xb
                    indices_to_prune_conv.extend(indices_b_conv_tmp.tolist())
            else:
                indices_x_conv = indices_x_conv + old_x_conv_start + h * self.headdim
                indices_b_conv = indices_b_conv + old_b_conv_start + h * self.headdim
                indices_to_prune_conv.extend(indices_x_conv.tolist())
                indices_to_prune_conv.extend(indices_b_conv.tolist())


        # Build mask for in_proj.weight and in_proj.bias
        mask_in_proj = torch.ones(self.in_proj.out_features, dtype=torch.bool, device=self.in_proj.weight.device)
        for idx in indices_to_prune:
            mask_in_proj[idx] = False


        if method != 'mean_pooling_heads':
            pruned_in_proj_weight = self.in_proj.weight[mask_in_proj, :]
            if self.in_proj.bias is not None:
                pruned_in_proj_bias = self.in_proj.bias[mask_in_proj]
            new_d_xb = self.d_xb - headdim_to_prune * xb_heads

        else:
            mean_pooled_XB = self._mean_pool_xb(ratio)
            X_weights, B_weights, conv_weight_X, conv_weight_B, conv_bias_X, conv_bias_B = mean_pooled_XB
            pruned_in_proj_weight = self.in_proj.weight[mask_in_proj, :]
            if self.in_proj.bias is not None:
                pruned_in_proj_bias = self.in_proj.bias[mask_in_proj, :]
            new_d_xb = X_weights.shape[0]
            pruned_in_proj_weight[in_proj_indices['x'][0]:in_proj_indices['x'][0] + new_d_xb, :] = X_weights
            pruned_in_proj_weight[in_proj_indices['x'][0] + new_d_xb:in_proj_indices['x'][0] + new_d_xb + new_d_xb, :] = B_weights
        # Update in_proj weights and biases
        self.in_proj.weight = nn.Parameter(pruned_in_proj_weight)
        if self.in_proj.bias is not None:
            self.in_proj.bias = nn.Parameter(pruned_in_proj_bias)



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
        if self.conv1d.bias is not None:
            pruned_conv_bias = self.conv1d.bias.data[conv_mask]
        new_x_conv_start = 0
        new_b_conv_start = new_d_xb if not self.repeat_kv_before_conv else (self.d_inner - headdim_to_prune * xb_heads * self.repeat_group)
        if method == 'mean_pooling':
            pruned_conv_weights[new_x_conv_start: new_x_conv_start + new_d_xb, :] = conv_weight_X
            pruned_conv_weights[new_b_conv_start: new_b_conv_start + new_d_xb, :] = conv_weight_B
            if self.conv1d.bias is not None:
                pruned_conv_bias[new_x_conv_start: new_x_conv_start + new_d_xb] = conv_bias_X
                pruned_conv_bias[new_b_conv_start: new_b_conv_start + new_d_xb] = conv_bias_B

        # Transfer weights and biases from the old conv1d layer
        new_conv.weight.data.copy_(pruned_conv_weights)
        if self.conv1d.bias is not None:
            new_conv.bias.data.copy_(pruned_conv_bias)

        self.conv1d = new_conv


        # Adjust output features of in_proj and update d_xb
        self.in_proj.out_features = self.in_proj.weight.shape[0]
        self.d_xb = new_d_xb

        self.repeat_group = self.d_inner // self.d_xb



        self.just_pruned = True



    def _mean_pool_xb(self, ratio):
        """
        mean pool the consecutive xb heads
        """
        in_proj_indices = self.get_in_proj_indices()
        x = self.in_proj.weight[in_proj_indices['x'][0]:in_proj_indices['x'][1], :]
        B = self.in_proj.weight[in_proj_indices['B'][0]:in_proj_indices['B'][1], :]
        x = rearrange(x, "(xb_group dstate) l  -> xb_group l dstate", dstate=self.d_state)
        B = rearrange(B, "(xb_group dstate) l  -> xb_group l dstate", dstate=self.d_state)
        current_group_size = x.shape[0]
        new_group_size = current_group_size - int(current_group_size * ratio)
        # assert that we can mean pool the current_group_size to get the new_group_size
        assert current_group_size % new_group_size == 0, f"current_group_size: {current_group_size}, new_group_size: {new_group_size}"
        x = rearrange(x, "(xb_to_merge group_size) l dstate -> (group_size dstate) l xb_to_merge", group_size=new_group_size)
        B = rearrange(B, "(xb_to_merge group_size) l dstate -> (group_size dstate) l xb_to_merge", group_size=new_group_size)
        x = x.mean(dim=-1)
        B = B.mean(dim=-1)


        conv_weight_X = self.conv1d.weight[:self.d_xb, :]
        conv_weight_B = self.conv1d.weight[self.d_xb:2*self.d_xb, :]
        conv_bias_X = None
        conv_bias_B = None
        if self.conv1d.bias is not None:
            conv_bias_X = self.conv1d.bias[:self.d_xb]
            conv_bias_B = self.conv1d.bias[self.d_xb:2*self.d_xb]
        conv_weight_X = rearrange(conv_weight_X, "(xb_to_merge d_state group_size) 1 kernel_size -> (d_state group_size) 1 kernel_size  xb_to_merge",
                                  d_state=self.d_state, group_size=new_group_size)
        conv_weight_X = conv_weight_X.mean(dim=-1)
        conv_weight_B = rearrange(conv_weight_B, "(xb_to_merge d_state group_size) 1 kernel_size ->  (d_state group_size) 1 kernel_size  xb_to_merge",
                                    d_state=self.d_state, group_size=new_group_size)
        conv_weight_B = conv_weight_B.mean(dim=-1)
        if conv_bias_X is not None:
            conv_bias_X = rearrange(conv_bias_X, "(xb_to_merge d_state group_size) -> d_state group_size xb_to_merge", d_state=self.d_state, group_size=new_group_size)
            conv_bias_X = conv_bias_X.mean(dim=-1)
            conv_bias_X = rearrange(conv_bias_X, "d_state group_size -> (d_state group_size)")
        if conv_bias_B is not None:
            conv_bias_B = rearrange(conv_bias_B, "(xb_to_merge d_state group_size) -> d_state group_size xb_to_merge", d_state=self.d_state, group_size=new_group_size)
            conv_bias_B = conv_bias_B.mean(dim=-1)
            conv_bias_B = rearrange(conv_bias_B, "d_state group_size -> (d_state group_size)")

        # revert to original order


        return x, B, conv_weight_X, conv_weight_B, conv_bias_X, conv_bias_B




    def get_in_proj_indices(self):
        """
        Returns a dictionary of (start, end) row indices for each logical part
        of self.in_proj.weight, corresponding to [z, x, B, C, dt]

        This is adapted for the default (repeat_kv_before_conv=False) layout:
        d_in_proj = self.d_inner + self.d_xb + self.d_xb + self.d_inner + self.nheads
        """
        indices = {}
        idx = 0
        indices['z'] = (idx, idx + self.d_inner)
        idx += self.d_inner
        indices['x'] = (idx, idx + self.d_xb)
        idx += self.d_xb
        indices['B'] = (idx, idx + self.d_xb)
        idx += self.d_xb
        indices['C'] = (idx, idx + self.d_inner)
        idx += self.d_inner
        indices['dt'] = (idx, idx + self.nheads)

        idx = 0
        indices['x_conv'] = (idx, idx + self.d_xb)
        idx += self.d_xb
        indices['B_conv'] = (idx, idx + self.d_xb)
        idx += self.d_xb
        indices['C_conv'] = (idx, idx + self.d_inner)
        return indices


    def update_config(self, config: MambaConfig):
        layer_idx = int(self.layer_idx)
        config.d_inner_dict[layer_idx] = (self.d_inner)
        config.d_xb_dict[layer_idx] = (self.d_xb)
        config.in_proj_bias_dict[layer_idx] = (self.in_proj.bias is not None)
        config.in_proj_out_features_dict[layer_idx] = (self.in_proj.out_features)
        config.out_proj_bias_dict[layer_idx] = (self.out_proj.bias is not None)
        config.out_proj_in_features_dict[layer_idx] = (self.out_proj.in_features)
        config.conv_dim_dict[layer_idx] = (self.conv1d.weight.shape[0])
        config.headdim_dict[layer_idx] = (self.headdim)
        config.nheads_dict[layer_idx] = (self.nheads)
        config.ngroups_dict[layer_idx] = (self.ngroups)
        config.d_state_dict[layer_idx] = (self.d_state)
        config.d_ssm_dict[layer_idx] = (self.d_ssm)
        config.D_dict[layer_idx] = (self.D.shape[0])


