import json
from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class MambaConfig:
    d_model: int = 2560
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm_eps: float = 1e-5
    vocab_size: int = None
    d_inner: int = None
    d_xb: int = 2560
    intermediate_size: int = 10240
    hidden_act: str = "silu"
    n_layer: int = 32
    attn_layers: List[int] = field(default_factory=list)

    # added by tGhattas
    d_inner_dict: dict[int, int] = field(default_factory=dict)
    d_xb_dict: dict[int, int] = field(default_factory=dict)
    headdim_dict: dict[int, int] = field(default_factory=dict)
    nheads_dict: dict[int, int] = field(default_factory=dict)
    ngroups_dict: dict[int, int] = field(default_factory=dict)
    d_state_dict: dict[int, int] = field(default_factory=dict)
    d_ssm_dict: dict[int, int] = field(default_factory=dict)

    in_proj_bias_dict: dict[int, bool] = field(default_factory=dict)
    in_proj_out_features_dict: dict[int, int] = field(default_factory=dict)

    out_proj_bias_dict: dict[int, bool] = field(default_factory=dict)
    out_proj_in_features_dict: dict[int, int] = field(default_factory=dict)

    conv_dim_dict: dict[int, int] = field(default_factory=dict)

    D_dict: dict[int, int] = field(default_factory=dict)


    # MLP
    gate_proj_in_features_dict: dict[int, int] = field(default_factory=dict)
    gate_proj_out_features_dict: dict[int, int] = field(default_factory=dict)
    gate_proj_bias_dict: dict[int, bool] = field(default_factory=dict)

    up_proj_in_features_dict: dict[int, int] = field(default_factory=dict)
    up_proj_out_features_dict: dict[int, int] = field(default_factory=dict)
    up_proj_bias_dict: dict[int, bool] = field(default_factory=dict)

    down_proj_in_features_dict: dict[int, int] = field(default_factory=dict)
    down_proj_out_features_dict: dict[int, int] = field(default_factory=dict)
    down_proj_bias_dict: dict[int, bool] = field(default_factory=dict)

    def to_json_string(self):
        return json.dumps(asdict(self), indent=2)

    def to_dict(self):
        return asdict(self)