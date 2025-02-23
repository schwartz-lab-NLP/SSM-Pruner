import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class MambaConfig:
    d_model: int = 2560
    hidden_size: Optional[int] = None
    use_cache: Optional[bool] = None
    d_intermediate: int = 0
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
    # added by tGhattas
    d_inner_list: list[int] = field(default_factory=list)
    headdim_list: list[int] = field(default_factory=list)
    nheads_list: list[int] = field(default_factory=list)
    ngroups_list: list[int] = field(default_factory=list)
    d_state_list: list[int] = field(default_factory=list)
    d_ssm_list: list[int] = field(default_factory=list)

    in_proj_bias_list: list[bool] = field(default_factory=list)
    in_proj_out_features_list: list[int] = field(default_factory=list)

    out_proj_bias_list: list[bool] = field(default_factory=list)
    out_proj_in_features_list: list[int] = field(default_factory=list)

    conv_dim_list: list[int] = field(default_factory=list)

    def to_json_string(self):
        return json.dumps(asdict(self), indent=2)

    def to_dict(self):
        return asdict(self)
