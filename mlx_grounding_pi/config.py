"""Configuration for the audited single-image GroundingPI architecture."""

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 27
    num_attention_heads: int = 12
    qkv_hidden_size: int = 1536
    patch_size: int = 14
    num_channels: int = 3
    init_pos_emb_height: int = 64
    init_pos_emb_width: int = 64
    init_pos_emb_time: int = 4
    spatial_merge_size: int = 2
    merge_kernel_size: tuple[int, int] = (2, 2)
    projector_hidden_size: int = 4096
    out_hidden_size: int = 2560
    projector_ln_eps: float = 1e-5
    rope_theta: float = 10000.0
    norm_type: str = "rmsnorm"
    mlp_type: str = "mlp2"
    merge_type: str = "sd2_tpool"
    pos_emb_type: str = "divided_fixed"
    pos_emb_interpolation_mode: str = "bilinear"
    attn_bias: bool = False
    linear_bias: bool = False
    patch_embed_proj_bias: bool = False
    activation_func: str = "gelu_pytorch_tanh"
    projector_hidden_act: str = "gelu"

    def __post_init__(self):
        positive = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "qkv_hidden_size",
            "patch_size",
            "num_channels",
            "init_pos_emb_height",
            "init_pos_emb_width",
            "init_pos_emb_time",
            "spatial_merge_size",
            "projector_hidden_size",
            "out_hidden_size",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.qkv_hidden_size % self.num_attention_heads:
            raise ValueError("qkv_hidden_size must divide evenly into attention heads")
        if (self.qkv_hidden_size // self.num_attention_heads) % 4:
            raise ValueError("2D rotary head dimension must be divisible by four")
        expected = {
            "norm_type": "rmsnorm",
            "mlp_type": "mlp2",
            "merge_type": "sd2_tpool",
            "pos_emb_type": "divided_fixed",
            "pos_emb_interpolation_mode": "bilinear",
            "activation_func": "gelu_pytorch_tanh",
            "projector_hidden_act": "gelu",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"unsupported {name}: {getattr(self, name)!r}")
        if tuple(self.merge_kernel_size) != (self.spatial_merge_size,) * 2:
            raise ValueError("merge_kernel_size and spatial_merge_size disagree")
        if self.num_channels != 3 or self.rope_theta != 10000.0:
            raise ValueError("only the audited RGB / theta=10000 vision path is supported")

    @classmethod
    def from_dict(cls, values: dict):
        kwargs = {f.name: values[f.name] for f in fields(cls) if f.name in values}
        if "merge_kernel_size" in kwargs:
            kwargs["merge_kernel_size"] = tuple(kwargs["merge_kernel_size"])
        return cls(**kwargs)


def load_config(path: str | Path) -> dict:
    """Read JSON only; never load or execute checkpoint Python files."""
    config = json.loads(Path(path).read_text())
    if config.get("model_type") != "groundingpi":
        raise ValueError("expected model_type=groundingpi")
    vision = VisionConfig.from_dict(config["vision_config"])
    if config["text_config"].get("model_type") != "qwen3":
        raise ValueError("only Qwen3 language backbones are supported")
    if vision.out_hidden_size != config["text_config"]["hidden_size"]:
        raise ValueError("projector output does not match language embedding size")
    return config
