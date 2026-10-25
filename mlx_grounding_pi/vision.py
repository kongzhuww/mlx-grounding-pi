"""Native MLX MoonViT path for single-image grids.

Architecture follows the pinned GroundingPI/Moonshot reference; see
THIRD_PARTY_NOTICES.md and upstream/hf/LICENSE-Kimi-K3. Video is intentionally
rejected until temporal positioning and processor parity are validated.
"""

import mlx.core as mx
from mlx import nn

from .config import VisionConfig


def validate_grids(grids, count: int) -> list[tuple[int, int, int]]:
    rows = grids.tolist() if hasattr(grids, "tolist") else grids
    if not rows:
        raise ValueError("at least one grid is required")
    validated = []
    for row in rows:
        if len(row) != 3 or any(type(v) is not int or v <= 0 for v in row):
            raise ValueError("grids must contain positive integer (t,h,w) triples")
        if row[0] != 1:
            raise ValueError("video/temporal grids are not supported in this milestone")
        if row[1] > 512 or row[2] > 512:
            raise ValueError("grid exceeds upstream 512x512 rotary limit")
        validated.append(tuple(row))
    if sum(t * h * w for t, h, w in validated) != count:
        raise ValueError("patch count does not match grid dimensions")
    return validated


def resize_position_embedding(weight: mx.array, height: int, width: int) -> mx.array:
    """HWC bilinear interpolation, PyTorch align_corners=False edge semantics."""
    ih, iw, _ = weight.shape
    if (ih, iw) == (height, width):
        return weight.reshape(-1, weight.shape[-1])
    y = mx.maximum((mx.arange(height, dtype=mx.float32) + 0.5) * ih / height - 0.5, 0)
    x = mx.maximum((mx.arange(width, dtype=mx.float32) + 0.5) * iw / width - 0.5, 0)
    y0, x0 = mx.floor(y).astype(mx.int32), mx.floor(x).astype(mx.int32)
    y1, x1 = mx.minimum(y0 + 1, ih - 1), mx.minimum(x0 + 1, iw - 1)
    wy, wx = (y - y0)[:, None, None], (x - x0)[None, :, None]
    a, b = weight[y0[:, None], x0[None, :]], weight[y0[:, None], x1[None, :]]
    c, d = weight[y1[:, None], x0[None, :]], weight[y1[:, None], x1[None, :]]
    out = (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy
    return out.astype(weight.dtype).reshape(height * width, -1)


def rotary_angles(grids, head_dim: int, theta: float = 10000.0) -> mx.array:
    """Alternating x,y complex-pair angles, repeated for each image grid."""
    inv = theta ** (-mx.arange(0, head_dim, 4, dtype=mx.float32) / head_dim)
    parts = []
    for _, h, w in grids:
        pos = mx.arange(h * w)
        x = (pos % w).astype(mx.float32)[:, None] * inv
        y = (pos // w).astype(mx.float32)[:, None] * inv
        parts.append(mx.stack([x, y], axis=-1).reshape(h * w, head_dim // 2))
    return mx.concatenate(parts)


def apply_rotary(x: mx.array, angles: mx.array) -> mx.array:
    """Adjacent-pair rotation in FP32, then cast back to the input dtype."""
    pairs = x.astype(mx.float32).reshape(*x.shape[:-1], -1, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    cos, sin = mx.cos(angles)[:, None, :], mx.sin(angles)[:, None, :]
    out = mx.stack([real * cos - imag * sin, real * sin + imag * cos], axis=-1)
    return out.reshape(x.shape).astype(x.dtype)


def spatial_merge(x: mx.array, grids, kernel=(2, 2)) -> list[mx.array]:
    """Preserve each 2x2 patch, in row-major order; never spatially average."""
    rows = validate_grids(grids, x.shape[0])
    kh, kw = kernel
    outputs, offset = [], 0
    for _, h, w in rows:
        if h % kh or w % kw:
            raise ValueError("grid dimensions must be divisible by the merge kernel")
        seq = x[offset : offset + h * w]
        seq = seq.reshape(h // kh, kh, w // kw, kw, x.shape[-1])
        outputs.append(seq.transpose(0, 2, 1, 3, 4).reshape(-1, kh * kw, x.shape[-1]))
        offset += h * w
    return outputs


class SpatialPositionEmbedding(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.weight = mx.zeros((cfg.init_pos_emb_height, cfg.init_pos_emb_width, cfg.hidden_size))

    def __call__(self, x, grids):
        positions = [resize_position_embedding(self.weight, h, w) for _, h, w in grids]
        return x + mx.concatenate(positions)


class PatchEmbedding(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.proj = nn.Conv2d(
            3,
            cfg.hidden_size,
            cfg.patch_size,
            stride=cfg.patch_size,
            bias=cfg.patch_embed_proj_bias,
        )
        self.pos_emb = SpatialPositionEmbedding(cfg)

    def __call__(self, patches, grids):
        if patches.ndim != 4 or patches.shape[1:] != (3, self.patch_size, self.patch_size):
            raise ValueError("expected patchified NCHW [N,3,patch_size,patch_size] input")
        x = self.proj(patches.transpose(0, 2, 3, 1)).reshape(patches.shape[0], -1)
        return self.pos_emb(x, grids)


class VisionMLP(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.fc0 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.linear_bias)
        self.fc1 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.linear_bias)

    def __call__(self, x):
        return self.fc1(nn.gelu_approx(self.fc0(x)))


class VisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.qkv_hidden_size // cfg.num_attention_heads
        self.norm0 = nn.RMSNorm(cfg.hidden_size, eps=1e-6)
        self.norm1 = nn.RMSNorm(cfg.hidden_size, eps=1e-6)
        self.wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.qkv_hidden_size, bias=cfg.attn_bias)
        self.wo = nn.Linear(cfg.qkv_hidden_size, cfg.hidden_size, bias=cfg.attn_bias)
        self.mlp = VisionMLP(cfg)

    def __call__(self, x, grids, angles):
        packed = self.wqkv(self.norm0(x)).reshape(-1, 3, self.num_heads, self.head_dim)
        q, k, v = packed[:, 0], packed[:, 1], packed[:, 2]
        q, k = apply_rotary(q, angles), apply_rotary(k, angles)
        outputs, offset = [], 0
        # Separate calls avoid a quadratic block-diagonal mask and prevent cross-image leakage.
        for _, h, w in grids:
            end = offset + h * w
            qi, ki, vi = [a[offset:end].transpose(1, 0, 2)[None] for a in (q, k, v)]
            out = mx.fast.scaled_dot_product_attention(qi, ki, vi, scale=self.head_dim**-0.5)
            outputs.append(out[0].transpose(1, 0, 2).reshape(h * w, -1))
            offset = end
        x = x + self.wo(mx.concatenate(outputs))
        return x + self.mlp(self.norm1(x))


class VisionEncoder(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.head_dim = cfg.qkv_hidden_size // cfg.num_attention_heads
        self.theta = cfg.rope_theta
        self.blocks = [VisionBlock(cfg) for _ in range(cfg.num_hidden_layers)]
        self.final_layernorm = nn.RMSNorm(cfg.hidden_size, eps=1e-6)

    def __call__(self, x, grids):
        angles = rotary_angles(grids, self.head_dim, self.theta)
        for block in self.blocks:
            x = block(x, grids, angles)
        return self.final_layernorm(x)


class VisionTower(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.patch_embed = PatchEmbedding(cfg)
        self.encoder = VisionEncoder(cfg)
        self.merge_kernel_size = cfg.merge_kernel_size

    def __call__(self, patches, grids):
        rows = validate_grids(grids, patches.shape[0])
        x = self.encoder(self.patch_embed(patches, rows), rows)
        return spatial_merge(x, rows, self.merge_kernel_size)
