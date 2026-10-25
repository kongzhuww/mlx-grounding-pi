"""Independent small-array NumPy checks, NOT pretrained PyTorch parity."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_grounding_pi.projector import Projector
from mlx_grounding_pi.vision import (
    VisionBlock,
    VisionTower,
    apply_rotary,
    resize_position_embedding,
    rotary_angles,
    spatial_merge,
)


def arr(x):
    return np.array(x)


def linear(module, x):
    result = x @ arr(module.weight).T
    return result + arr(module.bias) if "bias" in module else result


def rms(module, x, eps):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * arr(module.weight)


def layernorm(module, x, eps):
    z = x - x.mean(axis=-1, keepdims=True)
    return z / np.sqrt((z**2).mean(axis=-1, keepdims=True) + eps) * arr(module.weight) + arr(
        module.bias
    )


def gelu(x, approximate=False):
    if approximate:
        return x * 0.5 * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    return x * 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def test_spatial_merge_row_major_and_images():
    x = mx.arange(24).reshape(24, 1)
    a, b = spatial_merge(x, [(1, 4, 4), (1, 2, 4)])
    np.testing.assert_array_equal(
        arr(a[..., 0]), [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
    )
    np.testing.assert_array_equal(arr(b[..., 0]), [[16, 17, 20, 21], [18, 19, 22, 23]])


@pytest.mark.parametrize(
    "grid,count", [([(2, 2, 2)], 8), ([(1, 3, 2)], 6), ([(1, 2, 2)], 8), ([(1.0, 2, 2)], 4)]
)
def test_invalid_grids(grid, count):
    with pytest.raises(ValueError):
        spatial_merge(mx.zeros((count, 8)), grid)


def test_rotary_axes_and_adjacent_pairs():
    angles = rotary_angles([(1, 2, 3)], 8)
    expected = np.array([[x, y, x * 0.01, y * 0.01] for y in range(2) for x in range(3)])
    np.testing.assert_allclose(arr(angles), expected, atol=1e-7)
    x = np.random.default_rng(4).normal(size=(6, 2, 8)).astype(np.float32)
    z = x[..., 0::2] + 1j * x[..., 1::2]
    rotated = z * np.exp(1j * expected[:, None, :])
    result = np.stack([rotated.real, rotated.imag], axis=-1).reshape(x.shape)
    np.testing.assert_allclose(arr(apply_rotary(mx.array(x), angles)), result, atol=3e-7)


@pytest.mark.parametrize("shape", [(1, 1), (2, 3), (4, 7), (3, 2)])
def test_bilinear_half_pixel_positions(shape):
    h, w = shape
    weight = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    expected = np.empty((h, w, 4), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            yy, xx = max((y + 0.5) * 2 / h - 0.5, 0), max((x + 0.5) * 3 / w - 0.5, 0)
            y0, x0 = int(yy), int(xx)
            y1, x1 = min(y0 + 1, 1), min(x0 + 1, 2)
            wy, wx = yy - y0, xx - x0
            expected[y, x] = (1 - wy) * ((1 - wx) * weight[y0, x0] + wx * weight[y0, x1]) + wy * (
                (1 - wx) * weight[y1, x0] + wx * weight[y1, x1]
            )
    np.testing.assert_allclose(
        arr(resize_position_embedding(mx.array(weight), h, w)), expected.reshape(-1, 4), atol=3e-6
    )


def test_block_against_numpy_and_image_isolation(tiny_vision):
    mx.random.seed(7)
    block = VisionBlock(tiny_vision)
    x = np.random.default_rng(7).normal(size=(12, 8)).astype(np.float32)
    grids = [(1, 2, 4), (1, 2, 2)]
    angles = rotary_angles(grids, 8)
    packed = linear(block.wqkv, rms(block.norm0, x, 1e-6)).reshape(12, 3, 2, 8)
    q, k, v = packed[:, 0], packed[:, 1], packed[:, 2]
    # Rotation itself is independently tested above.
    q, k = arr(apply_rotary(mx.array(q), angles)), arr(apply_rotary(mx.array(k), angles))
    pieces = []
    for start, end in [(0, 8), (8, 12)]:
        qi, ki, vi = [a[start:end].transpose(1, 0, 2) for a in (q, k, v)]
        scores = qi @ ki.transpose(0, 2, 1) / math.sqrt(8)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        pieces.append((probs @ vi).transpose(1, 0, 2).reshape(end - start, 16))
    h = x + linear(block.wo, np.concatenate(pieces))
    expected = h + linear(
        block.mlp.fc1, gelu(linear(block.mlp.fc0, rms(block.norm1, h, 1e-6)), True)
    )
    actual = arr(block(mx.array(x), grids, angles))
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    changed = x.copy()
    changed[8:] += 100
    isolated = arr(block(mx.array(changed), grids, angles))
    np.testing.assert_array_equal(actual[:8], isolated[:8])


def test_projector_norm_order_exact_gelu(tiny_vision):
    mx.random.seed(9)
    module = Projector(tiny_vision)
    x = np.random.default_rng(2).normal(size=(3, 4, 8)).astype(np.float32)
    h = layernorm(module.pre_norm, x, 1e-5).reshape(3, 32)
    h = linear(module.fc2, gelu(linear(module.fc1, h)))
    expected = rms(module.post_norm, h, 1e-5)
    np.testing.assert_allclose(arr(module([mx.array(x)])[0]), expected, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError):
        module([mx.zeros((3, 32))])


def test_tower_patch_convolution_and_output_shape(tiny_vision):
    tower = VisionTower(tiny_vision)
    x = np.random.default_rng(0).normal(size=(16, 3, 2, 2)).astype(np.float32)
    weight = arr(tower.patch_embed.proj.weight).transpose(0, 3, 1, 2)
    expected = x.reshape(16, -1) @ weight.reshape(8, -1).T
    actual = tower.patch_embed(mx.array(x), [(1, 4, 4)])
    np.testing.assert_allclose(arr(actual), expected, atol=1e-6)
    output = tower(mx.array(x), [(1, 4, 4)])
    assert len(output) == 1 and output[0].shape == (4, 4, 8)
    assert np.isfinite(arr(output[0])).all()
