import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache

from mlx_grounding_pi.convert import map_weight_key
from mlx_grounding_pi.model import Model


def test_multimodal_forward_and_cached_decode(tiny_config):
    mx.random.seed(10)
    model = Model(tiny_config)
    patches = mx.random.normal((16, 3, 2, 2))
    grids = [(1, 4, 4)]
    prompt = mx.array([[1, 31, 31, 31, 31, 2]])
    full = model(prompt, pixel_values=patches, image_grid_thw=grids)
    assert full.shape == (1, 6, 64)
    assert np.isfinite(np.array(full)).all()
    cache = make_prompt_cache(model)
    model(prompt[:, :-1], cache=cache, pixel_values=patches, image_grid_thw=grids)
    decoded = model(prompt[:, -1:], cache=cache)
    np.testing.assert_allclose(np.array(decoded[:, 0]), np.array(full[:, -1]), atol=3e-6, rtol=3e-5)
    with pytest.raises(ValueError, match="prefill"):
        model(prompt, cache=cache, pixel_values=patches, image_grid_thw=grids)


def test_count_and_batch_guards(tiny_config):
    model = Model(tiny_config)
    patches = mx.zeros((16, 3, 2, 2))
    with pytest.raises(ValueError, match="placeholder count"):
        model(mx.array([[1, 31, 2]]), pixel_values=patches, image_grid_thw=[(1, 4, 4)])
    with pytest.raises(ValueError, match="supplied together"):
        model(mx.array([[1, 2]]), pixel_values=patches)
    with pytest.raises(ValueError, match="only"):
        model(mx.array([[1], [2]]))
    with pytest.raises(ValueError, match="require image inputs"):
        model(mx.array([[1, 31, 2]]))
    with pytest.raises(ValueError, match="nonempty"):
        model(mx.zeros((1, 0), dtype=mx.int32))


def test_fp32_processor_pixels_follow_bf16_vision_weights(tiny_config):
    from mlx.utils import tree_map

    model = Model(tiny_config)
    model.update(tree_map(lambda x: x.astype(mx.bfloat16), model.parameters()))
    pixels = mx.random.normal((16, 3, 2, 2))
    features = model.visual(pixels, [(1, 4, 4)])
    assert pixels.dtype == mx.float32 and features.dtype == mx.bfloat16
    actual = model(
        mx.array([[1, 31, 31, 31, 31, 2]]), pixel_values=pixels, image_grid_thw=[(1, 4, 4)]
    )
    assert actual.dtype == mx.bfloat16


def test_actual_parameter_names_match_checkpoint(tiny_config):
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    index = json.loads((root / "upstream/hf/model.safetensors.index.json").read_text())
    # Use only layer 0/1: tiny models intentionally have fewer layers.
    actual = dict(tree_flatten(Model(tiny_config).parameters()))
    mapped = {map_weight_key(k) for k in index["weight_map"]}
    assert set(actual) <= mapped
    assert len({map_weight_key(k) for k in index["weight_map"]}) == 569
