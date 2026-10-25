import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mlx_grounding_pi.processing import ImageProcessor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def processor():
    return ImageProcessor.from_config(ROOT / "upstream/hf/preprocessor_config.json")


@pytest.mark.parametrize("size", [(448, 448), (640, 480), (4096, 3072), (1, 1), (10000, 10)])
def test_dynamic_resize_budget(processor, size):
    cfg = processor.resize_config(*size)
    assert cfg["new_width"] <= size[0] and cfg["new_height"] <= size[1]
    assert (cfg["new_width"] + cfg["pad_width"]) % 28 == 0
    assert (cfg["new_height"] + cfg["pad_height"]) % 28 == 0
    assert cfg["num_tokens"] <= 1024
    if size == (448, 448):
        assert cfg["num_tokens"] == 256


def test_patch_order_normalization_and_padding(processor):
    image = np.zeros((28, 29, 3), dtype=np.uint8)
    image[:14, :14] = [255, 128, 0]
    result = processor(Image.fromarray(image))
    assert result.pixel_values.shape == (8, 3, 14, 14)
    assert result.image_grid_thw.tolist() == [[1, 2, 4]]
    assert result.padding == (27, 0)
    np.testing.assert_allclose(
        result.pixel_values[0, :, 0, 0], [1, 128 / 255 * 2 - 1, -1], atol=1e-7
    )
    assert np.all(result.pixel_values[1:] == -1)


def test_alpha_checkerboard_after_resize(processor):
    image = Image.new("RGBA", (28, 28), (255, 0, 0, 0))
    result = processor(image)
    expected = np.full((28, 28, 3), 255, dtype=np.uint8)
    for y in range(28):
        for x in range(28):
            if (y // 8 + x // 8) % 2:
                expected[y, x] = 180
    # Reassemble the row-major NCHW patches.
    actual = (
        result.pixel_values.reshape(2, 2, 3, 14, 14).transpose(0, 3, 1, 4, 2).reshape(28, 28, 3)
    )
    np.testing.assert_allclose(actual, expected.astype(np.float32) / 255 * 2 - 1, atol=2e-7)


def test_explicit_smaller_budget():
    cfg = json.loads((ROOT / "upstream/hf/preprocessor_config.json").read_text())["media_proc_cfg"]
    processor = ImageProcessor(cfg, max_output_tokens=64)
    assert processor.resize_config(1920, 1080)["num_tokens"] <= 64
    with pytest.raises(ValueError):
        ImageProcessor(cfg, max_output_tokens=0)
