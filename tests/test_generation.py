import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx_grounding_pi.generate import greedy_generate, parse_result, save_result
from mlx_grounding_pi.model import Model


class FixedTokens:
    def __init__(self, tokens):
        self.layers = [None]
        self.tokens = iter(tokens)
        self.calls = []

    def __call__(self, ids, **kwargs):
        self.calls.append(kwargs)
        token = next(self.tokens)
        return mx.where(mx.arange(64) == token, 1.0, 0.0).reshape(1, 1, 64)


def test_greedy_eos_and_images_only_on_prefill():
    model = FixedTokens([10, 20, 63])
    inputs = {
        "input_ids": np.array([[1, 31, 2]]),
        "pixel_values": np.zeros((4, 3, 2, 2)),
        "image_grid_thw": [(1, 2, 2)],
    }
    ids, reason, timing = greedy_generate(model, inputs, eos_token_id=63, max_tokens=5)
    assert ids == [10, 20] and reason == "eos"
    assert "pixel_values" in model.calls[0] and "pixel_values" not in model.calls[1]
    assert timing["elapsed_seconds"] >= timing["prefill_seconds"] >= 0
    with pytest.raises(ValueError, match="max_context"):
        greedy_generate(model, inputs, eos_token_id=63, max_tokens=10, max_context=5)


def test_truncated_and_malformed_outputs_never_valid(tmp_path):
    raw = "<|object_ref_start|>red block<|object_ref_end|><|box_start|><0><999><|box_end|>"
    result = parse_result(raw, "point", (640, 480), "eos")
    assert result["valid"] and result["predictions"][0]["pixels"] == [(0, 479)]
    assert not parse_result(raw, "point", (640, 480), "length")["valid"]
    assert not parse_result(raw + "junk", "point", (640, 480), "eos")["valid"]
    image = tmp_path / "image.png"
    Image.new("RGB", (640, 480)).save(image)
    save_result(tmp_path / "output", image, result)
    assert (tmp_path / "output/prediction.png").is_file()
    with pytest.raises(FileExistsError):
        save_result(tmp_path / "output", image, result)


def test_last_logits_optimization(tiny_config):
    model = Model(tiny_config)
    ids = mx.array([[1, 2, 3, 4]])
    np.testing.assert_allclose(
        np.array(model(ids)[:, -1:]), np.array(model(ids, logits_to_keep=1)), atol=1e-6
    )
