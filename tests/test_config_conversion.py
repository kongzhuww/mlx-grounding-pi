import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mlx_grounding_pi.config import load_config
from mlx_grounding_pi.convert import convert_tensor, map_weight_key

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_sources_unchanged():
    manifest = json.loads((ROOT / "upstream/manifest.json").read_text())
    for record in manifest["files"]:
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    for name, original in [
        ("protocol.py", "github/grounding_pi/protocol.py"),
        ("media_utils.py", "hf/media_utils.py"),
    ]:
        assert (ROOT / "mlx_grounding_pi/_vendor" / name).read_bytes() == (
            ROOT / "upstream" / original
        ).read_bytes()


def test_real_config():
    cfg = load_config(ROOT / "upstream/hf/config.json")
    assert cfg["vision_config"]["qkv_hidden_size"] == 1536
    assert cfg["text_config"]["rope_parameters"]["rope_theta"] == 5000000


def test_reject_unsupported_config(tiny_vision):
    for changes in [
        {"norm_type": "layernorm"},
        {"qkv_hidden_size": 15},
        {"merge_kernel_size": (1, 2)},
        {"rope_theta": 5000000.0},
    ]:
        with pytest.raises(ValueError):
            replace(tiny_vision, **changes)


def test_key_and_conv_layout():
    key = "model.visual.vision_tower.patch_embed.proj.weight"
    x = np.arange(8 * 3 * 2 * 2).reshape(8, 3, 2, 2)
    assert map_weight_key(key) == "visual.vision_tower.patch_embed.proj.weight"
    np.testing.assert_array_equal(convert_tensor(key, x), x.transpose(0, 2, 3, 1))
    assert map_weight_key("model.language_model.norm.weight") == "language_model.model.norm.weight"
    assert map_weight_key("lm_head.weight") == "language_model.lm_head.weight"
    with pytest.raises(ValueError):
        map_weight_key("unknown.weight")
