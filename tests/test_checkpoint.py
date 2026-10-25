import copy
import json
from pathlib import Path

import pytest
from mlx.utils import tree_flatten

from mlx_grounding_pi.checkpoint import expected_shapes, validate_header, verify_schema
from mlx_grounding_pi.config import load_config
from mlx_grounding_pi.convert import map_weight_key
from mlx_grounding_pi.model import Model

ROOT = Path(__file__).resolve().parents[1]


def test_real_checkpoint_headers_match_every_tensor():
    index = json.loads((ROOT / "upstream/hf/model.safetensors.index.json").read_text())
    headers = {
        shard: json.loads((ROOT / "upstream/headers" / (shard + ".json")).read_text())
        for shard in set(index["weight_map"].values())
    }
    result = verify_schema(load_config(ROOT / "upstream/hf/config.json"), index, headers)
    assert result == {
        "verified_tensors": 569,
        "parameters": 4843664384,
        "payload_bytes": 9687328768,
    }
    broken = copy.deepcopy(headers)
    first = next(iter(broken.values()))["tensors"]
    key = next(k for k in first if k != "__metadata__")
    first[key]["shape"][0] += 1
    with pytest.raises(ValueError):
        verify_schema(load_config(ROOT / "upstream/hf/config.json"), index, broken)


def test_schema_matches_allocated_tiny_model(tiny_config):
    actual = {k: tuple(v.shape) for k, v in tree_flatten(Model(tiny_config).parameters())}
    expected = {}
    for k, shape in expected_shapes(tiny_config).items():
        if k == "model.visual.vision_tower.patch_embed.proj.weight":
            shape = (shape[0], shape[2], shape[3], shape[1])
        expected[map_weight_key(k)] = shape
    assert actual == expected


def test_reject_header_gaps_and_payload_mismatch():
    header = {"x": {"shape": [2, 3], "dtype": "BF16", "data_offsets": [0, 12]}}
    validate_header(header, 12)
    for total, offsets in [(13, [0, 12]), (12, [1, 13]), (12, [0, 10])]:
        with pytest.raises(ValueError):
            validate_header({"x": {**header["x"], "data_offsets": offsets}}, total)
