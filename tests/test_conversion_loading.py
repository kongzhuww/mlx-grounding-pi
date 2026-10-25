import hashlib
import json
from dataclasses import asdict, replace

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten

from mlx_grounding_pi.checkpoint import expected_shapes
from mlx_grounding_pi.convert import convert_checkpoint, map_weight_key
from mlx_grounding_pi.loading import load_model
from mlx_grounding_pi.model import Model


def make_source(tmp_path, tiny_config, tiny_vision):
    config = {**tiny_config, "model_type": "groundingpi", "eos_token_id": 63}
    config["vision_config"] = asdict(replace(tiny_vision, out_hidden_size=32))
    config["text_config"] = {**config["text_config"], "hidden_size": 32, "intermediate_size": 64}
    model = Model(config)
    tensors = dict(tree_flatten(model.parameters()))
    source_tensors = {}
    for key in expected_shapes(config):
        value = tensors[map_weight_key(key)].astype(mx.bfloat16)
        if key.endswith("patch_embed.proj.weight"):
            value = value.transpose(0, 3, 1, 2)
        source_tensors[key] = value
    source = tmp_path / "source"
    source.mkdir()
    mx.save_safetensors(str(source / "original.safetensors"), source_tensors)
    size = sum(v.nbytes for v in source_tensors.values())
    params = sum(v.size for v in source_tensors.values())
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_parameters": params, "total_size": size},
                "weight_map": {k: "original.safetensors" for k in source_tensors},
            }
        )
    )
    (source / "config.json").write_text(json.dumps(config))
    model.load_weights(
        [
            (
                map_weight_key(k),
                v.transpose(0, 2, 3, 1) if k.endswith("patch_embed.proj.weight") else v,
            )
            for k, v in source_tensors.items()
        ]
    )
    return source, model


@pytest.mark.parametrize("bits", [0, 4, 8])
def test_stream_conversion_and_strict_reload(tmp_path, tiny_config, tiny_vision, bits):
    source, expected = make_source(tmp_path, tiny_config, tiny_vision)
    output = tmp_path / "converted"
    config = convert_checkpoint(source, output, bits=bits, group_size=32, shard_size_mb=1)
    actual, _ = load_model(output)
    quant = config["mlx_grounding_pi"]["quantization"]
    if bits:
        assert quant["modules"]
        assert all(k.startswith("language_model.") for k in quant["modules"])
        nn.quantize(
            expected, bits=bits, group_size=32, class_predicate=lambda p, m: p in quant["modules"]
        )
    ids = mx.array([[1, 31, 31, 31, 31, 2]])
    patches = mx.random.normal((16, 3, 2, 2)).astype(mx.bfloat16)
    kwargs = {"pixel_values": patches, "image_grid_thw": [(1, 4, 4)]}
    np.testing.assert_allclose(
        np.array(actual(ids, **kwargs).astype(mx.float32)),
        np.array(expected(ids, **kwargs).astype(mx.float32)),
        atol=1e-6,
    )
    with pytest.raises(FileExistsError):
        convert_checkpoint(source, output)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    index["weight_map"].pop(next(iter(index["weight_map"])))
    (output / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="tensor names"):
        load_model(output)


def test_source_sha256_mismatch_is_rejected(tmp_path, tiny_config, tiny_vision):
    source, _ = make_source(tmp_path, tiny_config, tiny_vision)
    (source / "checksums.sha256").write_text("0" * 64 + "  original.safetensors\n")
    with pytest.raises(ValueError, match="SHA-256"):
        convert_checkpoint(source, tmp_path / "bad-checksum")
    assert not (tmp_path / "bad-checksum").exists()


def test_multiple_output_shards_and_checksum_verification(tmp_path, tiny_config, tiny_vision):
    config = {**tiny_config, "text_config": {**tiny_config["text_config"], "vocab_size": 20000}}
    source, _ = make_source(tmp_path, config, tiny_vision)
    with (source / "original.safetensors").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    (source / "checksums.sha256").write_text(digest + "  original.safetensors\n")
    output = tmp_path / "multi"
    convert_checkpoint(source, output, shard_size_mb=1)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert len(set(index["weight_map"].values())) >= 2
    assert json.loads((output / "conversion.json").read_text())["published_checksums_verified"]
    load_model(output)


def test_incomplete_original_is_rejected_before_output(tmp_path, tiny_config, tiny_vision):
    source, _ = make_source(tmp_path, tiny_config, tiny_vision)
    shard = source / "original.safetensors"
    with shard.open("r+b") as f:
        f.truncate(shard.stat().st_size - 2)
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        convert_checkpoint(source, output)
    assert not output.exists()
