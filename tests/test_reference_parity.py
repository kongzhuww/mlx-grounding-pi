"""Execute pinned, hash-verified upstream classes with identical tiny FP32 weights.

Run with: TORCHDYNAMO_DISABLE=1 uv run --group reference pytest -q tests/test_reference_parity.py
This validates the actual upstream operators, but not pretrained checkpoint accuracy.
"""

import hashlib
import importlib
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_grounding_pi.convert import convert_tensor, map_weight_key
from mlx_grounding_pi.model import Model
from mlx_grounding_pi.projector import Projector
from mlx_grounding_pi.vision import VisionTower

ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch", reason="install the optional reference group")


@pytest.fixture(scope="module")
def reference():
    # Only the audited local snapshot is executable; no trust_remote_code or network calls.
    manifest = json.loads((ROOT / "upstream/manifest.json").read_text())
    for item in manifest["files"]:
        if item["path"].endswith(".py"):
            assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    package = types.ModuleType("groundingpi_reference")
    package.__path__ = [str(ROOT / "upstream/hf")]
    sys.modules[package.__name__] = package
    module = importlib.import_module("groundingpi_reference.modeling_groundingpi")
    vision = importlib.import_module("groundingpi_reference.modeling_groundingpi_vision")
    return module, vision


def transfer(torch_module, mlx_module, full=False):
    weights = []
    for key, value in torch_module.state_dict().items():
        source_key = key if full else "model.visual.vision_tower." + key
        tensor = mx.array(value.detach().cpu().float().numpy())
        tensor = convert_tensor(source_key, tensor)
        weights.append((map_weight_key(key) if full else key, tensor))
    mlx_module.load_weights(weights, strict=True)


def close(actual, expected, label):
    a = np.array(actual)
    b = expected.detach().cpu().numpy()
    error = float(np.abs(a - b).max())
    print(f"{label}: max_abs_error={error:.9g}")
    np.testing.assert_allclose(a, b, atol=5e-6, rtol=5e-5, err_msg=label)


@pytest.mark.parametrize("grids", [[(1, 4, 4)], [(1, 2, 4), (1, 2, 2)]])
def test_official_vision_every_stage(reference, tiny_vision, grids, monkeypatch):
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    module, vision = reference
    cfg = module.GroundingPIVisionConfig(**asdict(tiny_vision))
    cfg._attn_implementation = "eager"
    torch.manual_seed(11)
    original = vision.MoonViT3dPretrainedModel(cfg).eval()
    port = VisionTower(tiny_vision)
    transfer(original, port)
    count = sum(t * h * w for t, h, w in grids)
    patches = np.random.default_rng(3).normal(size=(count, 3, 2, 2)).astype(np.float32)
    tg, tx = torch.tensor(grids), torch.from_numpy(patches)
    with torch.inference_mode():
        expected = original.patch_embed(tx, tg)
        actual = port.patch_embed(mx.array(patches), grids)
        close(actual, expected, "patch convolution + learned interpolated positions")
        tfreq = original.encoder.rope_2d.get_freqs_cis(tg, torch.device("cpu"))
        from mlx_grounding_pi.vision import rotary_angles

        angles = rotary_angles(grids, 8)
        lengths = [t * h * w for t, h, w in grids]
        cumulative = torch.tensor([0] + list(np.cumsum(lengths)), dtype=torch.int32)
        for i, (tb, mb) in enumerate(zip(original.encoder.blocks, port.encoder.blocks)):
            expected = tb(expected, cumulative, max(lengths), tfreq)
            actual = mb(actual, grids, angles)
            close(actual, expected, f"vision block {i}")
        expected = original.encoder.final_layernorm(expected)
        actual = port.encoder.final_layernorm(actual)
        close(actual, expected, "vision final RMSNorm")
        expected_merged = vision.tpool_patch_merger(expected, tg)
        from mlx_grounding_pi.vision import spatial_merge

        for a, b in zip(spatial_merge(actual, grids), expected_merged):
            close(a, b, "2x2 spatial merge")


@pytest.mark.parametrize("mode,size", [("RGB", (29, 28)), ("RGBA", (120, 73)), ("L", (1, 1))])
def test_official_preprocessor(reference, mode, size):
    from PIL import Image

    from mlx_grounding_pi.processing import ImageProcessor

    original_module = importlib.import_module("groundingpi_reference.image_processing_groundingpi")
    cfg = json.loads((ROOT / "upstream/hf/preprocessor_config.json").read_text())["media_proc_cfg"]
    original = original_module.GroundingPIImageProcessor(cfg)
    port = ImageProcessor(cfg)
    bands = len(Image.new(mode, (1, 1)).getbands())
    shape = (size[1], size[0], bands) if bands > 1 else (size[1], size[0])
    pixels = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8)
    image = Image.fromarray(pixels)
    expected = original(images=[image], return_tensors="pt")
    actual = port(image)
    np.testing.assert_array_equal(actual.pixel_values, expected["pixel_values"].numpy())
    np.testing.assert_array_equal(actual.image_grid_thw, expected["image_grid_thw"].numpy())


def test_official_saved_tokenizer_and_chat(reference):
    from PIL import Image

    from mlx_grounding_pi.processing import Processor

    path = ROOT / "models/groundingpi-hf"
    if not (path / "tokenizer.json").is_file():
        pytest.skip("download the pinned tokenizer/config assets first")
    port = Processor(path)
    original_images = importlib.import_module("groundingpi_reference.image_processing_groundingpi")
    original_processor = importlib.import_module("groundingpi_reference.processing_groundingpi")
    cfg = json.loads((path / "preprocessor_config.json").read_text())["media_proc_cfg"]
    original = original_processor.GroundingPIProcessor(
        image_processor=original_images.GroundingPIImageProcessor(cfg),
        tokenizer=port.tokenizer,
    )
    image = Image.new("RGB", (448, 448), (32, 64, 128))
    instruction = "Point to the target referred to by the following description: the red block."
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
    ]
    text = original.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    expected = original(text=text, images=[image], return_tensors="pt")
    actual = port(image, instruction)
    np.testing.assert_array_equal(actual["input_ids"], expected["input_ids"].numpy())
    np.testing.assert_array_equal(actual["pixel_values"], expected["pixel_values"].numpy())
    assert port.tokenizer.encode("<0><999>", add_special_tokens=False) == [151669, 152668]


def test_official_projector(reference, tiny_vision):
    module, _ = reference
    cfg = module.GroundingPIVisionConfig(**asdict(tiny_vision))
    torch.manual_seed(12)
    original = module.GroundingPITwoLayerProjector(cfg).eval()
    port = Projector(tiny_vision)
    port.load_weights([(k, mx.array(v.detach().numpy())) for k, v in original.state_dict().items()])
    x = np.random.default_rng(4).normal(size=(5, 4, 8)).astype(np.float32)
    with torch.inference_mode():
        close(port([mx.array(x)])[0], original([torch.from_numpy(x)])[0], "projector")


def test_official_cached_generation_and_export_hooks(reference, tiny_config, monkeypatch):
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    from mlx_grounding_pi.generate import greedy_generate
    from mlx_grounding_pi.reference import capture_visual, generate_torch

    module, _ = reference
    cfg = module.GroundingPIConfig(**tiny_config)
    cfg._attn_implementation = "eager"
    torch.manual_seed(22)
    original = module.GroundingPIForConditionalGeneration(cfg).eval()
    port = Model(tiny_config)
    transfer(original, port, full=True)
    pixels = np.random.default_rng(6).normal(size=(16, 3, 2, 2)).astype(np.float32)
    ids = [[1, 31, 31, 31, 31, 2]]
    torch_inputs = {
        "input_ids": torch.tensor(ids),
        "pixel_values": torch.from_numpy(pixels),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    captured, handles = capture_visual(original)
    try:
        expected, finish, _, logits = generate_torch(
            original, torch_inputs, device="cpu", eos_token_id=63, max_tokens=4
        )
    finally:
        for handle in handles:
            handle.remove()
    mlx_inputs = {"input_ids": np.array(ids), "pixel_values": pixels, "image_grid_thw": [(1, 4, 4)]}
    actual, actual_finish, _ = greedy_generate(port, mlx_inputs, eos_token_id=63, max_tokens=4)
    assert actual == expected and finish == actual_finish
    assert logits.shape == (64,)
    assert captured["patch_embeddings"].shape == (16, 8)
    assert captured["vision_final"].shape == (16, 8)
    assert captured["projected_features"].shape == (4, 12)


def test_official_full_multimodal_logits(reference, tiny_config, monkeypatch):
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    module, _ = reference
    cfg = module.GroundingPIConfig(**tiny_config)
    cfg._attn_implementation = "eager"
    torch.manual_seed(13)
    original = module.GroundingPIForConditionalGeneration(cfg).eval()
    port = Model(tiny_config)
    transfer(original, port, full=True)
    patches = np.random.default_rng(5).normal(size=(16, 3, 2, 2)).astype(np.float32)
    ids = [[1, 31, 31, 31, 31, 2, 3]]
    with torch.inference_mode():
        expected = original(
            input_ids=torch.tensor(ids),
            pixel_values=torch.from_numpy(patches),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            use_cache=False,
        ).logits
        actual = port(mx.array(ids), pixel_values=mx.array(patches), image_grid_thw=[(1, 4, 4)])
        close(actual, expected, "full multimodal logits")
