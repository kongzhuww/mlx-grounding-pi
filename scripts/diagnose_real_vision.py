"""Local CPU-PyTorch vs GPU-MLX real-weight vision diagnostic; no LLM/cloud calls.

FP32 isolates architecture/layout errors from BF16 operator rounding. The optional
CPU BF16 path is diagnostic, not a replacement for the returned CUDA BF16 baseline.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import mlx.core as mx
import numpy as np
import torch
from mlx import nn
from safetensors import safe_open

from mlx_grounding_pi.config import VisionConfig, load_config
from mlx_grounding_pi.convert import convert_tensor
from mlx_grounding_pi.model import Visual
from mlx_grounding_pi.parity import numeric_difference
from mlx_grounding_pi.reference import load_reference_classes
from mlx_grounding_pi.vision import rotary_angles, spatial_merge

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--layers", type=int, default=27)
    parser.add_argument(
        "--case", choices=["red-center", "photo-left-cat-box"], default="red-center"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    config = load_config(ROOT / "models/groundingpi-hf/config.json")
    original, _, _ = load_reference_classes(ROOT)
    cfg = original.GroundingPIVisionConfig(**config["vision_config"])
    cfg._attn_implementation = "eager"
    tdtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    mdtype = mx.float32 if args.dtype == "fp32" else mx.bfloat16
    reference = original.GroundingPIVisionModel(cfg).eval().to(dtype=tdtype)
    port = Visual(VisionConfig.from_dict(config["vision_config"]))
    index = json.loads((ROOT / "models/groundingpi-hf/model.safetensors.index.json").read_text())
    state, weights = {}, []
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(ROOT / "models/groundingpi-hf" / shard, framework="pt", device="cpu") as f:
            keys = f.keys()
            for key in sorted(k for k in keys if k.startswith("model.visual.")):
                tensor = f.get_tensor(key)
                name = key.removeprefix("model.visual.")
                state[name] = tensor
                value = mx.array(tensor.float().numpy()).astype(mdtype)
                weights.append((name, convert_tensor(key, value)))
    reference.load_state_dict(state, strict=True)
    port.load_weights(weights, strict=True)
    mx.eval(port.parameters())
    del state, weights
    arrays = np.load(ROOT / "outputs/reference-bf16" / f"{args.case}.npz", allow_pickle=False)
    pixels = arrays["pixel_values"]
    grids = arrays["image_grid_thw"].tolist()
    tg = torch.tensor(grids)
    tx = torch.from_numpy(pixels).to(dtype=tdtype)
    x = mx.array(pixels).astype(mdtype)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "dtype": args.dtype,
        "case": args.case,
        "torch_device": "cpu",
        "mlx_device": str(mx.default_device()),
        "note": "Real vision weights only; not full-model generation parity or CUDA BF16 parity.",
        "stages": [],
        "primitives": [],
        "cuda_baseline_comparison": {},
    }
    started = time.perf_counter()

    def record(name, actual, expected, *, group="stages"):
        metrics = numeric_difference(
            expected.detach().float().numpy(), np.array(actual.astype(mx.float32))
        )
        report[group].append({"name": name, **metrics})
        report["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            name,
            "relative_L2",
            metrics["relative_l2_error"],
            "max_abs",
            metrics["max_abs_error"],
            flush=True,
        )

    with torch.inference_mode():
        expected = reference.vision_tower.patch_embed(tx, tg)
        actual = port.vision_tower.patch_embed(x, grids)
        record("patch_embeddings", actual, expected)
        # Common inputs isolate individual operators from accumulated drift.
        tb = reference.vision_tower.encoder.blocks[0]
        mb = port.vision_tower.encoder.blocks[0]
        common = mx.array(expected.float().numpy()).astype(mdtype)
        tn = tb.norm0(expected)
        record("norm0_common_input", mb.norm0(common), tn, group="primitives")
        cf = common.astype(mx.float32)
        normalized = cf * mx.rsqrt(mx.mean(cf * cf, axis=-1, keepdims=True) + 1e-6)
        rounded = normalized.astype(mdtype) * mb.norm0.weight
        record("norm0_two_roundings", rounded, tn, group="primitives")
        common_norm = mx.array(tn.float().numpy()).astype(mdtype)
        record("wqkv_common_input", mb.wqkv(common_norm), tb.wqkv(tn), group="primitives")
        tfc = tb.mlp.fc0(tb.norm1(expected))
        mfc = mx.array(tfc.float().numpy()).astype(mdtype)
        record("gelu_common_input", nn.gelu_approx(mfc), tb.mlp.activation(tfc), group="primitives")
        tfreq = reference.vision_tower.encoder.rope_2d.get_freqs_cis(tg, torch.device("cpu"))
        angles = rotary_angles(grids, cfg.qkv_hidden_size // cfg.num_attention_heads)
        length = pixels.shape[0]
        cumulative = torch.tensor([0, length], dtype=torch.int32)
        for i, (tb, mb) in enumerate(
            zip(reference.vision_tower.encoder.blocks, port.vision_tower.encoder.blocks)
        ):
            if i >= args.layers:
                break
            expected = tb(expected, cumulative, length, tfreq)
            actual = mb(actual, grids, angles)
            record(f"block_{i:02d}", actual, expected)
        if args.layers >= cfg.num_hidden_layers:
            expected = reference.vision_tower.encoder.final_layernorm(expected)
            actual = port.vision_tower.encoder.final_layernorm(actual)
            record("final_norm", actual, expected)
            if args.dtype == "bf16":
                report["cuda_baseline_comparison"]["vision_final"] = {
                    "cpu_torch": numeric_difference(
                        arrays["vision_final"], expected.float().numpy()
                    ),
                    "mlx": numeric_difference(
                        arrays["vision_final"], np.array(actual.astype(mx.float32))
                    ),
                }
            from groundingpi_reference.modeling_groundingpi_vision import tpool_patch_merger

            expected = reference.projector(tpool_patch_merger(expected, tg))[0]
            actual = port.projector(spatial_merge(actual, grids))[0]
            if args.dtype == "bf16":
                report["cuda_baseline_comparison"]["projected_features"] = {
                    "cpu_torch": numeric_difference(
                        arrays["projected_features"], expected.float().numpy()
                    ),
                    "mlx": numeric_difference(
                        arrays["projected_features"], np.array(actual.astype(mx.float32))
                    ),
                }
            record("projected_features", actual, expected)


if __name__ == "__main__":
    main()
