"""Export MLX artifacts / compare with official PyTorch, without cloud automation."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .evaluation import instruction_for


def numeric_difference(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError(f"tensor shape mismatch: {reference.shape} != {actual.shape}")
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    delta = a - b
    denominator = float(np.linalg.norm(a.reshape(-1)) * np.linalg.norm(b.reshape(-1)))
    return {
        "shape": list(a.shape),
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
        "relative_l2_error": float(
            np.linalg.norm(delta.reshape(-1)) / max(np.linalg.norm(a.reshape(-1)), 1e-12)
        ),
        "cosine_similarity": float(np.sum(a * b) / denominator) if denominator else None,
    }


def export_mlx(
    model_path,
    cases_path,
    output,
    *,
    capture_cases=("red-center", "absent-triangle", "photo-left-cat-box"),
):
    import mlx.core as mx

    from .generate import greedy_generate, parse_result
    from .loading import load_model
    from .processing import Processor
    from .vision import spatial_merge

    cases_path, output = Path(cases_path).resolve(), Path(output).resolve()
    manifest = json.loads(cases_path.read_text())
    if output.exists():
        raise FileExistsError(output)
    for asset in manifest.get("assets", []):
        if (
            hashlib.sha256((cases_path.parent / asset["path"]).read_bytes()).hexdigest()
            != asset["sha256"]
        ):
            raise ValueError("fixture hash mismatch")
    model, config = load_model(model_path)
    processor = Processor(model_path)
    output.mkdir(parents=True, exist_ok=False)
    rows = []

    class CaptureFirstLogits:
        @property
        def layers(self):
            return model.layers

        def __init__(self):
            self.logits = None

        def __call__(self, *args, **kwargs):
            value = model(*args, **kwargs)
            if self.logits is None:
                self.logits = np.array(value[0, -1].astype(mx.float32))
            return value

    for case in manifest["cases"]:
        image = cases_path.parent / case["image"]
        inputs = processor(image, instruction_for(case))
        capture = CaptureFirstLogits()
        ids, finish, timing = greedy_generate(capture, inputs, eos_token_id=config["eos_token_id"])
        arrays = {
            "input_ids": inputs["input_ids"],
            "image_grid_thw": inputs["image_grid_thw"],
            "first_logits": capture.logits,
            "generated_ids": np.array(ids, dtype=np.int64),
        }
        if case["id"] in capture_cases:
            rows_grid = inputs["image_grid_thw"].tolist()
            patches = mx.array(inputs["pixel_values"]).astype(
                model.visual.vision_tower.patch_embed.proj.weight.dtype
            )
            x = model.visual.vision_tower.patch_embed(patches, rows_grid)
            arrays["patch_embeddings"] = np.array(x.astype(mx.float32))
            x = model.visual.vision_tower.encoder(x, rows_grid)
            arrays["vision_final"] = np.array(x.astype(mx.float32))
            x = model.visual.projector(spatial_merge(x, rows_grid))
            arrays["projected_features"] = np.array(mx.concatenate(x).astype(mx.float32))
            arrays["pixel_values"] = inputs["pixel_values"]
        np.savez(output / (case["id"] + ".npz"), **arrays)
        raw = processor.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        rows.append(
            {
                "id": case["id"],
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "instruction": instruction_for(case),
                "result": parse_result(raw, case["task"], inputs["image"].original_size, finish),
                "timing_includes_capture_overhead": timing,
            }
        )
        report = {
            "backend": "mlx",
            "model": str(Path(model_path).resolve()),
            "manifest_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "quantization": config["mlx_grounding_pi"]["quantization"],
            "cases": rows,
        }
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"Exported MLX artifacts: {case['id']}", flush=True)
        mx.clear_cache()


def compare(reference_dir, mlx_dir, output):
    reference_dir, mlx_dir, output = map(Path, (reference_dir, mlx_dir, output))
    reference = json.loads((reference_dir / "report.json").read_text())
    mlx = json.loads((mlx_dir / "report.json").read_text())
    if reference["manifest_sha256"] != mlx["manifest_sha256"]:
        raise ValueError("case manifests differ; comparison is not valid")
    left = {case["id"]: case for case in reference["cases"]}
    right = {case["id"]: case for case in mlx["cases"]}
    if set(left) != set(right):
        raise ValueError("case coverage differs")
    if any(not re.fullmatch(r"[a-z0-9-]+", name) for name in left):
        raise ValueError("unsafe artifact case ID")
    rows = []
    for name in left:
        if (
            left[name]["image_sha256"] != right[name]["image_sha256"]
            or left[name]["instruction"] != right[name]["instruction"]
        ):
            raise ValueError(f"image or instruction differs: {name}")
        with (
            np.load(reference_dir / (name + ".npz"), allow_pickle=False) as a,
            np.load(mlx_dir / (name + ".npz"), allow_pickle=False) as b,
        ):
            for key in ["input_ids", "image_grid_thw"]:
                if not np.array_equal(a[key], b[key]):
                    raise ValueError(f"model inputs differ: {name}/{key}")
            stages = {
                key: numeric_difference(a[key], b[key])
                for key in [
                    "pixel_values",
                    "patch_embeddings",
                    "vision_final",
                    "projected_features",
                    "first_logits",
                ]
                if key in a and key in b
            }
            first_a, first_b = a["first_logits"], b["first_logits"]
            rows.append(
                {
                    "id": name,
                    "stages": stages,
                    "first_argmax_match": int(first_a.argmax()) == int(first_b.argmax()),
                    "top10_overlap": len(
                        set(first_a.argsort()[-10:]) & set(first_b.argsort()[-10:])
                    ),
                    "generated_ids_equal": np.array_equal(a["generated_ids"], b["generated_ids"]),
                    "reference_output": left[name]["result"]["raw_output"],
                    "mlx_output": right[name]["result"]["raw_output"],
                }
            )
    report = {
        "note": "Measured differences only; no invented pass threshold. Quantized LLM outputs need not match BF16 exactly.",
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-mlx")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--cases", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--mlx", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export-mlx":
        export_mlx(args.model, args.cases, args.output)
    else:
        compare(args.reference, args.mlx, args.output)


if __name__ == "__main__":
    main()
