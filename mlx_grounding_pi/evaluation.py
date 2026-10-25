"""Small, reproducible development evaluation; not a standard grounding benchmark."""

import argparse
import hashlib
import json
import math
import re
import time
from itertools import permutations
from pathlib import Path


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def assess(result, expected):
    """Keep protocol validity, exact synthetic scores and photo review separate."""
    if not result.get("valid"):
        return {"status": "invalid_output", "passed": False}
    if expected is None:
        return {"status": "requires_visual_review", "passed": None}
    predictions = result["predictions"]
    actual = [coords for prediction in predictions for coords in prediction["pixels"] or []]
    if expected.get("absent"):
        passed = bool(predictions) and all(p["coordinates"] is None for p in predictions)
        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "criterion": "absent_target",
        }
    key = "points" if "points" in expected else "boxes"
    truth = expected[key]
    if len(actual) != len(truth):
        return {
            "status": "failed",
            "passed": False,
            "criterion": "instance_count",
            "expected_count": len(truth),
            "actual_count": len(actual),
        }
    # Exhaustive assignment is only appropriate for this tiny smoke suite.
    if len(truth) > 6:
        raise ValueError("smoke matching is limited to six instances")
    if key == "points":
        if any(len(a) != 2 for a in actual):
            return {"status": "failed", "passed": False, "criterion": "point_arity"}
        error = min(
            max(math.dist(a, b) for a, b in zip(order, truth)) for order in permutations(actual)
        )
        passed = error <= expected["max_distance_pixels"]
        return {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "max_point_error_pixels": error,
        }
    if any(len(a) != 4 for a in actual):
        return {"status": "failed", "passed": False, "criterion": "box_arity"}
    iou = max(min(box_iou(a, b) for a, b in zip(order, truth)) for order in permutations(actual))
    passed = iou >= expected["min_iou"]
    return {"status": "passed" if passed else "failed", "passed": passed, "min_matched_iou": iou}


def instruction_for(case):
    from ._vendor.protocol import build_refer_bbox_prompt, build_refer_point_prompt

    if ("phrase" in case) == ("prompt" in case):
        raise ValueError("each case must contain exactly one of phrase/prompt")
    if case["task"] not in {"point", "bbox"}:
        raise ValueError("unsupported evaluation task")
    if "prompt" in case:
        return case["prompt"]
    builder = build_refer_point_prompt if case["task"] == "point" else build_refer_bbox_prompt
    return builder(case["phrase"])


def evaluate(model_path, manifest_path, output, *, max_image_tokens=None, max_tokens=256):
    import mlx.core as mx

    from .generate import greedy_generate, parse_result, save_result
    from .loading import load_model
    from .processing import Processor

    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    names = [c["id"] for c in cases]
    if len(set(names)) != len(names) or any(
        not re.fullmatch(r"[a-z0-9-]+", name) for name in names
    ):
        raise ValueError("case IDs must be unique safe directory names")
    for asset in manifest.get("assets", []):
        path = manifest_path.parent / asset["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"fixture digest mismatch: {path}")
    for case in cases:
        instruction_for(case)
        if not (manifest_path.parent / case["image"]).is_file():
            raise FileNotFoundError(case["image"])
    output.mkdir(parents=True, exist_ok=False)
    load_start = time.perf_counter()
    model, config = load_model(model_path)
    processor = Processor(model_path, max_output_tokens=max_image_tokens)
    load_seconds = time.perf_counter() - load_start
    rows = []
    for case in cases:
        image = (manifest_path.parent / case["image"]).resolve()
        instruction = instruction_for(case)
        mx.reset_peak_memory()
        try:
            inputs = processor(image, instruction)
            tokens, finish, timing = greedy_generate(
                model,
                inputs,
                eos_token_id=config["eos_token_id"],
                max_tokens=max_tokens,
                max_context=4096,
            )
            raw = processor.tokenizer.decode(
                tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            result = parse_result(raw, case["task"], inputs["image"].original_size, finish)
            result.update(
                generated_tokens=len(tokens),
                prompt_tokens=int(inputs["input_ids"].shape[1]),
                image_tokens=inputs["image"].num_tokens,
                timing=timing,
                peak_mlx_memory_bytes=mx.get_peak_memory(),
            )
            save_result(output / case["id"], image, result)
            row = {
                "id": case["id"],
                "task": case["task"],
                "instruction": instruction,
                "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "assessment": assess(result, case.get("expected")),
                "result": result,
            }
        except (ValueError, RuntimeError, MemoryError, OSError, TypeError) as exc:
            row = {
                "id": case["id"],
                "assessment": {"status": "inference_error", "passed": False},
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(f"{case['id']}: {row['assessment']}", flush=True)
        report = {
            "suite": manifest["description"],
            "model_path": str(Path(model_path).resolve()),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "load_model_and_processor_seconds": load_seconds,
            "max_image_tokens": max_image_tokens,
            "quantization": config["mlx_grounding_pi"]["quantization"],
            "cases": rows,
        }
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        mx.clear_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-image-tokens", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    report = evaluate(
        args.model,
        args.cases,
        args.output,
        max_image_tokens=args.max_image_tokens,
        max_tokens=args.max_tokens,
    )
    if any(r["assessment"]["passed"] is False for r in report["cases"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
