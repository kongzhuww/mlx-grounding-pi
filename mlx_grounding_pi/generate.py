"""Local single-image grounding with a converted MLX GroundingPI checkpoint."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from PIL import Image, ImageDraw

from ._vendor.protocol import build_refer_bbox_prompt, build_refer_point_prompt
from .coordinates import FormatError, parse_answer, to_pixels
from .loading import load_model
from .processing import Processor


def greedy_generate(model, inputs, *, eos_token_id, max_tokens=256, max_context=4096):
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    ids = mx.array(inputs["input_ids"])
    if ids.shape[1] + max_tokens > max_context:
        raise ValueError("prompt plus generation budget exceeds max_context")
    eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
    if not eos:
        raise ValueError("checkpoint must specify EOS token ID(s)")
    cache = make_prompt_cache(model)
    generated = []
    start = time.perf_counter()
    prefill_seconds = None
    for step in range(max_tokens):
        kwargs = (
            {
                "pixel_values": mx.array(inputs["pixel_values"]),
                "image_grid_thw": inputs["image_grid_thw"],
            }
            if step == 0
            else {}
        )
        logits = model(ids, cache=cache, logits_to_keep=1, **kwargs)
        token = int(mx.argmax(logits[0, -1]).item())
        if step == 0:
            prefill_seconds = time.perf_counter() - start
        if token in eos:
            return (
                generated,
                "eos",
                {
                    "elapsed_seconds": time.perf_counter() - start,
                    "prefill_seconds": prefill_seconds,
                },
            )
        generated.append(token)
        ids = mx.array([[token]])
    return (
        generated,
        "length",
        {"elapsed_seconds": time.perf_counter() - start, "prefill_seconds": prefill_seconds},
    )


def parse_result(text, task, size, finish_reason):
    result = {
        "task": task,
        "raw_output": text,
        "finish_reason": finish_reason,
        "valid": False,
        "validation_scope": "protocol_only",
        "predictions": [],
        "parse_error": None,
    }
    if finish_reason != "eos":
        result["parse_error"] = "generation reached its token limit; output may be truncated"
        return result
    try:
        entries = parse_answer(text, task, require_canonical=True)
        result["predictions"] = [
            {
                "label": e.phrase,
                "coordinates": e.coordinates,
                "pixels": ([to_pixels(c, *size) for c in e.coordinates] if e.coordinates else None),
            }
            for e in entries
        ]
        result["valid"] = True
    except FormatError as exc:
        result["parse_error"] = str(exc)
    return result


def save_result(output, image, result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if not result["valid"]:
        return
    with Image.open(image) as original:
        canvas = original.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for prediction in result["predictions"]:
        for pixels in prediction["pixels"] or []:
            if len(pixels) == 4:
                draw.rectangle(pixels, outline="red", width=3)
            else:
                x, y = pixels
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="red", width=3)
    canvas.save(output / "prediction.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, type=Path, help="local converted MLX checkpoint directory"
    )
    parser.add_argument("--image", required=True, type=Path)
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument(
        "--phrase", help="target description, wrapped in the official referring prompt"
    )
    text.add_argument("--prompt", help="complete custom instruction")
    parser.add_argument("--task", choices=["point", "bbox"], default="point")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument(
        "--max-image-tokens",
        type=int,
        default=None,
        help="optional lower image budget; may reduce grounding accuracy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="new output directory; existing directories are never overwritten",
    )
    parser.add_argument(
        "--allow-large-model",
        action="store_true",
        help="override conservative weight-memory preflight (may cause memory pressure)",
    )
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("output directory already exists")
    if not args.image.is_file():
        parser.error("image does not exist")
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    recommended = mx.device_info().get("max_recommended_working_set_size", 0)
    if (
        recommended
        and index["metadata"]["total_size"] > recommended * 0.75
        and not args.allow_large_model
    ):
        parser.error(
            "weights alone exceed 75% of recommended GPU working set; convert language weights to 4-bit"
        )
    processor = Processor(args.model, max_output_tokens=args.max_image_tokens)
    instruction = args.prompt or (
        build_refer_point_prompt(args.phrase)
        if args.task == "point"
        else build_refer_bbox_prompt(args.phrase)
    )
    inputs = processor(args.image, instruction)
    model, config = load_model(args.model)
    context_limit = min(args.max_context, config["text_config"]["max_position_embeddings"])
    tokens, finish, timing = greedy_generate(
        model,
        inputs,
        eos_token_id=config["eos_token_id"],
        max_tokens=args.max_tokens,
        max_context=context_limit,
    )
    raw = processor.tokenizer.decode(
        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    result = parse_result(raw, args.task, inputs["image"].original_size, finish)
    result.update(
        {
            "generated_tokens": len(tokens),
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "image_tokens": inputs["image"].num_tokens,
            "timing": timing,
            "peak_mlx_memory_bytes": mx.get_peak_memory(),
        }
    )
    if args.output:
        save_result(args.output, args.image, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
