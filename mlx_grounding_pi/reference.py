"""Official PyTorch reference export, portable to a user-authorized CUDA host.

No cloud SDK, deployment, login or automatic model download. Running this command
requires an existing original checkpoint and the repository's pinned source snapshot.
"""

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
import time
import types
from pathlib import Path

import numpy as np

from .evaluation import assess, instruction_for


def load_reference_classes(root):
    """Import only locally pinned upstream code after checking its recorded hashes."""
    root = Path(root)
    manifest = json.loads((root / "upstream/manifest.json").read_text())
    for item in manifest["files"]:
        if (
            item["path"].endswith(".py")
            and hashlib.sha256((root / item["path"]).read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError(f"reference source hash mismatch: {item['path']}")
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    package = types.ModuleType("groundingpi_reference")
    package.__path__ = [str(root / "upstream/hf")]
    sys.modules[package.__name__] = package
    model = importlib.import_module("groundingpi_reference.modeling_groundingpi")
    processing = importlib.import_module("groundingpi_reference.processing_groundingpi")
    images = importlib.import_module("groundingpi_reference.image_processing_groundingpi")
    return model, processing, images


def generate_torch(model, inputs, *, device, eos_token_id, max_tokens=256, max_context=4096):
    """Greedy generation with explicit cache positions; no dependence on HF generate()."""
    import torch

    device = torch.device(device)
    ids = inputs["input_ids"].to(device)
    if max_tokens <= 0 or ids.shape[1] + max_tokens > max_context:
        raise ValueError("invalid generation/context budget")
    eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
    if not eos:
        raise ValueError("EOS token IDs are required")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    cache, offset, generated = None, 0, []
    first_logits, prefill_seconds = None, None
    with torch.inference_mode():
        for step in range(max_tokens):
            kwargs = (
                {
                    "pixel_values": inputs["pixel_values"].to(
                        device=device, dtype=next(model.parameters()).dtype
                    ),
                    "image_grid_thw": inputs["image_grid_thw"],
                }
                if step == 0
                else {}
            )
            positions = torch.arange(offset, offset + ids.shape[1], device=device)
            output = model(
                input_ids=ids,
                past_key_values=cache,
                cache_position=positions,
                use_cache=True,
                logits_to_keep=1,
                **kwargs,
            )
            logits = output.logits[0, -1]
            token = int(logits.argmax().item())
            if step == 0:
                first_logits = logits.float().cpu().numpy()
                prefill_seconds = time.perf_counter() - start
            if token in eos:
                finish = "eos"
                break
            generated.append(token)
            offset += ids.shape[1]
            cache = output.past_key_values
            if cache is None:
                raise RuntimeError("official model did not return a cache for incremental decoding")
            ids = torch.tensor([[token]], device=device)
        else:
            finish = "length"
    return (
        generated,
        finish,
        {"elapsed_seconds": time.perf_counter() - start, "prefill_seconds": prefill_seconds},
        first_logits,
    )


def capture_visual(model):
    """Capture detached CPU arrays, not long-lived GPU tensors."""
    captured, handles = {}, []
    modules = {
        "patch_embeddings": model.model.visual.vision_tower.patch_embed,
        "vision_final": model.model.visual.vision_tower.encoder.final_layernorm,
        "projected_features": model.model.visual.projector,
    }

    def hook(name):
        def save(module, args, value):
            import torch

            if isinstance(value, (tuple, list)):
                value = torch.cat(value, dim=0)
            captured[name] = value.detach().float().cpu().numpy()

        return save

    for name, module in modules.items():
        handles.append(module.register_forward_hook(hook(name)))
    return captured, handles


def prepare_inputs(processor, tokenizer, image, instruction):
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=prompt, images=[str(image)], return_tensors="pt")


def export_reference(
    root,
    source,
    cases_path,
    output,
    *,
    device="cuda",
    max_tokens=256,
    capture_cases=("red-center", "absent-triangle", "photo-left-cat-box"),
):
    import torch
    from PIL import Image
    from transformers import AutoTokenizer

    # Protocol parsing has no MLX dependency; do not import generate.py on Linux.
    from .coordinates import FormatError, parse_answer, to_pixels

    root, source, cases_path, output = (
        Path(p).resolve() for p in (root, source, cases_path, output)
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; no automatic CPU fallback. Use --device cpu explicitly for tiny tests only."
        )
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(cases_path.read_text())
    names = [case["id"] for case in manifest["cases"]]
    if len(set(names)) != len(names) or any(
        not re.fullmatch(r"[a-z0-9-]+", name) for name in names
    ):
        raise ValueError("case IDs must be unique safe directory names")
    for case in manifest["cases"]:
        instruction_for(case)
        if not (cases_path.parent / case["image"]).is_file():
            raise FileNotFoundError(case["image"])
    # Validate fixtures before allocating a GPU model.
    for asset in manifest.get("assets", []):
        image = cases_path.parent / asset["path"]
        if hashlib.sha256(image.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"image digest mismatch: {image}")
    checksum = source / "checksums.sha256"
    if not checksum.is_file():
        raise ValueError("published original checksums.sha256 is required")
    hashes = {
        name.lstrip("*"): digest
        for digest, name in (line.split(maxsplit=1) for line in checksum.read_text().splitlines())
    }
    index = json.loads((source / "model.safetensors.index.json").read_text())
    if device == "cpu" and index["metadata"].get("total_parameters", 0) > 100_000_000:
        raise ValueError(
            "CPU reference export is limited to tiny tests; use CUDA for the full checkpoint"
        )
    for shard in set(index["weight_map"].values()):
        if Path(shard).name != shard:
            raise ValueError("invalid shard name")
        with (source / shard).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != hashes[shard]:
                raise ValueError(f"original weight checksum mismatch: {shard}")
    modules, processor_module, image_module = load_reference_classes(root)
    tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    tokenizer.chat_template = (source / "chat_template.jinja").read_text()
    proc_cfg = json.loads((source / "preprocessor_config.json").read_text())["media_proc_cfg"]
    processor = processor_module.GroundingPIProcessor(
        image_processor=image_module.GroundingPIImageProcessor(proc_cfg),
        tokenizer=tokenizer,
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    started = time.perf_counter()
    model = (
        modules.GroundingPIForConditionalGeneration.from_pretrained(
            source,
            dtype=dtype,
            attn_implementation="eager",
            local_files_only=True,
        )
        .eval()
        .to(device)
    )
    if device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in manifest["cases"]:
        image = (cases_path.parent / case["image"]).resolve()
        instruction = instruction_for(case)
        inputs = prepare_inputs(processor, tokenizer, image, instruction)
        capture, handles = capture_visual(model) if case["id"] in capture_cases else ({}, [])
        try:
            ids, finish, timing, first_logits = generate_torch(
                model,
                inputs,
                device=device,
                eos_token_id=model.config.eos_token_id,
                max_tokens=max_tokens,
            )
        finally:
            for handle in handles:
                handle.remove()
        raw = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        with Image.open(image) as im:
            size = im.size
        result = {
            "task": case["task"],
            "raw_output": raw,
            "finish_reason": finish,
            "valid": False,
            "predictions": [],
            "parse_error": None,
            "timing": timing,
        }
        try:
            if finish != "eos":
                raise ValueError("generation truncated")
            entries = parse_answer(raw, case["task"], require_canonical=True)
            result["predictions"] = [
                {
                    "label": e.phrase,
                    "coordinates": e.coordinates,
                    "pixels": [to_pixels(c, *size) for c in e.coordinates]
                    if e.coordinates
                    else None,
                }
                for e in entries
            ]
            result["valid"] = True
        except (FormatError, ValueError) as exc:
            result["parse_error"] = str(exc)
        arrays = {
            "input_ids": inputs["input_ids"].numpy(),
            "image_grid_thw": inputs["image_grid_thw"].numpy(),
            "first_logits": first_logits,
            "generated_ids": np.array(ids, dtype=np.int64),
            **capture,
        }
        if capture:
            arrays["pixel_values"] = inputs["pixel_values"].numpy()
        np.savez(output / (case["id"] + ".npz"), **arrays)
        row = {
            "id": case["id"],
            "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "instruction": instruction,
            "assessment": assess(result, case.get("expected")),
            "result": result,
        }
        rows.append(row)
        report = {
            "backend": "official-pytorch",
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "transformers_version": importlib.import_module("transformers").__version__,
            "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "manifest_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "load_seconds": load_seconds,
            "cases": rows,
        }
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"{case['id']}: {row['assessment']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="repository with pinned upstream/ snapshots"
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="local original BF16 checkpoint, not MLX weights"
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    export_reference(
        args.root,
        args.source,
        args.cases,
        args.output,
        device=args.device,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
