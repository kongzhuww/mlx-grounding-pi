"""Bounded-memory GroundingPI -> MLX conversion; never executes checkpoint code."""

import argparse
import gc
import hashlib
import json
import shutil
import struct
import uuid
from pathlib import Path

from .checkpoint import MAX_HEADER_BYTES, verify_schema
from .config import load_config

ASSET_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "generation_config.json",
    "LICENSE",
    "LICENSE-Apache-2.0",
    "LICENSE-Kimi-K3",
)


def map_weight_key(key: str) -> str:
    if key.startswith("model.visual."):
        return key.removeprefix("model.")
    if key.startswith("model.language_model."):
        return "language_model.model." + key.removeprefix("model.language_model.")
    if key == "lm_head.weight":
        return "language_model.lm_head.weight"
    raise ValueError(f"unrecognized checkpoint tensor: {key}")


def convert_tensor(key, value):
    """Torch OIHW -> MLX OHWI for the patch convolution only; linear stays OI."""
    if key == "model.visual.vision_tower.patch_embed.proj.weight":
        if value.ndim != 4:
            raise ValueError("expected four-dimensional patch convolution weight")
        return value.transpose(0, 2, 3, 1)
    return value


def local_headers(source: Path, index: dict) -> dict:
    headers = {}
    for shard in sorted(set(index["weight_map"].values())):
        if Path(shard).name != shard:
            raise ValueError("shard filenames must not contain directory components")
        path = source / shard
        with path.open("rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                raise ValueError(f"not a safetensors shard: {shard}")
            length = struct.unpack("<Q", prefix)[0]
            if not 2 <= length <= MAX_HEADER_BYTES:
                raise ValueError(f"invalid header size: {shard}")
            raw = f.read(length)
            if len(raw) != length:
                raise ValueError(f"truncated header: {shard}")
        headers[shard] = {
            "header_bytes": length,
            "file_bytes": path.stat().st_size,
            "tensors": json.loads(raw),
        }
    return headers


def convert_checkpoint(source, output, *, bits=0, group_size=64, shard_size_mb=256):
    """Convert one tensor at a time. Only language matrices are optionally quantized.

    Output is published only after conversion finishes; failed partial directories
    are left intact for diagnosis, never silently removed or reused.
    """
    import mlx.core as mx
    import numpy as np

    source, output = Path(source).resolve(), Path(output).resolve()
    if bits not in (0, 4, 8) or group_size not in (32, 64, 128):
        raise ValueError("bits must be 0/4/8; group_size must be 32/64/128")
    if type(shard_size_mb) is not int or shard_size_mb <= 0:
        raise ValueError("shard_size_mb must be a positive integer")
    if output.exists() or output == source:
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    config = load_config(source / "config.json")
    if "mlx_grounding_pi" in config or "quantization" in config:
        raise ValueError("conversion expects the original, unquantized checkpoint")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    headers = local_headers(source, index)
    verified = verify_schema(config, index, headers)
    expected_hashes = {}
    checksum_file = source / "checksums.sha256"
    if checksum_file.is_file():
        for line in checksum_file.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            expected_hashes[name.lstrip("*")] = digest
        if not set(headers) <= set(expected_hashes):
            raise ValueError("source checksum manifest is missing weight shards")
    output.parent.mkdir(parents=True, exist_ok=True)
    working = output.with_name(output.name + ".partial-" + uuid.uuid4().hex[:8])
    working.mkdir(exist_ok=False)
    pending, weight_map, quantized_modules, source_hashes = {}, {}, [], {}
    size, total_size, shard_id = 0, 0, 0
    limit = shard_size_mb * 1024**2

    def flush():
        nonlocal pending, size, shard_id
        if not pending:
            return
        shard_id += 1
        name = f"model-{shard_id:05d}.safetensors"
        mx.save_safetensors(str(working / name), pending, metadata={"format": "mlx"})
        for key in pending:
            weight_map[key] = name
        pending, size = {}, 0
        gc.collect()
        mx.clear_cache()

    for shard in sorted(headers):
        print(f"Converting {shard}", flush=True)
        with (source / shard).open("rb") as f:
            source_hashes[shard] = hashlib.file_digest(f, "sha256").hexdigest()
        if expected_hashes and source_hashes[shard] != expected_hashes[shard]:
            raise ValueError(f"source shard SHA-256 does not match published checksum: {shard}")
        with (source / shard).open("rb") as f:
            specs = headers[shard]["tensors"]
            for key in sorted(k for k in specs if k != "__metadata__"):
                spec = specs[key]
                start, end = spec["data_offsets"]
                f.seek(8 + headers[shard]["header_bytes"] + start)
                raw = f.read(end - start)
                if len(raw) != end - start:
                    raise ValueError(f"truncated tensor payload: {key}")
                # Preserve BF16 bits without NumPy/ml_dtypes BF16 support or Torch.
                value = mx.array(np.frombuffer(raw, dtype="<u2")).view(mx.bfloat16)
                value = convert_tensor(key, value.reshape(spec["shape"]))
                del raw
                target = map_weight_key(key)
                if (
                    bits
                    and target.startswith("language_model.")
                    and target.endswith(".weight")
                    and value.ndim == 2
                    and value.shape[-1] % group_size == 0
                ):
                    path = target.removesuffix(".weight")
                    w, scales, biases = mx.quantize(
                        value, group_size=group_size, bits=bits, mode="affine"
                    )
                    entries = {target: w, path + ".scales": scales, path + ".biases": biases}
                    quantized_modules.append(path)
                else:
                    entries = {target: value}
                mx.eval(*entries.values())
                amount = sum(v.nbytes for v in entries.values())
                if pending and size + amount > limit:
                    flush()
                pending.update(entries)
                size += amount
                total_size += amount
                del entries, value
                # The largest single matrix can exceed the soft shard target.
                if size >= limit:
                    flush()
    flush()
    config["mlx_grounding_pi"] = {
        "format_version": 1,
        "quantization": (
            {
                "bits": bits,
                "group_size": group_size,
                "mode": "affine",
                "modules": sorted(quantized_modules),
            }
            if bits
            else None
        ),
    }
    for name in ASSET_NAMES:
        if (source / name).is_file():
            shutil.copyfile(source / name, working / name)
    (working / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": dict(sorted(weight_map.items())),
            },
            indent=2,
        )
        + "\n"
    )
    (working / "conversion.json").write_text(
        json.dumps(
            {
                "source_schema": verified,
                "source_sha256": source_hashes,
                "published_checksums_verified": bool(expected_hashes),
                "quantized_language_modules": len(quantized_modules),
                "vision_quantized": False,
                "output_tensor_bytes": total_size,
                "note": "Conversion success is not a claim of grounding accuracy or quantization parity.",
            },
            indent=2,
        )
        + "\n"
    )
    (working / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if output.exists():
        raise FileExistsError(f"output appeared during conversion: {output}")
    working.rename(output)
    print(f"Saved {len(weight_map)} tensors in {shard_id} shards to {output}", flush=True)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--bits",
        type=int,
        choices=[0, 4, 8],
        default=0,
        help="language-only quantization; 0 preserves original BF16",
    )
    parser.add_argument("--group-size", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--shard-size-mb", type=int, default=256)
    args = parser.parse_args()
    convert_checkpoint(
        args.source,
        args.output,
        bits=args.bits,
        group_size=args.group_size,
        shard_size_mb=args.shard_size_mb,
    )


if __name__ == "__main__":
    main()
