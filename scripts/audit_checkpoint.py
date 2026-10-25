"""Offline audit of pinned source integrity and checkpoint weight-name coverage."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlx_grounding_pi.checkpoint import verify_schema
from mlx_grounding_pi.config import load_config
from mlx_grounding_pi.convert import map_weight_key


def main():
    manifest = json.loads((ROOT / "upstream/manifest.json").read_text())
    for item in manifest["files"]:
        digest = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"source digest mismatch: {item['path']}")
    config = load_config(ROOT / "upstream/hf/config.json")
    index = json.loads((ROOT / "upstream/hf/model.safetensors.index.json").read_text())
    headers = {
        shard: json.loads((ROOT / "upstream/headers" / (shard + ".json")).read_text())
        for shard in set(index["weight_map"].values())
    }
    schema = verify_schema(config, index, headers)
    mapped = [map_weight_key(k) for k in index["weight_map"]]
    if len(mapped) != len(set(mapped)):
        raise ValueError("weight-name conversion collision")
    print(
        json.dumps(
            {
                "hf_revision": manifest["hf_revision"],
                "github_revision": manifest["github_revision"],
                "verified_reference_files": len(manifest["files"]),
                "parameters": index["metadata"]["total_parameters"],
                "weight_bytes": index["metadata"]["total_size"],
                "tensors": len(mapped),
                "shards": sorted(set(index["weight_map"].values())),
                "vision_head_dim": config["vision_config"]["qkv_hidden_size"]
                // config["vision_config"]["num_attention_heads"],
                "verified_tensor_shapes_and_offsets": schema["verified_tensors"],
                "note": "Offline source and real checkpoint-header audit; no tensor payload or inference checked by this script.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
