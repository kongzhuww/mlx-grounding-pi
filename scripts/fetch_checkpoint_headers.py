"""Download only bounded safetensors metadata from the pinned official checkpoint."""

import json
from pathlib import Path

from mlx_grounding_pi.checkpoint import fetch_header, verify_schema
from mlx_grounding_pi.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "upstream/manifest.json").read_text())
    index = json.loads((ROOT / "upstream/hf/model.safetensors.index.json").read_text())
    config = load_config(ROOT / "upstream/hf/config.json")
    headers = {}
    out = ROOT / "upstream/headers"
    out.mkdir(exist_ok=True)
    for shard in sorted(set(index["weight_map"].values())):
        url = f"https://huggingface.co/GroundingPI/GroundingPI/resolve/{manifest['hf_revision']}/{shard}"
        header = fetch_header(url)
        headers[shard] = header
        (out / (shard + ".json")).write_text(json.dumps({"url": url, **header}, indent=2) + "\n")
        print(f"{shard}: read {header['header_bytes'] + 8:,} header bytes, not payload", flush=True)
    print(json.dumps(verify_schema(config, index, headers), indent=2))


if __name__ == "__main__":
    main()
