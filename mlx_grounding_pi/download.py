"""Download the pinned official checkpoint, without importing any model code."""

import argparse
from pathlib import Path

HF_REPO = "GroundingPI/GroundingPI"
HF_REVISION = "08d397c8769f7f10b96ff6d56ee6f56acb04181f"
FILES = [
    "checksums.sha256",
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "generation_config.json",
    "LICENSE",
    "LICENSE-Apache-2.0",
    "LICENSE-Kimi-K3",
]


def download(output, *, weights=False):
    from huggingface_hub import snapshot_download

    patterns = FILES + (["model-*.safetensors"] if weights else [])
    return Path(
        snapshot_download(
            repo_id=HF_REPO,
            revision=HF_REVISION,
            allow_patterns=patterns,
            local_dir=output,
            max_workers=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--weights",
        action="store_true",
        help="also download approximately 9.69 GB of weight payload",
    )
    args = parser.parse_args()
    if args.weights:
        print("Downloading about 9.69 GB of pinned original BF16 weights (resumable).", flush=True)
    else:
        print("Downloading tokenizer/configuration only; no model weights.", flush=True)
    print(download(args.output, weights=args.weights))


if __name__ == "__main__":
    main()
