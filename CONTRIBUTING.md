# Contributing

This is an unofficial, experimental Apple Silicon implementation of GroundingPI.
Please distinguish protocol validity, localization correctness, and agreement with
the original model when reporting results.

## Development

On Apple Silicon with Python 3.11+:

```sh
uv sync --locked --group reference
TORCHDYNAMO_DISABLE=1 uv run --group reference pytest -q
uv run ruff check .
uv run ruff format --check .
uv run python scripts/audit_checkpoint.py
uv build --wheel
```

Tests use small local models. Some real-tokenizer checks explicitly skip if the
optional original assets are absent. No test starts a cloud task. MLX tests require
Apple Silicon; a standard Linux runner is not a substitute.

## Reproduction reports

Include model revision, quantization, image-token budget, exact instruction,
versions, and sanitized output. For accuracy claims include independent labels,
sample count and false positives as well as successes. Do not upload private images,
credentials, raw cloud logs or copyrighted dataset archives.

The absent-triangle development case is a known failure in the original CUDA BF16
model too. Do not silence the failure or return invented confidence values.

## Scope and licensing

Keep inference improvements here. Missing-target datasets and fine-tuning experiments
live in the separate rejection lab. Preserve upstream licenses, source hashes and
notices. Original project contributions use Apache-2.0; applicable Kimi-derived
material remains subject to Kimi K3 terms. See THIRD_PARTY_NOTICES.md.

Cloud jobs are optional and billable. They require explicit approval of provider,
resources and budget; the watchdog also requires `--approve-paid-run`.
