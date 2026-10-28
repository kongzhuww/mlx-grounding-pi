# mlx-grounding-pi

**English** | [简体中文](README_zh.md)

Experimental native Apple MLX port of [GroundingPI](https://huggingface.co/GroundingPI/GroundingPI).
The short-term goal is local **image + instruction -> points/boxes** on Apple Silicon.
No robot hardware, camera calibration or actuation is involved.

**Unofficial community implementation; not affiliated with or endorsed by GroundingPI.**
The initial experimental baseline is kept separate from the
[missing-target fine-tuning research lab](https://github.com/kongzhuww/grounding-pi-rejection-lab).
That lab does not yet provide a validated training backend or trained adapter.

**Status:** real official weights run locally on a 16 GiB Mac M4, with language-only
4-bit or 8-bit quantization and a BF16 vision tower/projector. Point grounding,
boxes and two-instance grounding pass a small synthetic positive suite; four real-photo
results have been visually reviewed. **The absent-target case fails at both bit widths.**
The official CUDA BF16 baseline reproduces that false positive. On these 10 cases,
8-bit matches 8 complete official token sequences, versus 0 for 4-bit; sequence equality
is **not accuracy**. Real-weight FP32 vision diagnostics closely agree, but BF16 backend
numerical differences and benchmark accuracy remain unresolved. This is an experimental
port, not a validated detector. See [measured results](docs/mac-results.md).

## Install

Apple Silicon and Python 3.11+ (development uses Python 3.12):

```sh
uv sync --locked
```

## Download and convert

Download only tokenizer/configuration first (no model code is executed):

```sh
uv run python -m mlx_grounding_pi.download --output models/groundingpi-hf
```

Download the pinned original checkpoint, approximately **9.69 GB**:

```sh
uv run python -m mlx_grounding_pi.download \
  --output models/groundingpi-hf --weights
```

For a 16 GiB Mac, **language-only 4-bit** is the memory-saving option. For closer
agreement with the official model in this small development comparison, prefer **8-bit**
if memory allows (observed MLX peak about 6.6 GB, not total process memory). The vision
tower/projector stay BF16. Neither option has benchmark-level accuracy validation:

```sh
uv run python -m mlx_grounding_pi.convert \
  --source models/groundingpi-hf --output models/groundingpi-mlx-4bit \
  --bits 4 --group-size 64
# Alternative: --output models/groundingpi-mlx-8bit --bits 8 --group-size 64
```

Conversion reads one tensor at a time, verifies the source schema and published SHA-256
checksums when available, and writes bounded-size shards. Existing output directories
are never overwritten. Failed conversions leave a `.partial-*` directory for diagnosis.
Allow at least 15 GB free disk for originals plus a 4-bit conversion, or 18 GB for
originals plus an 8-bit conversion, and more for both conversions or unquantized exports. To preserve original weights use `--bits 0`; that does not guarantee
full unquantized inference will fit a 16 GiB Mac.

## Local image grounding

```sh
uv run python -m mlx_grounding_pi.generate \
  --model models/groundingpi-mlx-4bit \
  --image /path/to/image.jpg \
  --phrase "the red block" --task point \
  --output outputs/red-block
```

Use `--task bbox` for boxes, or `--prompt` instead of `--phrase` for a complete instruction.
The output is JSON with the raw protocol text, validity, 0–999 coordinates and original-image
pixel coordinates; valid results also save `prediction.png`. Existing output directories
are not overwritten. Truncated or malformed output is invalid (exit code 2), never `(0,0)`.
**`valid` means complete, parseable protocol, not proof the target exists or the location is
correct.** Both tested quantizations hallucinate a location for a missing yellow triangle.

Inference is local-only, greedy, one image/request. `--max-tokens` defaults to 256 and
`--max-context` to 4096. `--max-image-tokens 256` can reduce the default image-token budget,
but may hurt localization; omit it to follow the original processor limit.
The CLI conservatively refuses weights larger than 75% of the GPU's recommended working set.
It does not change system memory settings.

## First-run smoke test (development)

`examples/synthetic_scene.png` has a known red-square center; its expected location and
30-pixel smoke-test tolerance are in `examples/synthetic_scene.json`. The session's
`scripts/first_run.py` can wait for an existing download process, convert language weights
to 4-bit, run this image, and check the actual pixel error. It stops on the first error.
It does not establish benchmark accuracy or official full-checkpoint numerical parity.

For a session-owned background run, inspect `models/first-run-status.json` and
`models/first-run.log`. `waiting_for_download` is **not** success; only `completed`
means that conversion, generation, strict parsing and the known-target check passed.

## Reproduce the development evaluation

```sh
uv run python scripts/prepare_evaluation.py
uv run python -m mlx_grounding_pi.evaluation \
  --model models/groundingpi-mlx-4bit --cases examples/evaluation_cases.json \
  --output outputs/evaluation-new
```

This currently exits **2** because the missing-target test fails. Five positive synthetic
cases pass; four photo cases require separate visual review, not an automatic accuracy
score. Photos are public documentation fixtures with pinned hashes and stay in ignored
`outputs/`; the runner loads the model once and records each case independently.

The authorized one-shot [official CUDA reference job](docs/cloud-reference.md) completed
within its 20-minute cap; the App is confirmed stopped. Its 10-case BF16 artifacts have
been compared with both Mac versions. The negative case still fails in the original;
8/10 exact token agreement at 8-bit does not constitute full numerical parity or 80%
accuracy. See the report for remaining BF16 feature differences and local FP32 diagnostics.

## Development and validation

```sh
uv run pytest -q
uv run ruff check .
uv run python scripts/audit_checkpoint.py
# Optional official PyTorch comparison; Torch is not an inference dependency:
TORCHDYNAMO_DISABLE=1 uv run --group reference pytest -q -s tests/test_reference_parity.py
```

Reference snapshots are pinned and checksummed in `upstream/manifest.json`.
The released checkpoint has **4,843,664,384 parameters, 569 tensors and two weight shards**.
Only metadata/source snapshots are tracked; weights and generated outputs are ignored by Git.

See [release notes](docs/release-notes.md), [contributing](CONTRIBUTING.md),
[architecture audit](docs/architecture.md), [validation details](docs/validation.md)
and [third-party notices](THIRD_PARTY_NOTICES.md). Kimi-derived material carries additional
upstream license conditions; the complete model/port is not blanket Apache-only.
