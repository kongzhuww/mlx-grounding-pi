# Validation status

## Host and environment

Apple M4, 16 GiB unified memory, Python 3.12. MLX runs on `Device(gpu, 0)`.
The reported recommended GPU working set is 11,453,251,584 bytes. Exact dependency
versions are recorded in `uv.lock`; PyTorch is isolated in the optional `reference` group.

```sh
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run python scripts/audit_checkpoint.py
TORCHDYNAMO_DISABLE=1 uv run --group reference pytest -q -s tests/test_reference_parity.py
```

## Verified real checkpoint metadata

HTTP Range reads fetched only the two safetensors headers: 24,696 and 49,704 bytes
including their eight-byte length prefixes. All **569** tensor names, shapes, BF16
dtypes, byte lengths, continuous offsets, shard assignments and parameter totals
match the independently generated architecture schema. Header snapshots are in
`upstream/headers/`; no full-model allocation is needed for this check.

## Verified official PyTorch / MLX parity

Tests execute the active classes from the pinned, locally hash-verified upstream
source. No remote model code is downloaded or executed by the inference CLI.

Using identical **tiny random FP32 weights**, with fixed seeds:

| Comparison | Observed maximum absolute error |
|---|---:|
| Patch convolution + interpolated learned positions | 4.77e-7 |
| Individual vision blocks | 1.20e-6 |
| Vision final RMSNorm / spatial merger | 5.97e-7 |
| Projector | 7.16e-7 |
| Full multimodal logits (vision + connector + Qwen3) | 5.97e-8 |

The configured assertions use `atol=5e-6, rtol=5e-5`; these are small-model FP32
operator checks, not justified BF16/4-bit tolerances for the full checkpoint.

Original RGB/RGBA/grayscale processing is elementwise identical. If the pinned
local tokenizer assets are present, tests also compare the official chat prompt,
expanded image placeholder token IDs and normalized pixels end to end. That test
skips explicitly if tokenizer assets have not been downloaded.

## Verified conversion and loading

Synthetic checkpoint tests use actual BF16 safetensors and check:

- Source header/schema validation and rejection of truncated shards.
- Published source SHA-256 verification and rejection of a corrupted digest.
- Streaming per-tensor BF16 reading without NumPy BF16 or Torch requirements.
- Patch-convolution OIHW -> OHWI conversion and unchanged linear layouts.
- Unquantized, language-only 4-bit and 8-bit export/reload.
- Converted weights produce the same tiny-model logits as the corresponding
  in-memory model/quantization (this is **not** an accuracy comparison to BF16).
- Multiple output shards, exact names/shapes/coverage on load, no directory overwrite.
- Vision tower/projector excluded from quantization.

## Verified inference plumbing

- Image embeddings replace the exact number of saved-template placeholders.
- Cached decoding agrees with full causal decoding for a tiny multimodal model.
- Last-token-only vocabulary projection agrees with full logits, avoiding a large
  `[prompt_length,vocab_size]` prefill allocation during generation.
- Greedy decoding feeds images only on prefill, observes EOS and enforces context limits.
- Strict official GAM grammar, missing targets, invalid/truncated output handling.
- Result JSON and original-image point/box overlays without overwriting existing outputs.

## Real checkpoint runs

Both original shards have been downloaded and match the published SHA-256 checksums.
Real language-4-bit/BF16-vision and language-8-bit/BF16-vision conversions run on this Mac.
The first known-target smoke test passed; an expanded 10-case development suite has now
run at both bit widths. Five synthetic positives pass, one missing-target negative fails,
and four photo outputs have been visually inspected (not scored against benchmark labels).
See [measured results](mac-results.md) for per-case metrics and memory/latency.

A dtype regression was fixed: processor FP32 pixels are explicitly converted to the
patch-convolution weight dtype, matching the official image-feature path. Earlier
`*-default` reports used promoted vision activations; current `*-bf16vision` reports
supersede them. A unit test prevents reintroducing this accidental promotion.

Real MLX artifacts have been exported for both bit widths. On three captured cases,
processor pixels, patch embeddings, final visual features and projector outputs are
identical between 4-bit and 8-bit runs, as expected for their unchanged vision path.
That equality alone is not comparison to official BF16 PyTorch.

## Real official BF16 reference and FP32 isolation

The authorized one-shot Modal A10G export completed, and its App is confirmed stopped.
All 10 returned artifacts were compared against both MLX exports after exact manifest,
image, instruction, input-ID and grid checks. Captured pixels match exactly.

- Official CUDA BF16 reproduces the missing-target false positive `(915,46)`, exactly
  matching the 8-bit complete token sequence. This specific failure is not unique to
  the Mac port; reliable rejection of missing objects remains unverified.
- Complete token sequences agree in 8/10 cases for 8-bit and 0/10 for 4-bit. This is
  literal agreement, not 80%/0% accuracy; the five synthetic positives pass all versions.
- Real BF16 final vision/projector features differ by about 7%/4.8% relative L2.
  No arbitrary pass tolerance is applied to call this full numerical parity.
- Local real-weight FP32 checks cover all 27 vision blocks and the projector on the
  synthetic red-square scene and real cat photo. Final-vision relative L2 is about
  `1.04e-5` and projector relative L2 below `7.8e-6` on both.
- Official CPU BF16 vs CUDA BF16 also differs by 6.47% final-vision relative L2 on the
  synthetic scene. Common-input RMSNorm/GELU probes and FP32 agreement support low-precision
  backend drift; they do not prove every BF16 mismatch is understood or fixed.

See the [cloud reference record](cloud-reference.md) for versions, metrics, remaining
limitations, exact artifact paths and local diagnostic commands.

## Not yet verified / unresolved

- Full-model unquantized MLX BF16 or FP32 language logits/generation parity.
- Eliminating BF16 cross-backend feature drift or establishing validated error bounds.
- Reliable missing-target rejection; the official model fails this negative too.
- Quantization accuracy on a representative dataset, BF16 error bounds, benchmark scores.
- Statistical performance across repeats, cold-start filesystem conditions or other Macs.

A successful unit test run or successful cloud export does not erase a semantic
evaluation failure. `valid` in
inference output denotes protocol validity only, not a ground-truth correctness claim.

## Latest local regression check

68 tests pass, including static cloud-resource/upload/watchdog scope checks and
an execution test that refuses a cloud launch without explicit paid opt-in.
Ruff lint/format checks, the offline 569-tensor header/source audit, and wheel build
also pass. This is separate from the development localization evaluation, which
continues to report the known missing-target failure and exit code 2.

## Intentional limitations

- Single-image grids only (`t=1`); videos fail explicitly.
- Batch size 1; no padded batched language generation.
- No camera access, hardware connectivity or robot control.
- The default image processor can emit up to 1024 visual tokens. Lower budgets are
  explicit arguments; this wrapper does not automatically read `IMAGE_MAX_TOKEN_NUM`.
- The official-reference test uses upstream classes with the locally installed
  Transformers version, not the upstream CUDA/vLLM serving environment.
