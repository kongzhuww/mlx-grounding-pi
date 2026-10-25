# Official BF16 reference run

## Purpose

The Mac's 4-bit and 8-bit models both hallucinate a location for a missing yellow
triangle. The completed official CUDA BF16 run also returns `<915><46>`, exactly matching
8-bit. Thus this particular false positive is reproduced in the original model; it is
not uniquely a Mac-port or 4-bit problem. This does not explain every quantization or
backend difference, nor establish general missing-target performance.

The portable exporter is implemented in `mlx_grounding_pi/reference.py` and its
cached-generation positions / capture hooks are tested against a tiny official
PyTorch model on CPU. A one-shot Modal A10G job has now been explicitly authorized
and completed using the existing Modal CLI 1.6.2 and existing login. No SDK installation
or credential-content access was needed. All 10 returned artifacts have been compared
against both MLX exports, checking manifest/image hashes, instructions, IDs and grids.
Local wall time was **194.6 seconds including build/wait/transfer**, not GPU-billed time.
The session's App was independently confirmed **already stopped**; no new
cloud run was launched. Personal workspace identifiers are omitted from this public record. Status/logs are local at `outputs/modal-reference-status.json`
and `outputs/modal-reference.log`. Actual billing is not measured by these timings.

## Authorized job scope

- One NVIDIA GPU: **24 GB VRAM minimum starting point**, 48 GB for more headroom.
- About 32 GiB host RAM and 40 GB free scratch space for weights/environment/cache.
- One-shot batch process; no HTTP service, public port or persistent deployment.
- Provider-enforced **20-minute execution timeout**, plus a local 20-minute total wall
  watchdog (including build/wait), automatic single-use container termination, retries=0.
- CPU is configured `(4,4)` and RAM `(32768,32768)` to set hard limits, not merely minimums.
- No persistent volumes or deployment; only the explicit source/test allowlist is uploaded.
- This is a runtime ceiling, not a price estimate or a guarantee the download finishes.
- Inputs: the two included synthetic images and two hash-pinned public documentation
  photos. No private user images, SSH keys, browser state or other home-directory data.

Modal is suitable for a short-lived job; Azure is equally usable if an appropriate
instance already exists. User approval of provider, GPU tier and spending cap is
required before launching; that approval has been given for this single Modal run.
Do not request or paste credential contents into chat.

## Explicit opt-in for a new paid run

The published watchdog uses the active Modal profile (or explicit `--profile NAME`),
not the original developer's account. It refuses to run without `--approve-paid-run`.
A new run incurs charges and requires your own approval and authenticated account:

```sh
python scripts/run_bounded_modal.py --approve-paid-run
```

Do not run this merely to install or test the Mac package. Ordinary tests do not
connect to Modal. Local logs/returned arrays remain ignored and are not published.

## Commands on the authorized CUDA host

Copy the project source, `upstream/`, `examples/` and `scripts/`, but **not** the
Mac `.venv`, `models/`, `outputs/`, or unrelated home-directory files.

From the repository root (Python 3.12, CUDA-capable Torch runtime):

```sh
uv sync --locked --group reference
uv run python scripts/prepare_evaluation.py
uv run python -m mlx_grounding_pi.download --output models/groundingpi-hf --weights
TORCHDYNAMO_DISABLE=1 uv run --group reference python -m mlx_grounding_pi.reference \
  --root . --source models/groundingpi-hf \
  --cases examples/evaluation_cases.json --output outputs/reference-bf16
```

The command explicitly imports hash-verified local upstream classes and verifies
both original shard SHA-256 values. It does not use `trust_remote_code` or require
vLLM/FlashAttention. Vision/text attention use the eager reference path. It fails
rather than silently falling back from CUDA to CPU; full-size CPU export is refused.

If using a provider-supplied Torch environment, confirm it matches the locked
reference dependencies and record any intentional deviation. Do not blindly
replace its working CUDA build as part of a billable job.

## Returned artifacts

Return **only** `outputs/reference-bf16/`:

- `report.json`: versions/device, exact instructions, image hashes, raw output,
  parsed predictions, generation timings and per-case assessments.
- Ten `.npz` files: exact input IDs, image grids, first-token logits and generated IDs.
- For `red-center`, `absent-triangle`, `photo-left-cat-box`: additionally FP32-exported
  patch embeddings, normalized visual features, projected features and processor pixels.

Array casts are for portable artifact storage; they do not turn a BF16 model run
into an FP32 baseline. Timings include reference/capture overhead and are not
cross-hardware performance comparisons. A completed cloud export is not a claim
that its negative-target cases passed.

## Compare on the Mac

Mac artifacts have already been exported under `outputs/parity-mlx-4bit/` and
`outputs/parity-mlx-8bit/`. Regenerate them if the implementation or test inputs change:

```sh
uv run python -m mlx_grounding_pi.parity export-mlx \
  --model models/groundingpi-mlx-4bit --cases examples/evaluation_cases.json \
  --output outputs/parity-mlx-4bit-new
uv run python -m mlx_grounding_pi.parity compare \
  --reference outputs/reference-bf16 --mlx outputs/parity-mlx-4bit \
  --output outputs/comparison-official-vs-4bit.json
```

The comparator refuses mismatched test manifests, images, instructions, input IDs
or image grids. It reports stage-wise max/mean absolute error, relative L2 and cosine,
first-token argmax/top-10 overlap and full generated-token equality. It does not
invent a numerical pass threshold, nor expect a quantized LLM to be bit-identical.

Interpretation order:

1. Match preprocessing/input IDs exactly.
2. Examine unchanged BF16 vision/projector differences before blaming quantization.
3. Compare original missing-target behavior to Mac behavior.
4. Only then assess 4-bit vs 8-bit localization changes across more than one example.

## Observed comparison

Reference runtime: Torch `2.14.1+cu130`, Transformers `5.20.2`, NVIDIA A10,
BF16, eager vision/text attention. Five synthetic positives pass; the missing-target
negative fails. The four photographs have no benchmark ground truth.

| Measurement | 4-bit language | 8-bit language |
|---|---:|---:|
| Complete generated token sequences matching official | 0 / 10 | 8 / 10 |
| Processor pixels, captured cases | exact | exact |
| Final visual features: relative L2 | 6.96–7.01% | same |
| Projected features: relative L2 | 4.83–4.86% | same |
| First-token logits: relative L2 range | 12.9–142.9% | 4.64–19.2% |

These are agreement/error measurements, **not detection accuracy**. Exact agreement
includes the false positive. The two differing 8-bit cases are the left-cat box and
dog-nose point. For example, dog-nose normalized coordinates are `(188,209)` in the
reference and `(187,208)` in MLX. No full-model BF16 parity threshold is claimed.

Reports: `outputs/reference-bf16/report.json`,
`outputs/comparison-official-vs-4bit.json`, `outputs/comparison-official-vs-8bit.json`.

## Local real-weight precision isolation (no additional cloud run)

`scripts/diagnose_real_vision.py` loads only the real vision tower/projector, not the
4B language model. CPU PyTorch and GPU MLX use the same real BF16 checkpoint values
cast to FP32. Both execute all 27 vision layers and the projector:

| Image | Final normalized vision relative L2 | Projector relative L2 |
|---|---:|---:|
| Red-square synthetic scene | 1.032e-5 | 7.780e-6 |
| Real cat photograph | 1.039e-5 | 7.672e-6 |

The two FP32 cases support the structural/layout implementation, but do not validate
full FP32 language generation or every image. Internal residual max-absolute errors
can be much larger than final normalized errors; see per-layer reports rather than
applying tiny-random-weight tolerances to these real activations.

In BF16 with common first-layer inputs, CPU PyTorch vs MLX RMSNorm and tanh-GELU have
relative L2 differences of about 0.281% and 0.320%. Running the same official model on
CPU BF16 vs the returned CUDA BF16 baseline yields **6.47%** final-vision / **4.39%**
projector relative L2 on the synthetic scene, versus MLX's 7.01% / 4.86%. This supports
low-precision backend/operator differences amplified through layers, not an assertion
that every mismatch has been fixed or isolated. Production precision behavior has
not been changed to force a match.

```sh
TORCHDYNAMO_DISABLE=1 uv run --group reference python scripts/diagnose_real_vision.py \
  --dtype fp32 --output outputs/vision-diagnostic-fp32-new
TORCHDYNAMO_DISABLE=1 uv run --group reference python scripts/diagnose_real_vision.py \
  --dtype fp32 --case photo-left-cat-box --output outputs/vision-diagnostic-fp32-cat-new
TORCHDYNAMO_DISABLE=1 uv run --group reference python scripts/diagnose_real_vision.py \
  --dtype bf16 --output outputs/vision-diagnostic-bf16-new
```

These local diagnostics require the already-downloaded original weights and returned
`outputs/reference-bf16/` arrays; they never download/deploy anything. Output directories
must be new. Reports are under `outputs/vision-diagnostic-{fp32,fp32-cat,bf16}/report.json`.
