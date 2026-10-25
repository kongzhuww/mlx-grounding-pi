# Mac development results — real checkpoint

## Scope and reproducibility

Host: Apple M4, 16 GiB unified memory; Python 3.12; MLX 0.32.4 / mlx-lm 0.32.2.
Weights: official HF revision `08d397c8769f7f10b96ff6d56ee6f56acb04181f`;
both source shard SHA-256 hashes match the published manifest.

Two versions quantize only **254 language matrices**, affine group size 64:

| Version | Tensor bytes | Vision/projector |
|---|---:|---|
| 4-bit language | 3,340,786,048 (3.34 GB) | BF16 |
| 8-bit language | 5,548,279,168 (5.55 GB) | BF16 |

Current measurements use the corrected BF16 visual input path. Historical reports
ending `*-default` predate the explicit processor-pixel dtype cast and are **not**
the current baseline. Their data are retained rather than rewritten.

Commands:

```sh
uv run python scripts/prepare_evaluation.py
uv run python -m mlx_grounding_pi.evaluation \
  --model models/groundingpi-mlx-4bit --cases examples/evaluation_cases.json \
  --output outputs/evaluation-4bit-new
# Repeat with the 8-bit model and a new output directory.
```

The evaluation currently exits **2** at both bit widths because the negative case
fails. All 10 responses finish at EOS and parse, but that is not 100% accuracy.

## Exact synthetic geometry (five positive cases)

Point threshold: maximum 30 pixels. Box threshold: minimum matched IoU 0.70.
Multi-instance matching is order-independent and requires the exact instance count.

| Case | 4-bit | 8-bit |
|---|---:|---:|
| Red-square center error | 2.962 px | 0.289 px |
| Blue-circle center error | 1.752 px | 2.639 px |
| Red-square box IoU | 0.9924 | 0.9887 |
| Blue-circle box IoU | 0.9686 | 0.9722 |
| Two red squares: minimum matched IoU | 0.9794 | 0.9830 |

All five positives pass both quantizations. This tiny hand-selected set cannot
establish that 8-bit is generally better, nor approximate a standard benchmark.

## Failed negative case — do not suppress

Input image contains one red square and one blue circle, **no yellow triangle**.
Instruction: `Point to the target referred to by the following description: the yellow triangle.`
Expected payload: `None`.

- 4-bit returns coordinate tokens `<177><80>`.
- 8-bit returns coordinate tokens `<915><46>`.

Both outputs are grammatical but **semantically false positives**. No confidence
threshold or output filter has been invented to make this test pass. Persistence
at 8-bit alone would not identify its cause. The official CUDA BF16 run now also returns
`<915><46>`, matching the complete 8-bit output. This particular failure is reproduced
in the original model, not unique to the port. It remains a failed negative case; no
output filter or correctness claim has been added.

## Real photographs: visual inspection, not benchmark accuracy

Hash-pinned public documentation photos are listed in `examples/evaluation_cases.json`.
They stay locally under ignored `outputs/`; original photos are not redistributed
as package assets. No official COCO/VOC annotations have been claimed or invented.

Four corrected 4-bit overlays were opened and visually reviewed:

- Left cat: box encloses the left cat, not the right cat.
- Right cat: point lies on the body of the right cat.
- Dog: box encloses the dog, including its tail.
- Dog nose: point lies on the nose.

These results are visually plausible. They have no numerical accuracy score without
independent annotations. The 8-bit cases also return parsed outputs; the primary
qualitative inspection above is specifically of the corrected 4-bit contact sheet.

Local review artifact:
`outputs/evaluation-4bit-bf16vision/photo-contact-sheet.png`.

## Observed latency / MLX memory

One sequential run per quantization; model loaded once; no statistical repetitions.
Times below include image prefill and greedy decoding, **exclude loading and image
preprocessing**, and use the original dynamic image-token budget.

| Case group | 4-bit generation | 8-bit generation | Image tokens |
|---|---:|---:|---:|
| Synthetic images | 1.32–1.55 s | 1.56–1.85 s | 256 |
| Cat photo | 2.11–2.15 s | 2.36–2.38 s | 414 |
| Dog photo | 4.78–4.87 s | 5.00–5.13 s | 1008 |

- Model + processor load: 1.78 s (4-bit), 2.23 s (8-bit), with local filesystem cache.
- Per-case MLX peak allocation: 4.24–4.40 GB (4-bit), 6.45–6.61 GB (8-bit).
- MLX allocation is **not** total process RSS or total system unified-memory usage.
- These are development measurements, not a cold-start or cross-hardware benchmark.

Raw reports: `outputs/evaluation-4bit-bf16vision/report.json` and
`outputs/evaluation-8bit-bf16vision/report.json`.

## Official reference comparison and remaining limits

MLX logits/tokens and selected vision stages are saved under
`outputs/parity-mlx-4bit/` and `outputs/parity-mlx-8bit/` (about 55 MB each).
On the three captured cases, the two versions have exactly identical processor
pixels, patch embeddings, final visual features and projected features. Language
outputs differ; first-token argmax agreement alone is not meaningful accuracy evidence
because the first token is a shared protocol marker.

The authorized one-shot Modal CUDA BF16 run completed in 194.6 seconds local wall time
including build/wait/transfer. Its App is independently confirmed stopped. All 10
artifacts are under `outputs/reference-bf16/`, with comparisons under
`outputs/comparison-official-vs-{4,8}bit.json`.

- 8-bit: **8/10 complete generated token sequences match** the official reference.
  The left-cat box and dog-nose point differ; the latter is `(187,208)` vs `(188,209)`
  in normalized coordinates.
- 4-bit: **0/10 complete token sequences match**; this is not zero localization accuracy.
  The five synthetic positives still pass their point/IoU thresholds.
- Exact sequence agreement is **not accuracy**: it includes the missing-target false positive.
- Captured processor pixels match exactly. Unquantized BF16 vision features differ
  by about 7% relative L2, and projector outputs by about 4.8%; these differences are
  the same for 4-bit and 8-bit, so cannot be attributed to language quantization.

Local diagnostics using only real vision/projector weights, cast to FP32 on both CPU
PyTorch and GPU MLX, give final-vision relative L2 around `1.04e-5` and projector relative
L2 below `7.8e-6` on both the synthetic scene and real cat photo. The official PyTorch
CPU BF16 path itself differs from CUDA BF16 by 6.47% final-vision relative L2 on the
synthetic scene. This supports low-precision backend drift, but does not eliminate it
or establish full-model numerical parity. See the [cloud reference record](cloud-reference.md)
for operator probes, exact report paths and reproduction commands.

**Working recommendation:** use 8-bit when memory allows if agreement with this official
baseline is the priority; use 4-bit for lower memory. Neither setting is validated for
reliable absence detection or a standard benchmark, and no general quality ranking is
inferred from just 10 hand-selected cases.
