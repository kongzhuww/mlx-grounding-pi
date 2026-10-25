# Initial experimental baseline — 0.1.0

## Included

- Native MLX MoonViT vision encoder and GroundingPI projector.
- Language backbone reused from `mlx_lm.models.qwen3`, not rewritten.
- Pinned original checkpoint download, header/checksum validation, language-only
  4/8-bit conversion, strict loading and single-image point/box inference.
- GAM parsing and overlays, optional official PyTorch export and stage comparison.
- Real Mac M4 development measurements and documented official CUDA BF16 comparison.
- Local real-weight FP32 vision diagnostic and explicit paid-cloud opt-in guard.

## Evidence and limits

Real weights run locally on a 16 GiB M4. All five synthetic positive cases pass at
both bit widths; four photo cases have qualitative review only. The missing-target
negative fails in both Mac variants **and the original CUDA BF16 model**. The
semantic evaluation therefore continues to return exit code 2.

8-bit matches 8/10 complete reference token sequences, including the false positive;
4-bit matches 0/10 exactly but still passes the positive localization thresholds.
Neither fraction is an accuracy score. Real-weight FP32 visual features closely
agree on two images; BF16 cross-backend differences remain. No standard benchmark
accuracy, general absence detection or full-model numerical parity is claimed.

This publication includes source, small synthetic examples and pinned source/header
snapshots only. No pretrained/converted weights, downloaded photos, private images,
credentials, raw logs or generated tensor dumps are included.

## Separate research track

The companion `grounding-pi-rejection-lab` explores missing-target rejection without
changing this baseline silently. Initial scope is data preparation and evaluation
planning, not a trained model. No fine-tuning benefit has been demonstrated.

Read THIRD_PARTY_NOTICES.md before redistribution. This combined project is not
blanket Apache-only; applicable Kimi K3 conditions remain in force.
