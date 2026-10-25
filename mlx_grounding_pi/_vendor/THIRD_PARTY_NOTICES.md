# Third-party notices

New project contributions are offered under the Apache License 2.0 (`LICENSE`).
That grant does not override third-party rights or upstream restrictions.

## GroundingPI

Source: https://github.com/groundingpi/GroundingPI
Pinned revision: `090932430bbbeea004cfa461767ede7f563c0f3a`.

- Original files are retained in `upstream/github/`.
- `mlx_grounding_pi/_vendor/protocol.py` is an unmodified copy of
  `grounding_pi/protocol.py`, under the upstream Apache License 2.0.
- Full upstream terms: `upstream/github/LICENSE` and
  `upstream/github/THIRD_PARTY_NOTICES.md`.

## GroundingPI model package / Moonshot Kimi K3 vision architecture

Source: https://huggingface.co/GroundingPI/GroundingPI
Pinned revision: `08d397c8769f7f10b96ff6d56ee6f56acb04181f`.

- Audited reference files are in `upstream/hf/`.
- `mlx_grounding_pi/_vendor/media_utils.py` is an unmodified copy of the pinned
  model package's pixel utilities. It is conservatively treated as Kimi-derived.
- `vision.py`, `projector.py`, and the visual composition in `model.py` implement
  the architecture described by those references. Treat applicable derivative
  material as subject to Kimi K3 upstream terms, not Apache-only.
- Preserve the copyright: Copyright 2025–2026 The Moonshot AI Team and HuggingFace
  Inc. team. All rights reserved (as stated in the reference vision source).
- Complete model-package license scope and texts are retained at
  `upstream/hf/LICENSE`, `LICENSE-Apache-2.0`, and `LICENSE-Kimi-K3`.
- Copies of these texts and this notice are also included in the Python package's
  `_vendor/` directory for wheel distribution.

The Kimi K3 license includes conditional commercial Model-as-a-Service and
product-attribution requirements. Review the complete license for your use case;
this notice is attribution, not legal advice or a claim of unrestricted licensing.

## MLX / mlx-lm

The language backbone is imported from `mlx_lm.models.qwen3`; its source is not
copied into this package. MLX and mlx-lm remain under their respective upstream
licenses, distributed by their installed packages.

## Reproducibility

`upstream/manifest.json` records original URLs, pinned revisions and SHA-256
hashes. No pretrained model weights are redistributed in this repository.
