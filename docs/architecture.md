# GroundingPI source audit

## Pinned evidence

- HF revision: `08d397c8769f7f10b96ff6d56ee6f56acb04181f`
- GitHub revision: `090932430bbbeea004cfa461767ede7f563c0f3a`
- Individual URLs and SHA-256 hashes: `upstream/manifest.json`.
- Source/config only; no checkpoint shards downloaded or custom model code executed.

## Verified checkpoint facts

`model.safetensors.index.json` reports **4,843,664,384 parameters**,
**9,687,328,768 bytes**, **569 tensors**, **2 shards**. Thus the complete multimodal
checkpoint is about 4.84B parameters, not exactly 4B.

Text: Qwen3, hidden=2560, layers=36, heads=32, KV heads=8, head dimension=128,
MLP=9728, vocabulary=152670. `rope_parameters.rope_theta=5000000`.
Do not infer head dimension from hidden size / head count. Current local
`mlx-lm 0.32.2` Qwen3 accepts `input_embeddings` and nested `rope_parameters`.

Vision: 27 layers; hidden=1024; 12 heads; QKV internal size=1536 (128/head);
MLP=4096; patch=14; no QKV/output/MLP/patch-convolution biases.

## Actual active forward path

The checkpoint's active classes are `GroundingPIVisionModel` and
`GroundingPITwoLayerProjector` in `modeling_groundingpi.py`, using
`MoonViT3dPretrainedModel` in `modeling_groundingpi_vision.py`.
That latter file also includes **unused legacy projector and backbone classes**;
porting those instead would produce an incorrect connector.

For a single image with patch grid `(1,H,W)`:

1. Input is **already patchified**, `[H*W,3,14,14]`, row-major.
2. Conv2d 3->1024, kernel/stride 14, produces `[H*W,1024]`.
3. Add learned `[64,64,1024]` spatial positions, bilinearly interpolated to `(H,W)`
   with PyTorch's default `align_corners=False`. `divided_fixed` does **not** mean
   the spatial weights are fixed. For t=1 no temporal sinusoid is added.
4. 27 pre-RMSNorm residual blocks. Attention is **noncausal**, bidirectional within
   each grid; different images/grids must not attend to one another. QKV projection
   1024->4608, then Q/K/V each 1536, output projection 1536->1024.
5. Two-dimensional rotary embedding rotates adjacent real pairs; complex frequency
   axes alternate **x then y**, despite ambiguous nearby upstream comments. Theta
   is 10000; dimension groups use powers `4*i/head_dim`.
6. Vision MLP uses **tanh-approximate GELU**, final tower norm is RMSNorm (1e-6).
7. `sd2_tpool` groups row-major 2x2 neighbors into `[H/2*W/2,4,1024]`.
   It averages the temporal axis only, **not** the four spatial neighbors.
8. Projector: LayerNorm over each 1024-channel patch (epsilon=1e-5, affine weight
   **and bias**), flatten four neighbors ->4096, bias-free Linear 4096->4096,
   **exact GELU**, bias-free Linear 4096->2560, RMSNorm (1e-5).
9. Replace image placeholder embeddings with projected features, then ordinary
   Qwen3 causal text decoding. Image token count must match features exactly.

For 448x448, H=W=32: 1024 patches ->256 image embeddings. But **448 is not a fixed
processor output size**: the actual processor dynamically resizes and pads.

## Processor contract

Pinned `preprocessor_config.json`: input patch limit=4096; 2x2 merge; max output
image tokens=1024; per-side patch limit=512; channel mean/std all 0.5.
The resize algorithm never upscales, resizes bicubically, then zero-pads the
right/bottom to multiples of 28. Because padding may exceed the token budget,
`get_resize_config` loops with a reduced patch limit until tokens fit.
Transparent images are composited **after resizing** onto the specified checkerboard.
Normalization is `(uint8/255 - 0.5)/0.5`, then row-major patchification.
The runtime `IMAGE_MAX_TOKEN_NUM` setting can further reduce the token budget.

Chat uses the saved template and expands each `<|image_pad|>` to the number of
merged image tokens. Do not substitute the K3 `make_image_prompt()` helper for the
released checkpoint's actual Qwen image placeholders.

## Coordinate protocol

- `<0>` through `<999>` are token IDs **151669 through 152668** (contiguous).
- Image token=151655; EOS=151645; vision start/end=151652/151653.
- Both points and boxes use `<|box_start|>...<|box_end|>`.
- Label wrapper: `<|object_ref_start|>label<|object_ref_end|>`.
- Points contain 2 coordinates, boxes 4; multiple tuples are comma-separated.
- Absent target is `None`. Truncated or malformed output is invalid, not `(0,0)`.
- Reuse the official dependency-free protocol parser (vendored with attribution)
  instead of matching arbitrary digit strings.
- Mapping normalized coordinates to displayed original-image pixels is a client
  convention, not recovered camera geometry; no depth or robot pose is implied.

## License

The downloaded model package has Apache and Kimi K3 components. Kimi K3 includes
conditional commercial/service and attribution provisions. Keep both complete
texts and provenance. Do not describe the full port/model as unconditionally
Apache-only. Legal review is still required before redistribution/service launch.
