# Upstream submodule API contracts

Pinned submodules, verified by a fresh `git submodule update --init --recursive`
on macOS arm64:

| Submodule | URL | Pinned commit | License |
| --- | --- | --- | --- |
| `third_party/nanoowl` | https://github.com/NVIDIA-AI-IOT/nanoowl.git | `fb553dee4c58f8c53ec2bf01f355a54648e67dbd` | Apache-2.0 |
| `third_party/nanosam` | https://github.com/NVIDIA-AI-IOT/nanosam.git | `653633614b2eb93b06ba3be9adb2aeffb117bd72` | Apache-2.0 |

Both pins are each submodule's default-branch `HEAD` at the time of pinning.
Do not edit files inside these submodules; changes must come from upstream
releases. Reviewing a submodule SHA bump means reading the upstream diff and
release notes at that SHA, not just the one-line gitlink diff in this repo.

## Update policy

```sh
cd third_party/nanoowl && git fetch && git checkout <new-sha> && cd -
git add third_party/nanoowl
git commit -m "build: bump nanoowl submodule to <new-sha>"
```

Same pattern for `third_party/nanosam`. PR description must explain *why*
(upstream fix/feature, with a link to the upstream commit/release), not just
state the SHA changed.

## Text -> boxes: `nanoowl.owl_predictor.OwlPredictor`

- `encode_text(text: List[str]) -> OwlEncodeTextOutput` — CLIP/OWL-ViT text
  embeddings for one or more text prompts.
- `encode_image(image: torch.Tensor) -> OwlEncodeImageOutput` — dispatches to
  `encode_image_torch` or `encode_image_trt` depending on whether a TensorRT
  image-encoder engine was loaded at construction.
- `decode(image_output, text_output, threshold=0.1) -> OwlDecodeOutput` with
  fields `boxes` (`[N, 4]`, `xyxy` pixel coordinates), `scores` (`[N]`),
  `labels` (`[N]`, index into the input text list), `input_indices`.
- Input image: a normalized `torch.Tensor` sized to the model's configured
  patch grid; see `owl_predictor.py`'s `encode_image_torch`/preprocessing.
- Device selection: standard `torch` device of the module/tensors; no
  separate device argument.
- **Platform note**: `OwlPredictor` itself has no module-level TensorRT
  import — `import tensorrt as trt` / `torch2trt.TRTModule` are imported
  lazily inside `load_image_encoder_engine`, so the module *imports* on
  macOS as long as a TensorRT engine is never loaded. Torch-only inference
  (`encode_image_torch`) is the macOS-feasible path; validated in #2.

## Box -> masks: `nanosam.utils.predictor.Predictor`

- `set_image(image: PIL.Image)` — runs the image encoder engine once per
  frame.
- `predict(points, point_labels, mask_input=None) -> (hi_res_mask, mask_iou, low_res_mask)`.
  A box prompt is passed as two points — top-left and bottom-right corners —
  with SAM's standard point labels `2` and `3`; `points` are in original
  image pixel coordinates and rescaled internally by `preprocess_points`.
- **Platform note**: `nanosam/utils/predictor.py` imports `tensorrt` and
  `torch2trt.TRTModule` at **module level**, so this module cannot be
  imported at all without a TensorRT installation — it is Jetson/TensorRT
  only. Any macOS substitute backend (#2) must not import this module; it
  needs its own adapter using the vendored MobileSAM/ONNX weights, not a
  duplicate implementation here (see #4/#5/#13).

## Model/checkpoint acquisition

Neither submodule vendors model weights. Both require a manual download step
(documented in each submodule's own `README.md`) plus a TensorRT engine
build via `trtexec`/the provided export scripts — Jetson-only. Weights and
built `.engine`/`.onnx` files must never be committed; they belong under a
configurable, git-ignored artifact directory (finalized in #17).

## Dependencies (platform-specific)

- `third_party/nanoowl`: `torch`, `transformers` (OWL-ViT backbone); `tensorrt`
  + `torch2trt` only required for the TensorRT engine path.
- `third_party/nanosam`: `tensorrt` + `torch2trt` at import time (always
  required to even `import` the predictor module), plus `torch`, `numpy`,
  `Pillow`.
