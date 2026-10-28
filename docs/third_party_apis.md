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

- `OwlPredictor(model_name="google/owlvit-base-patch32", device="cuda", ...)` —
  `device` **defaults to `"cuda"`** and is applied via `.to(self.device)` to
  the model, box bias and mesh grid; callers must pass `device="cpu"` (or
  `"mps"` if supported) explicitly on macOS or construction fails/targets CUDA.
- `encode_text(text: List[str]) -> OwlEncodeTextOutput` — CLIP/OWL-ViT text
  embeddings for one or more text prompts.
- `encode_image(image: torch.Tensor) -> OwlEncodeImageOutput` — dispatches to
  `encode_image_torch` or `encode_image_trt` depending on whether a TensorRT
  image-encoder engine was loaded at construction. `encode_image_torch`'s
  `pred_boxes` are in **normalized (image-fraction) units, not pixel
  coordinates**: center/size components come from a sigmoid so they lie in
  `[0, 1]`, but after center-to-corner conversion the resulting `x0/y0/x1/y1`
  are not guaranteed to stay within `[0, 1]`.
- `encode_rois(image, rois, pad_square=True) -> OwlEncodeImageOutput` — crops
  and re-encodes each ROI, then rescales `pred_boxes` from normalized to
  **global pixel `xyxy`** via `_owl_box_roi_to_box_global(boxes, rois)`.
- `predict(image: PIL.Image, text, text_encodings, threshold=0.1, pad_square=True) -> OwlDecodeOutput` —
  the high-level entry point: builds one ROI covering the whole image, calls
  `encode_rois` (so `decode`'s output boxes end up in pixel `xyxy`), then `decode`.
- `decode(image_output, text_output, threshold=0.1) -> OwlDecodeOutput` with
  fields `boxes` (`[N, 4]`, same coordinate space as `image_output.pred_boxes`
  — pixel `xyxy` only if `image_output` came from `encode_rois`/`predict`,
  otherwise normalized image-fraction units, not necessarily within
  `[0, 1]`), `scores` (`[N]`), `labels` (`[N]`, index
  into the input text list), `input_indices`.
- **Platform note**: `OwlPredictor` itself has no module-level TensorRT
  import — `import tensorrt as trt` / `torch2trt.TRTModule` are imported
  lazily inside `load_image_encoder_engine`, so the module *imports* on
  macOS as long as a TensorRT engine is never loaded. `predict`/`encode_rois`
  with an explicit non-CUDA `device` is the candidate macOS-feasible path;
  it also calls `torchvision.ops.roi_align` inside `extract_rois`, whose
  macOS CPU/MPS device support must still be checked — both are pending
  validation in #2, not yet validated by this issue.

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
  only, with no torch-only fallback path (unlike `OwlPredictor` above). Any
  macOS substitute backend (#2) must not import this module; its exact
  substitute implementation is chosen and documented where it's implemented
  (#2/#5), not prescribed here.

## Model/checkpoint acquisition

Neither submodule vendors model weights, but acquisition differs by path:
- `OwlPredictor`'s **torch-only path** (`encode_image_torch`/`encode_rois`/
  `predict`, `image_encoder_engine=None` by default) needs no manual
  download or engine build — `OwlViTForObjectDetection.from_pretrained(...)`
  fetches weights from the Hugging Face Hub automatically on first use.
- `OwlPredictor`'s **TensorRT image-encoder path**, and `nanosam.Predictor`
  (which has no other path), both require a manual export step (documented
  in each submodule's own `README.md`) plus a TensorRT engine build via
  `trtexec`/the provided export scripts — Jetson-only.

Weights and built `.engine`/`.onnx` files must never be committed; they
belong under a configurable, git-ignored artifact directory (finalized in #17).

## Dependencies (platform-specific)

- `third_party/nanoowl`: `torch`, `transformers` (OWL-ViT backbone); `tensorrt`
  + `torch2trt` only required for the TensorRT engine path.
- `third_party/nanosam`: `tensorrt` + `torch2trt` at import time (always
  required to even `import` the predictor module), plus `torch`, `numpy`,
  `Pillow`.
