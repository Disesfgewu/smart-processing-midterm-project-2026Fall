# macOS inference feasibility (Issue #2)

Verified on real hardware; no mocked models or fabricated numbers.

## Platform

- macOS 26.6 (BuildVersion 25G72), arm64, Apple M5.
- Python 3.11.16 (pyenv), project-local `.venv` per `environments/macos.md` (#17).
- `torch==2.14.1`, `torchvision==0.29.1`, `transformers==5.21.0`,
  `pillow==13.0.0`, `segment-anything==1.0` — none pinned into
  `pyproject.toml` yet; this is a feasibility spike, not the package
  dependency set (adapters land in #4/#5).
- Apple Silicon `mps` backend available and used for both models
  (`torch.backends.mps.is_available() == True`).

## Backend selection and licenses

| Capability | Backend used here | License | Why |
| --- | --- | --- | --- |
| Text -> boxes | `transformers.OwlViTForObjectDetection` (`google/owlvit-base-patch32`) | Apache-2.0 (transformers); model weights under the original OWL-ViT license (Apple Sample Code / research use, see model card) | This is the **exact backbone NanoOWL wraps** (`nanoowl.owl_predictor.OwlPredictor.__init__` calls `OwlViTForObjectDetection.from_pretrained`, see `docs/third_party_apis.md`). `OwlPredictor` itself only imports TensorRT lazily, so it is macOS-importable, but running it end-to-end here uses the underlying HF classes directly since #18's submodule isn't wired into this environment yet (#4 adapts `OwlPredictor` directly). |
| Box -> masks | `segment_anything.SamPredictor` (`vit_b`, official Meta checkpoint `sam_vit_b_01ec64.pth`) | Apache-2.0 | `nanosam.utils.predictor.Predictor` imports `tensorrt`/`torch2trt` at **module level** and cannot be imported on macOS at all (confirmed in #18). NanoSAM's own engine-build docs reference a MobileSAM checkpoint from the same SAM/TinyViT family; the official `segment-anything` PyPI package with the smallest public checkpoint (`vit_b`, ~375MB) is used here as the explicit macOS substitute. This is **not** nanosam and is documented as a substitute, not evidence of nanosam or TensorRT performance. |

Neither substitute claims to run on, or represent, Jetson/TensorRT. No
engine was built; no TensorRT import was attempted on macOS.

## Real detection run

Command: `.venv/bin/python scripts/owlvit_smoke_test.py tests/data/sample_cats.jpg`

Image: COCO val2017 `000000039769.jpg` (CC BY 4.0, the same image used in
upstream HF OWL-ViT docs/examples), two cats and two remote controls.

Actual output:

```
device=mps torch=2.14.1
model load: 2.07s
inference: 0.248s
image=tests/data/sample_cats.jpg size=(640, 480) prompts=['a photo of a cat', 'a remote control']
  label='a remote control' score=0.257 box_xyxy=[40.0, 72.4, 177.8, 115.6]
  label='a remote control' score=0.232 box_xyxy=[335.7, 74.2, 371.9, 187.5]
  label='a photo of a cat' score=0.707 box_xyxy=[325.0, 20.4, 640.6, 373.3]
  label='a photo of a cat' score=0.717 box_xyxy=[1.5, 55.3, 315.6, 472.2]
```

(First run's model load was 283.52s — first-time Hugging Face Hub download
over an unauthenticated connection; the 2.07s above is a warm local-cache
load. Both are real, not fabricated.)

## Real box-prompted segmentation run

Command: `.venv/bin/python scripts/sam_box_segment_smoke_test.py tests/data/sample_cats.jpg tests/data/sam_vit_b_01ec64.pth`

Box prompt: the `[1.5, 55.3, 315.6, 472.2]` cat box from the detection run
above, fed directly into SAM's box prompt.

Actual output:

```
device=mps torch=2.14.1
model load: 0.38s
inference: 3.190s
image=tests/data/sample_cats.jpg box_xyxy=[1.5, 55.3, 315.6, 472.2]
mask shape=(480, 640) dtype=bool score=0.992
mask pixel count=49968 of 307200 (16.3%)
```

## Limitations

- This validates the macOS-feasible *backend family*, not the pinned
  NanoOWL/NanoSAM submodules end-to-end (`OwlPredictor`/`nanosam.Predictor`
  wiring is #4/#5) and not any TensorRT engine or Jetson measurement
  (#12/#13/#14).
- Model weights are not committed; `tests/data/sam_vit_b_01ec64.pth` and
  downloaded Hugging Face caches are git-ignored.
