# Text-prompted detection and segmentation

Midterm project: detect objects from a text prompt, pass their bounding boxes to
segmentation, and display boxes and masks on a live camera image.

## Current status

**The application is not runnable yet.** The installable Python package
skeleton and a verified macOS development environment are established, and
macOS inference feasibility is confirmed with one real text-conditioned
detection and one real box-prompted segmentation run (see
[docs/macos_feasibility.md](docs/macos_feasibility.md)). `DetectSegmentPipeline`
connects a real `TextDetector` + `BoxSegmenter` pair end to end as a library
API (`src/smart_processing/pipeline.py`), verified with real models (see
`tests/test_pipeline_real.py`) -- but there is still no `smart-processing`
CLI command to run it, no manually-verified camera application (the
prompt-swap/failure-handling logic is unit-tested without hardware; see
`docs/camera_demo.md`), and no Jetson-measured result.

The implementation plan is tracked in
[GitHub issues](https://github.com/Disesfgewu/smart-processing-midterm-project-2026Fall/issues).
Contribution rules and the change-size CI workflow are merged into `main`.
Branch protection on `main` is active: the `size` status check is required
(strict), code-owner review is required with stale-approval dismissal,
admins are enforced, and force pushes/deletion are disabled (verified via
the GitHub branch protection API). The `size` check itself is currently
unable to run on GitHub's hosted runner due to an account billing issue,
separate from this branch-protection configuration.

## Intended workflow

1. Build and verify a macOS demo with a supported local inference backend.
2. Accept text prompts and camera frames, detect bounding boxes, and generate masks.
3. Change prompts without restarting and demonstrate manufacturing parts counting.
4. Migrate to Jetson Orin Nano with JetPack 6.2, NanoOWL and NanoSAM TensorRT engines.
5. Collect final performance and prompt-comparison evidence on the Jetson.

macOS substitutes and measurements will be identified explicitly. They are not
evidence of Jetson execution or performance.

## Get the repository

```sh
git clone --recurse-submodules https://github.com/Disesfgewu/smart-processing-midterm-project-2026Fall.git
cd smart-processing-midterm-project-2026Fall
```

The upstream NanoOWL/NanoSAM submodules are planned in issue #18; they are
not registered on `main` yet. Do not install NanoOWL/NanoSAM globally or
assume their TensorRT paths run on macOS.

## Environment setup (macOS, verified)

Prerequisite: `pyenv` installed, and you are inside the cloned repository
(see "Get the repository" above).

```sh
pyenv install -s 3.11.16
pyenv local 3.11.16
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

Verify:

```sh
python -c "import sys; print(sys.executable)"   # resolves inside .venv
pip check
smart-processing --version
pytest
```

All of the above have been run on macOS arm64 with Python 3.11.16 and pass.
`environments/macos.md` documents update/removal commands and why Python
3.11 is pinned instead of this machine's default Python 3.14 (model
compatibility there is unverified).

## Setup and run availability

| Step | Status / tracking issue |
| --- | --- |
| Supported Python version and `.venv` setup | **Done**; Conda path pending if needed |
| Pinned upstream submodules and dependency installation | Pending: #18 |
| macOS inference feasibility (real detection + segmentation run) | **Done**: `pip install -e ".[macos-feasibility]"`, then see `docs/macos_feasibility.md` for the checkpoint download and exact commands |
| Portable detection/segmentation data contracts (`src/smart_processing/types.py`) | **Done**: pure data/interface contracts, no model code |
| macOS text detection adapter (`OwlViTTextDetector`) | **Done**: `pip install -e ".[macos-feasibility]"`, then `RUN_FEASIBILITY_SMOKE=1 pytest tests/test_owlvit_text_detector_real.py -v` |
| macOS box-prompted segmentation adapter (`SamBoxSegmenter`) | **Done**: `RUN_FEASIBILITY_SMOKE=1 pytest tests/test_sam_box_segmenter_real.py -v` (needs the SAM checkpoint from `docs/macos_feasibility.md`) |
| Model/checkpoint download command | **Done**: see `docs/macos_feasibility.md` (manual `curl`/HF Hub fetch; `SamBoxSegmenter` takes a caller-supplied `checkpoint_path`, no built-in cache management) |
| Automated checkpoint cache configuration | Pending: later, not yet built |
| Single-image detect+segment pipeline API (`DetectSegmentPipeline`) | **Done** (this PR): see `tests/test_pipeline_real.py` for a real text->boxes->masks run saving an overlay image and structured JSON |
| `smart-processing` CLI command to run the pipeline | Pending: the `smart-processing` entry point has no subcommand for this yet; only the library API above is wired |
| Camera demo and live prompt changes | **Partial** (this PR): prompt-swap, camera-open-failure handling and clean shutdown are implemented and unit-tested; the live interactive demo itself is NOT manually verified -- camera access is not authorized in this autonomous session, see `docs/camera_demo.md` for the real verification command/output and why Owner follow-up is required |
| Manufacturing counting demo | Pending: #8 |
| Evaluation and benchmark commands | Pending: #10, #11 |
| Jetson setup, engine builds and run commands | Pending: #12, #13 |

There is currently no verified application launch command. As each feature
lands, this section must be replaced with the exact tested commands,
prerequisites, expected output, and troubleshooting steps. Do not guess
commands from the plan.

## Repository map

- `src/smart_processing/`: the installable package. `feasibility.py` holds
  the real OWL-ViT/SAM model-loading and inference logic for #2's smoke
  tests. `types.py` defines the portable `BoundingBox`/`Detection`/
  `DetectionResult`/`SegmentationMask` contracts and `TextDetector`/
  `BoxSegmenter` Protocols (#3; adds a runtime `numpy` dependency).
  `adapters/owlvit_text_detector.py` implements `TextDetector` (#4) and
  `adapters/sam_box_segmenter.py` implements `BoxSegmenter` (#5), both
  against the backends validated in #2. `pipeline.py`'s
  `DetectSegmentPipeline` composes any `TextDetector` + `BoxSegmenter` pair
  into a full text->boxes->masks run, plus `save_overlay_image`/
  `save_structured_output` (#6). `camera_demo.py` (this PR, #7) has the
  prompt-swap/camera-failure/clean-shutdown logic, unit-tested without
  hardware; see `docs/camera_demo.md` for what is and isn't manually
  verified. Evaluation and benchmarking logic lands as later issues (#8-#13).
- `tests/`: automated tests for the package.
- `scripts/`: thin CLI wrappers over `smart_processing.feasibility` (argv
  parsing + printing only; model loading/inference lives in the package).
- `docs/macos_feasibility.md`: real macOS detection/segmentation run,
  platform/version/license record and backend substitution rationale (#2).
- `environments/macos.md`: verified macOS `.venv` setup, update and removal
  commands. A documented Jetson Orin Nano / JetPack 6.2 environment strategy
  is tracked separately in #12/#13.
- `pyproject.toml`: package metadata, the `smart-processing` CLI entry point,
  and the pinned supported Python range.
- `CONTRIBUTING.md` / `AGENTS.md`: process, environment and review rules.
- `.github/`: review template, ownership and change-size automation.
- `hw.md`: assignment requirements supplied in the workspace; currently untracked.

Model weights, TensorRT engines, captured data, benchmark outputs and logs
are not tracked in Git; artifact locations will be documented as they are
introduced (#10-#14).

## Reviews and ownership

**PM-Reviwer** reviews scope, code, tests and documentation. **Co-Reviewer**
provides an independent second review. **Owner (Disesfgewu)** provides final
approval and merges. Each commit and PR is limited to 300 added plus deleted
lines by default. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full process.
