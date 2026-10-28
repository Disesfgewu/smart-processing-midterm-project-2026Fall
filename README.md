# Text-prompted detection and segmentation

Midterm project: detect objects from a text prompt, pass their bounding boxes to
segmentation, and display boxes and masks on a live camera image.

## Current status

**The application is not runnable yet.** The installable Python package
skeleton and a verified macOS development environment are established (this
PR). There is no model integration, detection/segmentation logic, camera
application, or measured inference result yet.

The implementation plan is tracked in
[GitHub issues](https://github.com/Disesfgewu/smart-processing-midterm-project-2026Fall/issues).
Contribution rules and the change-size CI workflow are merged into `main`.
Required branch protection must still be activated by the Owner before the
check blocks merges.

## Intended workflow

1. Build and verify a macOS demo with a supported local inference backend.
2. Accept text prompts and camera frames, detect bounding boxes, and generate masks.
3. Change prompts without restarting and demonstrate manufacturing parts counting.
4. Migrate to Jetson Orin Nano with JetPack 6.2, NanoOWL and NanoSAM TensorRT engines.
5. Collect final performance and prompt-comparison evidence on the Jetson.

macOS substitutes and measurements will be identified explicitly. They are not
evidence of Jetson execution or performance.

## Environment setup (macOS, verified)

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

## Get the repository

```sh
git clone --recurse-submodules https://github.com/Disesfgewu/smart-processing-midterm-project-2026Fall.git
cd smart-processing-midterm-project-2026Fall
```

The upstream NanoOWL/NanoSAM submodules are planned in issue #18; they are
not registered on `main` yet. Do not install NanoOWL/NanoSAM globally or
assume their TensorRT paths run on macOS.

## Setup and run availability

| Step | Status / tracking issue |
| --- | --- |
| Supported Python version and `.venv` setup | **Done** (this PR); Conda path pending #2 if needed |
| Pinned upstream submodules and dependency installation | Pending: #18 |
| Model/checkpoint download and cache configuration | Pending: #2, #4, #5 |
| Single-image detection and segmentation command | Pending: #6 |
| Camera demo and live prompt changes | Pending: #7 |
| Manufacturing counting demo | Pending: #8 |
| Evaluation and benchmark commands | Pending: #10, #11 |
| Jetson setup, engine builds and run commands | Pending: #12, #13 |

There is currently no verified application launch command. As each feature
lands, this section must be replaced with the exact tested commands,
prerequisites, expected output, and troubleshooting steps. Do not guess
commands from the plan.

## Repository map

- `src/smart_processing/`: the installable package. Reusable detection,
  segmentation, pipeline, camera, evaluation and benchmarking logic will live
  here as it is implemented (#3-#13); only the package skeleton and a thin
  CLI entry point exist so far.
- `tests/`: automated tests for the package.
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
