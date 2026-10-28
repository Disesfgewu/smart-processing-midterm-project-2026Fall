# Text-prompted detection and segmentation

Midterm project: detect objects from a text prompt, pass their bounding boxes to
segmentation, and display boxes and masks on a live camera image.

## Current status

**The application is not runnable yet.** This repository is in its setup phase.
There is no installed project environment, model integration, camera application,
or measured inference result verified for this project yet.

The implementation plan is tracked in
[GitHub issues](https://github.com/Disesfgewu/smart-processing-midterm-project-2026Fall/issues).
Contribution rules and a change-size workflow have been prepared locally; they
become repository policy/checks after review and merge. Required branch protection
must also be activated before the check blocks merges.

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

The upstream submodules are planned in issue #18; they are not registered yet.
Do not install NanoOWL/NanoSAM globally or assume their TensorRT paths run on macOS.

## Setup and run availability

| Step | Status / tracking issue |
| --- | --- |
| Supported Python version and `.venv`/Conda setup | Pending: #17 and #2 |
| Pinned upstream submodules and dependency installation | Pending: #18 |
| Model/checkpoint download and cache configuration | Pending: #2, #4, #5 |
| Single-image detection and segmentation command | Pending: #6 |
| Camera demo and live prompt changes | Pending: #7 |
| Manufacturing counting demo | Pending: #8 |
| Evaluation and benchmark commands | Pending: #10, #11 |
| Jetson setup, engine builds and run commands | Pending: #12, #13 |

There is currently no verified application launch command. As each feature lands,
this section must be replaced with the exact tested commands, prerequisites,
expected output, and troubleshooting steps. Do not guess commands from the plan.

## Repository map

- `README.md`: actual project status and verified operator instructions.
- `CONTRIBUTING.md`: implementation, environment, size and review rules.
- `AGENTS.md`: instructions for coding agents.
- `.github/`: review template, ownership and change-size automation.
- `hw.md`: assignment requirements supplied in the workspace; currently untracked.

The installable package, environment manifests and model adapters are planned;
their paths will be documented when they exist. Local agent artifacts are not
application source or submission evidence.

## Reviews and ownership

**PM-Reviwer** reviews scope, code, tests and documentation. **Co-Reviewer** provides
an independent second review. **Owner (Disesfgewu)** provides final approval and
merges. Each commit and PR is limited to 300 added plus deleted lines by default.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete process.
