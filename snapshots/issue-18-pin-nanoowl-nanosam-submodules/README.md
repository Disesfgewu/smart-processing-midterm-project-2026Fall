# Text-prompted detection and segmentation

Midterm project: detect objects from a text prompt, pass their bounding boxes to
segmentation, and display boxes and masks on a live camera image.

## Current status

**The application is not runnable yet.** This repository is in its setup phase.
There is no installed project environment, model integration, camera application,
or measured inference result verified for this project yet.

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

## Upstream submodules

`third_party/nanoowl` and `third_party/nanosam` are pinned Git submodules of the
official [NVIDIA-AI-IOT/nanoowl](https://github.com/NVIDIA-AI-IOT/nanoowl) and
[NVIDIA-AI-IOT/nanosam](https://github.com/NVIDIA-AI-IOT/nanosam) (both
Apache-2.0). Do not edit submodule files in place; see
[docs/third_party_apis.md](docs/third_party_apis.md) for pinned SHAs, the
update policy, and the text->boxes / box->masks API contracts and their
platform-specific (TensorRT) dependencies.

## Get the repository

```sh
git clone --recurse-submodules https://github.com/Disesfgewu/smart-processing-midterm-project-2026Fall.git
cd smart-processing-midterm-project-2026Fall
# or, on an existing checkout:
git submodule update --init --recursive
```

Verified: a fresh `git submodule update --init --recursive` resolves exactly
the pinned commits (checked on macOS arm64) — recorded as gitlinks in this
repository's tree (`git submodule status`), not literally stored in
`.gitmodules` itself (which only holds each submodule's URL and path).
Neither submodule's weights are included. Do not install NanoOWL/NanoSAM
globally or assume their TensorRT paths run on macOS.

## Setup and run availability

| Step | Status / tracking issue |
| --- | --- |
| Supported Python version and `.venv`/Conda setup | Pending: #17 and #2 |
| Pinned upstream submodules (this PR) | **Done**: `third_party/nanoowl`/`nanosam` pinned, API-mapped in `docs/third_party_apis.md` |
| Python package/dependency installation using these submodules | Pending: #2, #4, #5 (not provided by this pinning-only PR) |
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
- `third_party/`: pinned NanoOWL/NanoSAM submodules; see `docs/third_party_apis.md`.
- `hw.md`: assignment requirements supplied in the workspace; currently untracked.

The installable package, environment manifests and model adapters are planned;
their paths will be documented when they exist. Local agent artifacts are not
application source or submission evidence.

## Reviews and ownership

**PM-Reviwer** reviews scope, code, tests and documentation. **Co-Reviewer** provides
an independent second review. **Owner (Disesfgewu)** provides final approval and
merges. Each commit and PR is limited to 300 added plus deleted lines by default.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete process.
