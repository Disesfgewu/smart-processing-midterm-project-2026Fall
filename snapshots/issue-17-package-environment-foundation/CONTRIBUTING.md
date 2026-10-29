# Contributing

## Delivery order

Build a working macOS demo first. Then deploy NanoOWL and NanoSAM on Jetson
Orin Nano with JetPack 6.2 and collect final assignment results there.
Label substitute macOS backends and results explicitly. Never invent hardware evidence.

## Issues, branches, and commits

- Start from an assigned GitHub issue and respect its dependencies and acceptance criteria.
- Use `issue-<number>-<short-description>` branches; never work directly on `main`.
- Keep each commit and the entire PR at **300 added plus deleted lines or fewer**.
  This includes documentation, tests, generated text, and configuration.
- Split larger work into coherent, independently reviewable PRs. Do not split
  commits merely to evade the total PR limit or hide code in binary files.
- Use descriptive commits: `feat: ...`, `fix: ...`, `test: ...`, `docs: ...`,
  `build: ...`, or `ci: ...`.
- A necessary size exception requires Disesfgewu to comment exactly
  `/approve-size <full-current-head-sha>` on the PR, after reviewing the author's
  explanation. Re-run the size check after approval. New commits invalidate it.

## Repository and environments

- Maintain one installable Python package under `src/`, configured by `pyproject.toml`.
- Put reusable logic in package modules, with explicit detection, segmentation,
  and pipeline interfaces. Keep CLI and setup scripts thin.
- Use `tests/`, `configs/`, `docs/`, and `scripts/` for their named purposes.
  Create directories only when needed; avoid speculative scaffolding.
- Use a project-local `.venv` or named Conda environment with an explicitly
  selected Python 3 version; never install project dependencies globally.
- Keep platform environment definitions in `environments/`. Document named Conda
  environment creation, updates, removal, platform support, and dependency versions.
- Preserve JetPack-provided CUDA/TensorRT compatibility. Do not install into Conda
  base or claim untested Jetson environments are verified.
- Keep model weights, engines, local environments, captures, logs, caches, and raw
  generated outputs out of Git. Use configurable artifact locations and ignore rules.
- Track only curated submission evidence and small licensed test fixtures.
- Keep scratch experiments outside tracked source. Promote useful logic into the
  package; do not accumulate one-off experiment files or redundant implementations.
- Track upstream NanoOWL and NanoSAM as pinned Git submodules under `third_party/`.
  Document URLs, commit SHAs, licenses, initialization and update commands.
  Do not edit upstream source in place. Connect public APIs through package adapters;
  document input/output contracts and any macOS substitutes explicitly.

## Roles

- **PM-Reviwer**: this project's PM reviewer (role spelling requested by Owner).
  Prefix review comments with `[PM-Reviwer]`; record verdict, exact head SHA,
  test/CI evidence, findings and next step. Never claim another review as your own.
- **Co-Reviewer**: independent second reviewer; use `[Co-Reviewer]` and record
  a separate verdict and evidence for the same head. Identity is assigned by Owner.
- **Owner**: Disesfgewu; final review, size-exception approval and merge authority.
- Request Owner review after PM Pass; obtain the independent Co-Reviewer verdict
  before final merge. A role label does not create a GitHub identity or approval.
- Authors cannot provide independent approval of their own changes. PM-authored
  changes require Co-Reviewer and Owner review.

## Pull request format

README.md is the operator-facing source of truth. Every PR changing installation,
dependencies, configuration, CLI/API usage, model acquisition or output locations
must update the relevant README instructions in the same PR. Record only commands
actually tested on the stated platform; mark incomplete features and hardware
verification as pending. Include prerequisites, environment activation, submodule
initialization, model setup, launch/stop commands, expected outputs and relevant
troubleshooting as those capabilities become available. PM checks documentation
against implementation and verification evidence; stale or invented instructions
block Pass. README changes require Owner review through CODEOWNERS protection.

Use the PR template. Describe the concrete behavior, reference `Refs #<issue>`,
list relevant verification commands and results, and identify missing hardware
evidence. Avoid automatic closing keywords until all issue criteria are met.
Include environment/configuration details for model and performance results.
Never commit credentials or private data.

## Review and completion

1. PM reviews every PR at its exact head SHA, checks issue scope and repository
   hygiene, runs meaningful tests, and checks CI. Missing required evidence blocks Pass.
2. A Pass requests review from **Disesfgewu**. Only Disesfgewu authorizes merging.
   Any new commit invalidates the prior PM Pass.
3. After the owner merges, PM verifies the linked acceptance criteria before closing
   the issue. Partial implementation does not close an issue.
4. PM deletes the merged feature branch only after checking for active dependent
   PRs and unmerged work. Never delete the default branch.

The `Change size / size` check must be required by branch protection. CI passing
does not replace PM review or owner approval. Changes to governance or CI require
owner review and must not weaken these rules to make a PR pass.

Initial activation: the owner merges these governance files first; the workflow
then runs from the trusted base branch and never executes PR code. Configure main
branch protection to require `size`, code-owner review, dismissal of stale
approvals, and no force pushes/deletion. Do not enable a required check before
its initial workflow is available on main. Owner size approval requires a rerun.
