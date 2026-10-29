# macOS environment (verified)

Platform tested: macOS arm64 (Apple Silicon), Python 3.11.16 via `pyenv`.
Python 3.14 (the pyenv default here) is not used: NanoOWL/NanoSAM dependency
compatibility is unverified on it, so `pyproject.toml` pins `>=3.11,<3.12`.

## Create

```sh
pyenv install -s 3.11.16   # once per machine
pyenv local 3.11.16        # writes .python-version in the repo root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

## Verify

```sh
python -c "import sys; print(sys.executable)"   # must resolve inside .venv
pip check
smart-processing --version
pytest
```

## Update

```sh
source .venv/bin/activate
pip install -e ".[dev]" --upgrade
```

## Remove

```sh
deactivate 2>/dev/null || true
rm -rf .venv
```

## Notes

- `.venv/` is project-local and git-ignored; never install into a global or
  base Python/Conda environment.
- This environment covers the macOS demo path only (issues #2, #4-#8). The
  Jetson Orin Nano / JetPack 6.2 environment is tracked separately in #12/#13
  and must preserve JetPack-provided CUDA/TensorRT instead of reinstalling
  them through this manifest.
