# Contributing

Thanks for your interest in contributing to beamax!

## Development setup

```bash
git clone https://github.com/elma16/beamax.git
cd beamax
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,kwave,viz]"
pre-commit install
```

## Running tests

```bash
pytest                  # full suite
pytest tests/test_gb.py # single file
pytest -k "test_name"   # single test by name
```

Some tests require optional dependencies (`k-wave-python`, `h5py`). These are
skipped automatically if the packages are not installed.

## Code style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Pre-commit hooks run ruff, notebook stripping, and a fast pytest subset automatically.
- No maximum line length is enforced (`E501` is ignored), but keep lines reasonable.

## Pull requests

1. Create a feature branch from `main`.
2. Make your changes and add tests where appropriate.
3. Run `ruff check src tests` and `pytest` locally.
4. Open a PR against `main`. CI will run the full test suite.

## Reporting issues

Please open a GitHub issue with a minimal reproducible example where possible.
