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
3. Run `ruff check src tests tools examples` and `pytest` locally.
4. Open a PR against `main`. CI will run the full test suite.

## Examples

- Add supported, documented, self-contained examples under `examples/`.
- Preserve research, profiling, data-dependent, or dependency-heavy material under
  `examples/private/`; private examples are not linted, documented, or smoke-tested.
- Public scripts should have a module docstring, a `main()` guard, small default
  problem sizes, concise printed metrics, and a paired notebook.
- Run `python tools/finalize_examples.py` and `python tools/gen_examples_readme.py`
  after changing public examples.

## Reporting issues

Please open a GitHub issue with a minimal reproducible example where possible.
