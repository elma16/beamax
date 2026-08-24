# Contributing

## Setup

```bash
git clone https://github.com/elma16/beamax.git
cd beamax
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,kwave,viz-mpl,autodiff]"
pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Checks

```bash
pytest                  # full suite
pytest tests/test_gb.py # single file
pytest -k "test_name"   # single test by name
```

Missing optional dependencies skip their tests. Validated runners can enable
the otherwise-disabled k-Wave C++ tests with
`BEAMAX_RUN_KWAVE_CPP_TESTS=1`.

Use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Pre-commit
runs Ruff, strips notebooks, and runs fast tests; pre-push runs the full suite.
`E501` is disabled, but keep lines readable.

## Documentation

API pages are generated from docstrings with MkDocs:

```bash
mkdocs serve
```

## Examples

- Put self-contained public examples under `examples/`; keep research,
  profiling, and data-dependent scripts outside the gallery.
- Give public scripts a module docstring, a `main()` guard, and small defaults.
- Run `python tools/finalize_examples.py` to generate or update Colab
  notebooks. Mark notebook-free scripts with `Example notebook: false`.
- Run `python tools/gen_examples_readme.py` after public example changes.

## Pull requests

Branch from `main`, add relevant tests, run
`ruff check beamax tests tools examples` and `pytest`, then open a PR against
`main`.

## Reporting issues

Open a GitHub issue with a minimal reproducer.
