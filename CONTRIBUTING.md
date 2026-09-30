# Contributing

## Setup

Development uses [uv](https://docs.astral.sh/uv/). Install it, then:

```bash
git clone https://github.com/elma16/beamax.git
cd beamax
uv sync --extra dev --extra kwave --extra viz-mpl --extra autodiff
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

`uv sync` creates `.venv` with the locked dependencies from `uv.lock` and
installs beamax in editable mode. Keep the environment at `.venv`; the
pre-commit hooks run `.venv/bin/python`. Run tools with `uv run <cmd>` or
activate the environment with `source .venv/bin/activate`. Add or change
dependencies with `uv add` so `uv.lock` stays in sync (CI runs
`uv lock --check`).

## Checks

```bash
uv run pytest                  # full suite
uv run pytest tests/test_gb.py # single file
uv run pytest -k "test_name"   # single test by name
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
uv run mkdocs serve
```

## Examples

- Put self-contained public examples under `examples/`; keep research,
  profiling, and data-dependent scripts outside the gallery.
- Give public scripts a module docstring, a `main()` guard, and small defaults.
- Run `uv run python tools/finalize_examples.py` to generate or update Colab
  notebooks. Mark notebook-free scripts with `Example notebook: false`.
- Run `uv run python tools/gen_examples_readme.py` after public example changes.

## Pull requests

Branch from `main`, add relevant tests, run
`uv run ruff check beamax tests tools examples` and `uv run pytest`, then open a PR against
`main`.

## Reporting issues

Open a GitHub issue with a minimal reproducer.
