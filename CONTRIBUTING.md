# Contributing

Thanks for your interest in contributing to beamax!

## Development setup

```bash
git clone https://github.com/elma16/beamax.git
cd beamax
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,kwave,viz]"
tools/install-hooks.sh
```

## Running tests

```bash
pytest                  # full suite
pytest tests/test_gb.py # single file
pytest -k "test_name"   # single test by name
```

Some tests require optional dependencies such as k-Wave. These are
skipped automatically if the packages are not installed.
The k-Wave C++ OMP binary tests are skipped on CI by default because hosted
runners do not execute the bundled binaries reliably. Set
`BEAMAX_RUN_KWAVE_CPP_TESTS=1` to opt into those tests on a runner where the
binary has been validated.

## Code style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
- Git hooks run ruff, notebook stripping, a fast pytest subset on commit, and
  the full pytest suite before push. Install them with `tools/install-hooks.sh`
  so the tracked public-push guard is preserved.
- No maximum line length is enforced (`E501` is ignored), but keep lines reasonable.

## Documentation

The API docs are generated from docstrings via MkDocs. After installing the
development extra, run:

```bash
mkdocs serve   # live preview at http://127.0.0.1:8000
```

The navigation in `mkdocs.yml` mirrors the public modules in `beamax`.

## Pull requests

1. Create a feature branch from `main`.
2. Make your changes and add tests where appropriate.
3. Run `ruff check beamax tests tools examples` and `pytest` locally.
4. Open a PR against `main`. CI will run the full test suite.

## Examples

- Add supported, documented, self-contained examples under `examples/`.
- Keep research, profiling, data-dependent, or dependency-heavy material outside
  the tracked public gallery; local/private example directories are skipped by
  the example tooling.
- Public scripts should have a module docstring, a `main()` guard, small default
  problem sizes, concise printed metrics, and a paired notebook.
- Run `python tools/finalize_examples.py` and `python tools/gen_examples_readme.py`
  after changing public examples.

## Reporting issues

Please open a GitHub issue with a minimal reproducible example where possible.
