# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

beamax is a JAX library for photoacoustic tomography using multiscale Gaussian beams (MSGB), following Qian & Ying (2010), "Fast Multiscale Gaussian Wavepacket Transforms and Multiscale Gaussian Beams for the Wave Equation".

## Setup and commands

The environment is managed with uv (installed via Homebrew). `.python-version` pins Python 3.12.

```bash
uv sync --extra dev --extra kwave --extra viz-mpl --extra autodiff   # creates .venv from uv.lock, beamax editable
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

Pre-commit hooks invoke `.venv/bin/python`, so the venv must live at `.venv` (uv's default). Run tools with `uv run <cmd>` or activate `.venv`; a bare `python` resolves to the user's miniforge install. Re-run `uv sync` with the same extras after dependency changes, and use `uv add` so `uv.lock` stays in sync. Inside the Claude Code sandbox, `uv` fails on its cache at `~/.cache/uv`; call `.venv/bin/<tool>` directly instead.

Prefix the commands below with `uv run`:

```bash
pytest                                   # full suite
pytest tests/test_gb.py                  # single file
pytest tests/solvers/test_fwd_solver.py::test_hybrid_downsample  # single test
pytest -k "test_name"                    # by name
ruff check beamax tests tools examples
ruff format --check beamax tests tools examples
pyright                                  # basic mode, beamax/ only
mkdocs serve                             # docs; CI runs `mkdocs build --strict`
```

Additional CI checks worth reproducing locally when relevant:

- Coverage gate: `pytest --cov=beamax --cov-fail-under=90`.
- Sharding: `XLA_FLAGS=--xla_force_host_platform_device_count=2 pytest -q tests/solvers/test_sharding_contract.py`.
- float32 path: `JAX_ENABLE_X64=0 python tools/smoke_float32.py`.
- Examples: `python tools/finalize_examples.py --check`, `python tools/gen_examples_readme.py --check`, and `python tools/run_examples.py --directory examples --include-optional --fail-fast --silent-figures`.
- Packaging: `uv lock --check` (keep `uv.lock` in sync with `pyproject.toml`). The version string lives in both `pyproject.toml` and `beamax/__init__.py` and CI asserts `beamax.__version__` in `.github/workflows/run-tests.yml` — update all three on release.

Tests missing optional deps (k-Wave, matplotlib, optax) are skipped. The k-Wave C++ binary tests are skipped on CI unless `BEAMAX_RUN_KWAVE_CPP_TESTS=1`; `BEAMAX_KWAVE_BINARY_PATH` overrides the binary location. Most test modules enable `jax_enable_x64` at import time; there is no `conftest.py`.

Ruff ignores `E501` and `F722` (jaxtyping shape strings like `Float[Array, "b d"]`).

## Architecture

The pipeline for every MSGB operator is: **decompose → threshold → trace beams → sum at sensors**.

- `beamax/geometry.py` — `Domain` (grid, spacing, sound speed/density/absorption as arrays or callables via `c_fn`, time grid generation) and `Sensor` (positions / binary mask).
- `beamax/decomposition.py` — `DyadicDecomposition`: dyadic Fourier tiling into levels and boxes. Grid sizes and box counts are heavily validated (`validate_params`).
- `beamax/transforms.py` — `MSWPT`, the multiscale wave-packet transform built on a decomposition (`forward` = analysis to coefficients, `inverse` = synthesis). `beamax/coefficients.py` handles coefficient selection including streamed top-n; `beamax/utils/coeff_index.py` maps flat coefficient indices back to (level, box, multi-index).
- `beamax/gb/` — single-beam machinery. `gb_utils.py` has the Hamiltonian `G`, `Gx`, `Gp`; `gb_solvers.py` has the ray/beam ODE integrators (`solve_hom_diag` for homogeneous media, diffrax-based `solve_ODE_*` for heterogeneous) sharing the `SolverFn`/`SolverConfig` interface; `core.py` evaluates beam fields (XLA and Pallas variants, forward and time-reversal); `pallas_kernels.py` + `pallas_config.py` are experimental GPU/TPU kernels.
- `beamax/solvers/msgb_solvers/msgb_solver.py` — `MSGBSolver` (an `eqx.Module`) exposing `forward`, `time_reversal`, `adjoint`, each with a `*_with_params` variant that also returns intermediate beam parameters. Stage logic is split into `forward_solver_utils.py`, `tr_solver_utils.py`, `adjoint_solver_utils.py`. `ShardingStrategy` handles multi-device beam sharding; `MSGBExperimentalConfig` selects per-stage kernels (`"auto"` keeps Pallas off). Sensor data is shaped `(Nt, Ns)` — time on axis 0.
- `beamax/solvers/hybrid_solver.py` — `HybridSolver` splits input into low/high-frequency parts, runs MSGB on HF and a pluggable `HybridBackend` on LF (optionally on a downsampled grid), then recombines. Backends are plain callables `(component, HybridContext) -> result`; `HybridBackend.from_beamax_solver` wraps solvers with the beamax signature. See `docs/guides/custom-low-frequency-solvers.md`.
- `beamax/solvers/kwave_solver.py` — `KWaveSolver`, optional reference/LF solver wrapping k-Wave-python with the same forward/TR/adjoint API.
- `beamax/utils/memory.py` — static memory estimation (`estimate_msgb_memory`) for choosing batch sizes against a device budget.

Import conventions: `beamax/__init__.py`, `beamax/utils/__init__.py`, and `beamax/solvers/__init__.py` use lazy `__getattr__` loading so optional dependencies (k-Wave, matplotlib) are only imported on access. New public exports must be added to the relevant `__all__` and lazy-export table. Solvers and data classes are `equinox` modules; configuration that affects compilation is marked `static=True`.

## Docs and examples

- API pages in `docs/api/` are thin mkdocstrings stubs (`::: beamax.module`); a new public module needs a stub and an entry in `mkdocs.yml` `nav` (the strict build fails otherwise). Docstrings are NumPy style with MathJax (`$...$`, `$$...$$`).
- Public examples in `examples/` need a module docstring, a `main()` guard, and small defaults. Metadata lines in the docstring (`Example notebook: false`, `Example smoke: false`, …) are parsed by `tools/example_metadata.py`. After changing examples, run `python tools/finalize_examples.py` (regenerates Colab `.ipynb` files) and `python tools/gen_examples_readme.py` (regenerates `examples/README.md`). `docs/examples/*.md` embed the scripts via `--8<--` snippets. Notebooks are stripped by `nbstripout` in pre-commit.
