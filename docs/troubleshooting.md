# Troubleshooting

## `ValueError: 0th dimension of all xs should be replicated`

This is usually caused by importing an older `beamax` package from `site-packages` instead of the current source checkout.

Check:

```bash
python -c "import beamax; print(beamax.__file__)"
```

Fix:

```bash
pip uninstall -y beamax
pip install -e . --no-build-isolation
```

Then re-run the import-path check and confirm it points to `src/beamax`.

## `k-wave-python` on macOS

Beamax's `[kwave]` extra requires `k-wave-python>=0.6.2`, which includes the `binary_path` fix and the `v1.4.1` macOS OMP binary.

The macOS C++ binary is Apple Silicon (`arm64`) only. Intel Mac users should use `backend="python"` until upstream ships universal2 coverage.

Older Darwin OMP binaries, especially `v0.3.0rc3` and the bad `v1.4.0` release asset, can silently mishandle power-law absorption. `beamax.solvers.KWaveSolver` rejects those known-bad binaries when absorption is enabled. To test a custom binary, set `BEAMAX_KWAVE_BINARY_PATH` to either the `kspaceFirstOrder-OMP` file or the directory containing it.

Some older k-Wave binaries are linked against `libhdf5.310`, while Homebrew now ships `libhdf5.320`. Beamax applies a compatibility shim at runtime for this mismatch. If you still see the error:

1. Confirm you are running the local `beamax` source build.
2. Reinstall editable package in your active venv.
3. Confirm HDF5 is present:

```bash
ls /opt/homebrew/opt/hdf5/lib
```

## Editable install fails with `setuptools>=61` resolution errors

When build isolation cannot access package indexes (offline/restricted network), install with:

```bash
pip install -e . --no-build-isolation
```

## Matplotlib cache/font permission warnings

If plotting from restricted environments, set writable cache paths:

```bash
MPLCONFIGDIR=/tmp/mpl MPLBACKEND=Agg python examples/forward/forward-2d.py
```
