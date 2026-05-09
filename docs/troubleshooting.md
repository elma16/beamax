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

## `k-wave-python` fails with `libhdf5.310.dylib` missing (macOS)

Some k-Wave binaries are linked against `libhdf5.310`, while Homebrew now ships `libhdf5.320`.

`beamax.solvers.KWaveSolver` applies a compatibility shim at runtime for this mismatch. If you still see the error:

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
