"""
Smoke tests for the optional FNO solvers.

Skipped automatically on environments without the relevant pieces of the
``[fno]`` extra installed. Exercised end-to-end by the ``fno-smoke`` CI job,
which installs ``beamax[dev,fno]``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


def test_fno_pdequinox_smoke(tmp_path):
    """Round-trip a tiny FNO through FNOpdequinoxSolver."""
    pytest.importorskip("pdequinox")
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from pdequinox.arch._classic_fno import ClassicFNO
    from beamax.solvers.fno_solver_pdequinox import FNOpdequinoxSolver

    hparams = {
        "num_modes": (4, 4),
        "num_spatial_dims": 2,
        "hidden_channels": 8,
        "in_channels": 1,
        "out_channels": 1,
        "num_blocks": 2,
    }

    key = jax.random.PRNGKey(0)
    model = ClassicFNO(**hparams, key=key)

    # JSON can't preserve tuples; round-trip via list. ClassicFNO accepts either.
    hparams_json = {**hparams, "num_modes": list(hparams["num_modes"])}
    (tmp_path / "hparams.json").write_text(json.dumps(hparams_json))
    eqx.tree_serialise_leaves(tmp_path / "model.eqx", model)

    solver = FNOpdequinoxSolver(str(tmp_path), key=key)

    # Match the model dtype: ClassicFNO builds weights in the JAX default
    # float dtype, which depends on the global x64 flag. Other tests in the
    # suite may have enabled x64, so we cast to match.
    default_float = jnp.zeros(()).dtype
    p0 = jnp.asarray(np.random.default_rng(0).standard_normal((32, 32))).astype(
        default_float
    )
    out = solver.forward(p0)

    assert out.shape == (32, 32), f"expected (32, 32), got {out.shape}"
    assert jnp.all(jnp.isfinite(out)), "FNO output contains non-finite values"


def test_fno_neurops_importable():
    """FNONeuralOpsSolver should be importable when the fno extra is present."""
    pytest.importorskip("torch")
    pytest.importorskip("neuralop")

    # Construction requires a torch checkpoint on disk; this test just smokes
    # the import-guarded module so a missing dep surfaces as a clear skip/error
    # rather than a cryptic import chain failure.
    from beamax.solvers.fno_solver_neurops import FNONeuralOpsSolver  # noqa: F401
