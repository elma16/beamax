"""
Unit tests for thin validation / dispatch paths in ``kwave_solver`` that the
existing integration tests don't reach. Pure-Python, no k-Wave required.
"""

import numpy as np
import pytest
import jax.numpy as jnp

from beamax.geometry import Domain
from beamax.solvers import kwave_solver as kwave_solver_module
from beamax.solvers.kwave_solver import KWaveSolver


# ---------------------------------------------------------------------------
# _normalize_kwave_binary_path
# ---------------------------------------------------------------------------


def test_normalize_binary_path_missing_directory_raises(tmp_path):
    """An empty directory (no binary inside) must raise FileNotFoundError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        kwave_solver_module._normalize_kwave_binary_path(empty_dir, device="cpu")


def test_normalize_binary_path_nonexistent_file_raises(tmp_path):
    """A nonexistent explicit file path must raise FileNotFoundError."""
    bogus = tmp_path / "nope"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        kwave_solver_module._normalize_kwave_binary_path(bogus, device="cpu")


# The _metadata_marks_bad_darwin_omp and _domain_has_nonzero_absorption tests
# were removed along with the bad-Darwin-OMP guards once k-wave-python was
# pinned to >=0.6.2.


# ---------------------------------------------------------------------------
# _coerce_sensor_data_layout — full error-path matrix
# ---------------------------------------------------------------------------


class TestCoerceSensorDataLayout:
    def _mask(self, n: int) -> np.ndarray:
        m = np.zeros((4, 4))
        m.flat[:n] = 1
        return m

    def test_invalid_data_layout_string_raises(self):
        with pytest.raises(ValueError, match="Invalid data_layout"):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((2, 5)),
                self._mask(2),
                data_layout="bogus",
                op_name="test",
            )

    def test_empty_source_mask_raises(self):
        with pytest.raises(ValueError, match="at least one active"):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((2, 5)),
                np.zeros((4, 4)),
                data_layout="auto",
                op_name="test",
            )

    def test_1d_data_with_multiple_sources_raises(self):
        with pytest.raises(ValueError, match="1D data"):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((5,)),
                self._mask(2),
                data_layout="auto",
                op_name="test",
            )

    def test_1d_data_with_single_source_is_expanded(self):
        """A 1D record with Ns=1 is reshaped to (1, Nt)."""
        out = KWaveSolver._coerce_sensor_data_layout(
            np.arange(5.0),
            self._mask(1),
            data_layout="auto",
            op_name="test",
        )
        assert out.shape == (1, 5)

    def test_3d_data_raises(self):
        with pytest.raises(ValueError, match="2D"):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((2, 3, 4)),
                self._mask(2),
                data_layout="auto",
                op_name="test",
            )

    def test_explicit_ns_nt_with_wrong_rows_raises(self):
        with pytest.raises(ValueError, match="Ns="):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((5, 2)),
                self._mask(2),  # rows=5, ns=2
                data_layout="ns_nt",
                op_name="test",
            )

    def test_explicit_nt_ns_with_wrong_cols_raises(self):
        with pytest.raises(ValueError, match="Ns="):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((5, 3)),
                self._mask(2),  # cols=3, ns=2
                data_layout="nt_ns",
                op_name="test",
            )

    def test_square_auto_layout_requires_explicit_orientation(self):
        """When Ns == Nt, auto cannot determine the data orientation safely."""
        data = np.arange(4.0).reshape(2, 2)
        with pytest.raises(ValueError, match="ambiguous square"):
            KWaveSolver._coerce_sensor_data_layout(
                data,
                self._mask(2),
                data_layout="auto",
                op_name="test",
            )

    def test_auto_layout_indeterminate_raises(self):
        """Auto-inference must fail for shapes that match neither orientation."""
        with pytest.raises(ValueError, match="could not infer"):
            KWaveSolver._coerce_sensor_data_layout(
                np.zeros((7, 9)),
                self._mask(2),
                data_layout="auto",
                op_name="test",
            )


# ---------------------------------------------------------------------------
# _build_adjoint_source / _binary_name small helpers
# ---------------------------------------------------------------------------


def test_binary_name_cpu_vs_gpu():
    """The OMP binary is for CPU and the CUDA binary is for GPU."""
    assert kwave_solver_module._binary_name("cpu") == "kspaceFirstOrder-OMP"
    assert kwave_solver_module._binary_name("gpu") == "kspaceFirstOrder-CUDA"


def test_default_kwave_binary_path_returns_path_with_known_name(monkeypatch):
    """The default-path helper returns a path with the expected binary file name."""
    import types as _types

    fake_kwave = _types.SimpleNamespace(BINARY_PATH="/tmp/kwave_bin")
    monkeypatch.setitem(__import__("sys").modules, "kwave", fake_kwave)
    out = kwave_solver_module._default_kwave_binary_path("cpu")
    assert out.name == "kspaceFirstOrder-OMP"
    assert str(out).startswith("/tmp/kwave_bin")


def test_explicit_pml_options_are_honoured_and_defaults_are_safely_derived():
    domain = Domain(N=(16, 32), dx=(0.1, 0.1), c=1.0, periodic=(False, False))
    explicit = KWaveSolver(pml_inside=False, pml_size=3, backend="python")
    explicit_kwargs = explicit._kwargs_for_domain(domain)
    assert explicit_kwargs["pml_inside"] is False
    assert explicit_kwargs["pml_size"] == 3

    defaults = KWaveSolver()
    default_kwargs = defaults._kwargs_for_domain(domain)
    assert default_kwargs["pml_size"] == (7, 15)


def test_run_simulation_evaluates_callable_absorption_fields(monkeypatch):
    captured = {}

    def fake_medium(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(kwave_solver_module, "kWaveMedium", fake_medium)
    monkeypatch.setattr(
        kwave_solver_module, "kspaceFirstOrder", lambda *args, **kwargs: {"p": None}
    )
    solver = KWaveSolver(backend="python")
    monkeypatch.setattr(solver, "_create_kgrid", lambda domain, ts: object())
    domain = Domain(
        N=(4, 4),
        dx=(0.1, 0.1),
        c=lambda x: 1500.0 + 0.0 * x[..., 0],
        alpha_coeff=lambda x: 0.2 + 0.0 * x[..., 0],
        alpha_power=lambda x: 1.5 + 0.0 * x[..., 0],
        periodic=(False, False),
    )

    solver._run_simulation(
        domain,
        jnp.linspace(0.0, 0.1, 3),
        source=object(),
        sensor=object(),
    )

    assert np.asarray(captured["alpha_coeff"]).shape == domain.N
    assert np.asarray(captured["alpha_power"]).shape == domain.N


def test_adjoint_applies_appendix_b_source_and_terminal_scalings(monkeypatch):
    """The wrapper should expose the Euclidean transpose, not raw p_final."""
    solver = KWaveSolver(backend="python", smooth_p0=False)
    domain = Domain(
        N=(3, 2),
        dx=(0.1, 0.1),
        c=2.0,
        density=3.0,
        periodic=(False, False),
    )
    source_mask = np.zeros(domain.N)
    source_mask[0, :] = 1
    captured = {}

    def fake_run(domain_arg, ts, source, sensor, *, force_python=False):
        captured["source"] = source
        captured["force_python"] = force_python
        return {"p_final": np.full(domain_arg.N, 120.0)}

    monkeypatch.setattr(solver, "_run_simulation", fake_run)
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = solver.adjoint(
        data,
        domain,
        np.ones(domain.N),
        source_mask,
        np.array([0.0, 0.01, 0.02]),
        data_layout="ns_nt",
    )

    beta = np.array([[3.0, 5.0, 4.0], [6.0, 11.0, 13.0]])
    # rho*c*dx/(4*dt) = 3*2*0.1/(4*0.01) = 15.
    np.testing.assert_allclose(captured["source"].p, 15.0 * beta)
    assert captured["source"].p_mode == "additive-no-correction"
    assert captured["force_python"] is True
    # p_final/(c^2*rho) = 120/(4*3) = 10.
    np.testing.assert_allclose(out, 10.0)


def test_adjoint_rejects_anisotropic_grid_spacing():
    solver = KWaveSolver(backend="python", smooth_p0=False)
    domain = Domain(
        N=(3, 2),
        dx=(0.1, 0.2),
        c=2.0,
        periodic=(False, False),
    )
    mask = np.ones(domain.N)
    with pytest.raises(NotImplementedError, match="isotropic"):
        solver.adjoint(
            np.ones((mask.size, 3)),
            domain,
            mask,
            mask,
            np.array([0.0, 0.01, 0.02]),
            data_layout="ns_nt",
        )


def test_adjoint_rejects_nonlinear_restore_max_smoothing():
    solver = KWaveSolver(backend="python", smooth_p0=True)
    domain = Domain(
        N=(3, 2),
        dx=(0.1, 0.1),
        c=2.0,
        periodic=(False, False),
    )
    mask = np.ones(domain.N)
    with pytest.raises(ValueError, match="smooth_p0=False"):
        solver.adjoint(
            np.ones((mask.size, 3)),
            domain,
            mask,
            mask,
            np.array([0.0, 0.01, 0.02]),
            data_layout="ns_nt",
        )


if __name__ == "__main__":
    pytest.main([__file__])
