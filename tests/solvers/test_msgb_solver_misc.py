"""
Tests for thin error/dispatch paths in MSGBSolver and ShardingStrategy.

Focus: parts of `msgb_solver.py` that don't show up in the existing forward,
time-reversal, or sharding integration tests. In particular:

- `ShardingStrategy._beam_sharding_spec` input validation,
- `MSGBSolver._infer_planar_surface` happy + failure paths,
- `MSGBSolver.forward` / `.time_reversal` / `.adjoint` sensor + periodicity
  validation,
- `MSGBSolver.solve_ivp` (thin wrapper around `forward`),
- explicit `*_with_params` diagnostic variants,
- A small `adjoint` smoke test exercising the full
  `_prepare_adj_params → compute_TR_result` path.
"""

import os

# Force two host devices so we can build a Mesh and exercise sharding-spec helpers
# even on a single-CPU CI runner.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import pytest

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

from beamax import geometry
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import gb_solvers
from beamax.solvers.msgb_solvers import msgb_solver as msgb_solver_module
from beamax.solvers.msgb_solvers.msgb_solver import (
    MSGBSolver,
    _apply_adjoint_image_weight,
    _form_adjoint_source,
)
from beamax.solvers import ShardingStrategy

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _c_const(x):
    return 1.0 + 0.0 * x[..., 0]


@pytest.fixture(scope="module")
def two_device_mesh():
    """A 2-device CPU mesh for testing ShardingStrategy helpers."""
    devices = jax.devices()[:2]
    if len(devices) < 2:
        pytest.skip("requires at least 2 devices")
    return jax.make_mesh((len(devices),), ("x",))


@pytest.fixture
def sharding_strategy(two_device_mesh):
    return ShardingStrategy(mesh=two_device_mesh, beam_axis="x")


@pytest.fixture
def simple_solver():
    """Minimal MSGBSolver used by tests that don't need a real forward pass."""
    return MSGBSolver(
        thr=4,
        thr_strat="top_n",
        batch_size=4,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        sum_method="all_real",
    )


# ---------------------------------------------------------------------------
# ShardingStrategy._beam_sharding_spec input validation
# ---------------------------------------------------------------------------


class TestBeamShardingSpec:
    def test_scalar_input_rejected(self, sharding_strategy):
        """Sharding a scalar (ndim=0) array is undefined and must raise."""
        with pytest.raises(ValueError, match="scalar"):
            sharding_strategy._beam_sharding_spec(0, is_batched=False)

    def test_batched_with_only_one_axis_rejected(self, sharding_strategy):
        """A batched tensor must have at least two axes (batch + beam)."""
        with pytest.raises(ValueError, match="at least two"):
            sharding_strategy._beam_sharding_spec(1, is_batched=True)

    def test_batched_returns_fully_replicated(self, sharding_strategy):
        """Batched tensors are kept replicated so scan/vmap stay simple."""
        spec = sharding_strategy._beam_sharding_spec(3, is_batched=True)
        assert isinstance(spec, PartitionSpec)
        # Every axis should be unsharded (None).
        assert tuple(spec) == (None, None, None)

    def test_unbatched_shards_beam_axis(self, sharding_strategy):
        """Unbatched tensors are sharded on the configured beam axis only."""
        spec = sharding_strategy._beam_sharding_spec(2, is_batched=False)
        assert tuple(spec) == ("x", None)


# ---------------------------------------------------------------------------
# MSGBSolver._infer_planar_surface
# ---------------------------------------------------------------------------


class TestInferPlanarSurface:
    def test_planar_xline_returns_surface_axis_and_coord(self, simple_solver):
        """Sensors lying on a constant-y line should be detected as a planar surface."""
        # Five sensors along x=2.0, varying y => axis 0 is "constant".
        y_vals = jnp.linspace(0.0, 1.0, 5)
        positions = jnp.stack([jnp.full_like(y_vals, 2.0), y_vals], axis=1)

        surface, axis, coord = simple_solver._infer_planar_surface(positions)
        assert axis == 0
        assert float(coord) == pytest.approx(2.0)

        # The returned surface function should give zero on the constant-coord
        # axis and a nonzero residual off it.
        on = surface(jnp.array([2.0, 0.5]))
        off = surface(jnp.array([3.0, 0.5]))
        assert float(on) == pytest.approx(0.0)
        assert float(off) == pytest.approx(1.0)

    def test_nonplanar_sensors_raise(self, simple_solver):
        """Generic / non-coplanar sensor positions must raise ValueError."""
        positions = jnp.array(
            [[0.0, 0.0], [1.0, 0.5], [0.3, 0.8], [0.7, 0.2]],
            dtype=jnp.float64,
        )
        with pytest.raises(ValueError, match="planar surface"):
            simple_solver._infer_planar_surface(positions)


# ---------------------------------------------------------------------------
# Sensor-type / periodicity validation in forward, time_reversal, adjoint
# ---------------------------------------------------------------------------


def _build_small_1d_setup(periodic=False):
    """Tiny 1D MSWPT setup usable by validation-error tests."""
    N = (128,)
    domain = geometry.Domain(N=N, dx=(1.0 / N[0],), c=_c_const, periodic=(periodic,))
    decomp = DyadicDecomposition(
        num_levels=2, N=N, num_boxes_levels=(4, 8), box_aspect_ratio=(1,)
    )
    wpt = MSWPT(decomp, redundancy=2, windowing="rectangular")
    return domain, wpt


class TestSensorAndPeriodicityValidation:
    def test_forward_rejects_unsupported_sensor_type(self, simple_solver):
        """Passing something that's neither a Sensor nor a jnp.ndarray must raise."""
        domain, wpt = _build_small_1d_setup(periodic=False)
        p0 = jnp.zeros(domain.N)
        ts = jnp.linspace(0.0, 1.0, 8)
        with pytest.raises(ValueError, match="Unsupported sensor type"):
            simple_solver.forward(p0, domain, "not_a_sensor", ts, wpt)

    def test_time_reversal_rejects_periodic_domain(self, simple_solver):
        """TR explicitly forbids periodic spatial boundaries."""
        domain_periodic, wpt_p = _build_small_1d_setup(periodic=True)
        domain_data, wpt_d = _build_small_1d_setup(periodic=False)
        # Fake some data: just any (Nt, Ns) shape works because the check
        # is the first thing in time_reversal.
        data = jnp.zeros((8, 1))
        ts = jnp.linspace(0.0, 1.0, 8)
        sensor_mask = jnp.zeros(domain_data.N).at[0].set(1)
        sensors = geometry.Sensor(domain=domain_data, binary_mask=sensor_mask)
        with pytest.raises(ValueError, match="free space boundary"):
            simple_solver.time_reversal(
                data=data,
                domain=domain_periodic,
                sensors=sensors,
                sources=sensors,
                ts=ts,
                data_domain=domain_data,
                data_wpt=wpt_d,
            )

    def test_time_reversal_rejects_unsupported_sensor_type(self, simple_solver):
        """A bad `sensors` arg must raise ValueError, not a deep stack trace."""
        domain, wpt = _build_small_1d_setup(periodic=False)
        domain_data, wpt_d = _build_small_1d_setup(periodic=False)
        ts = jnp.linspace(0.0, 1.0, 8)
        data = jnp.zeros((8, 1))
        sensor_mask = jnp.zeros(domain_data.N).at[0].set(1)
        sources = geometry.Sensor(domain=domain_data, binary_mask=sensor_mask)
        with pytest.raises(ValueError, match="Unsupported sensor type"):
            simple_solver.time_reversal(
                data=data,
                domain=domain,
                sensors="bogus",
                sources=sources,
                ts=ts,
                data_domain=domain_data,
                data_wpt=wpt_d,
            )

    def test_adjoint_rejects_periodic_domain(self, simple_solver):
        """The MSGB adjoint solve does not currently support periodic boundaries."""
        domain_periodic, wpt = _build_small_1d_setup(periodic=True)
        domain_data, wpt_d = _build_small_1d_setup(periodic=False)
        data = jnp.zeros((8, 1))
        ts = jnp.linspace(0.0, 1.0, 8)
        sensor_mask = jnp.zeros(domain_data.N).at[0].set(1)
        sensors = geometry.Sensor(domain=domain_data, binary_mask=sensor_mask)
        with pytest.raises(ValueError, match="non-periodic"):
            simple_solver.adjoint(
                data=data,
                domain=domain_periodic,
                sensors=sensors,
                sources=sensors,
                ts=ts,
                data_domain=domain_data,
                data_wpt=wpt_d,
            )

    def test_adjoint_rejects_unsupported_sensor_type(self, simple_solver):
        """A non-Sensor/non-ndarray `sensors` arg must raise."""
        domain, wpt = _build_small_1d_setup(periodic=False)
        domain_data, wpt_d = _build_small_1d_setup(periodic=False)
        data = jnp.zeros((8, 1))
        ts = jnp.linspace(0.0, 1.0, 8)
        sensor_mask = jnp.zeros(domain_data.N).at[0].set(1)
        sources = geometry.Sensor(domain=domain_data, binary_mask=sensor_mask)
        with pytest.raises(ValueError, match="Unsupported sensor type"):
            simple_solver.adjoint(
                data=data,
                domain=domain,
                sensors=42,
                sources=sources,
                ts=ts,
                data_domain=domain_data,
                data_wpt=wpt_d,
            )


# ---------------------------------------------------------------------------
# solve_ivp — thin wrapper around forward
# ---------------------------------------------------------------------------


class TestSolveIvp:
    def test_solve_ivp_with_zero_dpdt_matches_forward(self):
        """solve_ivp(p0, dpdt=0) should give the same sensor data as forward(p0)."""
        domain, wpt = _build_small_1d_setup(periodic=True)
        sensors = geometry.Sensor(domain=domain, binary_mask=jnp.ones(domain.N))
        p0 = jnp.cos(2.0 * jnp.pi * jnp.arange(domain.N[0]) / domain.N[0])
        ts = jnp.linspace(0.0, 0.1, 8)

        solver = MSGBSolver(
            thr=8,
            thr_strat="top_n",
            batch_size=4,
            input_type="spatial",
            ode_solver=gb_solvers.solve_ODE_base,
            sum_method="all_real",
        )
        out_fwd = solver.forward(p0, domain, sensors, ts, wpt)
        out_ivp = solver.solve_ivp(p0, jnp.zeros_like(p0), domain, wpt, sensors, ts)

        assert jnp.allclose(out_fwd, out_ivp, atol=1e-12)


class TestDiagnosticParamVariants:
    def test_forward_with_params_matches_forward(self):
        """Diagnostic forward variant should expose params without changing data."""
        domain, wpt = _build_small_1d_setup(periodic=True)
        sensors = geometry.Sensor(domain=domain, binary_mask=jnp.ones(domain.N))
        p0 = jnp.cos(2.0 * jnp.pi * jnp.arange(domain.N[0]) / domain.N[0])
        ts = jnp.linspace(0.0, 0.1, 8)

        solver = MSGBSolver(
            thr=8,
            thr_strat="top_n",
            batch_size=4,
            input_type="spatial",
            ode_solver=gb_solvers.solve_ODE_base,
            sum_method="all_real",
        )

        sensor_data = solver.forward(p0, domain, sensors, ts, wpt)
        diagnostic_data, params = solver.forward_with_params(
            p0, domain, sensors, ts, wpt
        )

        assert jnp.allclose(sensor_data, diagnostic_data, atol=1e-12)
        assert len(params) == 6

    def test_oversized_real_top_n_matches_half_frame_cap(self):
        """Oversized requests must not propagate masked zero-amplitude rows."""
        N = (16,)
        domain = geometry.Domain(N=N, dx=(1.0 / N[0],), c=_c_const, periodic=(True,))
        decomp = DyadicDecomposition(
            num_levels=1,
            N=N,
            num_boxes_levels=(4,),
            box_aspect_ratio=(1,),
        )
        wpt = MSWPT(decomp, redundancy=2, windowing="rectangular")
        half_capacity = sum(
            (end - start) // 2
            for start, end in zip(wpt.coeffs_cumsum[:-1], wpt.coeffs_cumsum[1:])
        )
        p0 = jax.random.normal(jax.random.PRNGKey(37), N)
        sensors = geometry.Sensor(domain=domain, binary_mask=jnp.ones(N))
        ts = jnp.linspace(0.0, 0.02, 3)

        def run(threshold):
            solver = MSGBSolver(
                thr=threshold,
                thr_strat="top_n",
                batch_size=4,
                input_type="spatial",
                ode_solver=gb_solvers.solve_ODE_base,
                sum_method="all_real",
            )
            return solver.forward_with_params(p0, domain, sensors, ts, wpt)

        capped_data, capped_params = run(half_capacity)
        oversized_data, oversized_params = run(wpt.total_coeffs)

        assert capped_params[4].shape == (2 * half_capacity,)
        assert jnp.all(jnp.abs(capped_params[4]) > 0)
        assert jnp.allclose(oversized_data, capped_data, rtol=1e-12, atol=1e-12)
        for oversized, capped in zip(oversized_params, capped_params):
            assert jnp.allclose(oversized, capped, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# adjoint — smoke test the happy path (also covers _prepare_adj_params)
# ---------------------------------------------------------------------------


def test_form_adjoint_source_applies_spectral_derivative_sign_and_c_squared():
    n = 64
    dt = 0.25 / n
    mode = 5
    t = jnp.arange(n) * dt
    period = n * dt
    angular_frequency = 2.0 * jnp.pi * mode / period
    data = jnp.ones((n, 2))
    window = jnp.sin(angular_frequency * t)
    c_at_sources = jnp.array([2.0, 3.0])

    source = _form_adjoint_source(data, dt, c_at_sources, window)
    expected = (
        -(c_at_sources**2) * angular_frequency * jnp.cos(angular_frequency * t)[:, None]
    )

    assert jnp.allclose(source, expected, rtol=1e-12, atol=1e-10)
    nonzero = jnp.abs(source[:, 0]) > 0
    assert jnp.allclose(source[nonzero, 1] / source[nonzero, 0], 9.0 / 4.0)


def test_form_adjoint_source_folds_flat_speeds_onto_a_detector_grid():
    """3D planar data keeps its detector grid: (Nt, Ny, Nz), not (Nt, Ns).

    ``c_at_sources`` is always flat over detectors, ``(Ns,)``. A 2D problem's
    ``(Nt, Ns)`` data broadcasts against it directly, but 3D data does not, and
    the flat speeds must be folded back onto the trailing detector axes in the
    same row-major order the data uses.
    """
    nt, ny, nz = 8, 3, 2
    dt = 0.1
    key = jax.random.PRNGKey(0)
    k_data, k_c = jax.random.split(key)
    data_grid = jax.random.normal(k_data, (nt, ny, nz), dtype=jnp.float64)
    c_flat = jax.random.uniform(
        k_c, (ny * nz,), dtype=jnp.float64, minval=1.0, maxval=2.0
    )

    source_grid = _form_adjoint_source(data_grid, dt, c_flat, None)
    assert source_grid.shape == (nt, ny, nz)

    # The flattened problem is the reference: same physics, (Nt, Ns) layout.
    source_flat = _form_adjoint_source(data_grid.reshape(nt, ny * nz), dt, c_flat, None)
    assert jnp.allclose(source_grid.reshape(nt, ny * nz), source_flat, atol=1e-12)

    # Each detector must receive its own speed (catches a transposed fold).
    expected = (
        -(c_flat.reshape(ny, nz) ** 2)
        * jnp.fft.ifft(
            (2j * jnp.pi * jnp.fft.fftfreq(nt, d=dt)).reshape(nt, 1, 1)
            * jnp.fft.fft(data_grid, axis=0),
            axis=0,
        ).real
    )
    assert jnp.allclose(source_grid, expected, atol=1e-12)


def test_form_adjoint_source_rejects_mismatched_detector_count():
    data = jnp.ones((4, 3, 2))
    with pytest.raises(ValueError, match="does not match the data detector grid"):
        _form_adjoint_source(data, 0.1, jnp.ones(5), None)


def test_prepare_adjoint_uses_raw_spacetime_coefficients(monkeypatch):
    """Boundary-source analysis must not insert an IVP half-wave factor 1/2."""

    class FakeWPT:
        def forward(self, source, input_type):
            assert input_type == "spatial"
            return jnp.array([1.0, 4.0, 2.0, 3.0])

    def fake_compute_adj_parameters(
        indices, data_domain, data_wpt, sources, relative_guard
    ):
        del data_domain, data_wpt, sources
        assert relative_guard == pytest.approx(5e-2)
        b = indices.shape[0]
        return (
            jnp.zeros((b, 1)),
            1j * jnp.ones((b, 1, 1)),
            jnp.zeros((b, 1)),
            jnp.ones((b,)),
            jnp.ones((b, 1)),
            jnp.ones((b, 1)),
            jnp.zeros((b, 2)),
        )

    monkeypatch.setattr(
        msgb_solver_module, "compute_adj_parameters", fake_compute_adj_parameters
    )
    solver = MSGBSolver(
        thr=2,
        thr_strat="top_n",
        batch_size=2,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        sum_method="all_real",
    )
    params = solver._prepare_adj_params(
        source=jnp.ones(4),
        data_domain=object(),
        data_wpt=FakeWPT(),
        sources=object(),
    )

    # top_n returns the selected values in increasing magnitude: 3 then 4.
    assert jnp.array_equal(params[4].ravel(), jnp.array([3.0, 4.0]))


def test_c_inverse_squared_output_weight_closes_constant_speed_mode_identity():
    """The terminal mass-source field needs c^-2 for unweighted image L2."""
    c = jnp.array(2.5)
    spatial_frequency = 1.7
    ts = jnp.linspace(0.0, 0.8, 257)
    dt = ts[1] - ts[0]
    f = jnp.array(0.7)
    h = jnp.sin(2.3 * ts) + 0.2 * jnp.cos(0.4 * ts)
    propagator = jnp.cos(c * spatial_frequency * ts)

    forward_data = propagator * f
    terminal_mass_source_field = c**2 * jnp.sum(propagator * h) * dt
    adjoint_image = _apply_adjoint_image_weight(terminal_mass_source_field, c)

    lhs = jnp.sum(forward_data * h) * dt
    rhs = f * adjoint_image
    assert jnp.allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_adjoint_runs_on_small_2d_problem():
    """Adjoint should run end-to-end on a small 2D non-periodic setup.

    Regression test: `MSGBSolver.adjoint` used to call
    `domain.c_fn(sensor_positions)[0, :]`, which raised `IndexError` because
    `Domain.c_fn` returns a 1D array. The fix evaluates `c` at the source
    (acquisition) geometry, which broadcasts cleanly across the data's time
    axis. This test guards against that bug recurring.
    """
    N = (32, 32)
    dx = (1.0 / N[0], 1.0 / N[1])
    domain = geometry.Domain(N=N, dx=dx, c=_c_const, periodic=(False, False))

    sensor_mask = jnp.zeros(N).at[0, :].set(1)  # one row of sensors along x=0
    sensors = geometry.Sensor(domain=domain, binary_mask=sensor_mask)
    full_grid = geometry.Sensor(domain=domain, binary_mask=jnp.ones(N))

    Ns_x = int(sensors.positions.shape[0])
    Nt = 16
    ts = jnp.linspace(0.0, 0.1, Nt)
    data = jnp.zeros((Nt, Ns_x)).at[Nt // 2, Ns_x // 2].set(1.0)

    solver = MSGBSolver(
        thr=4,
        thr_strat="top_n",
        batch_size=4,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        tr_ode_solver=gb_solvers.solve_ODE_batch_t,
        sum_method="all_real",
    )

    dt = float(ts[1] - ts[0])
    data_dx = (dt, dx[1])
    data_domain = geometry.Domain(
        N=(Nt, Ns_x), dx=data_dx, c=_c_const, periodic=(False, False)
    )
    data_decomp = DyadicDecomposition(
        num_levels=1, N=(Nt, Ns_x), num_boxes_levels=(2,), box_aspect_ratio=(1, 1)
    )
    data_wpt = MSWPT(data_decomp, redundancy=2, windowing="rectangular")

    q_T = solver.adjoint(
        data=data,
        domain=domain,
        sensors=full_grid,
        sources=sensors,
        ts=ts,
        data_domain=data_domain,
        data_wpt=data_wpt,
        window=jnp.ones(Nt),
    )

    # `compute_TR_result` returns one value per evaluation sensor; for the
    # full-grid evaluation this is N0 * N1 values. Shape and exact layout are
    # downstream concerns — what matters here is that the call completes and
    # produces a finite result of the expected size.
    assert int(q_T.size) == N[0] * N[1]
    assert jnp.all(jnp.isfinite(q_T))


if __name__ == "__main__":
    pytest.main([__file__])
