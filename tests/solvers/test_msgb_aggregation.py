"""
Cross-method consistency tests for MSGB forward, time-reversal, and adjoint.

For each public method we sweep ``sum_method`` (all/scan/vmap × real/complex)
and assert that every variant produces the same result. The existing
:class:`tests.solvers.test_fwd_solver.TestMSGBSolverAggregation` covers the
forward path in 1D; this file extends that coverage to 2D and to the TR /
adjoint paths.

Each test uses a tiny grid (N <= (32, 32)) so the full matrix runs in a few
seconds. The cross-method assertions catch (a) regressions in the
``_prepare_tr_params`` / ``_prepare_adj_params`` batching path, and
(b) silent shape/sign drifts between aggregation strategies.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamax import geometry, utils
from beamax.decomposition import DyadicDecomposition
from beamax.gb import gb_solvers
from beamax.solvers.msgb_solvers.msgb_solver import MSGBSolver
from beamax.transforms import MSWPT, compute_frames


jax.config.update("jax_enable_x64", True)


def _make_test_signal_2d(dyadic, wpt, *, box_indices=(34, 6), k_values=None):
    """Build a sparse 2D signal from MSWPT frames (matches test_fwd_solver helper)."""
    if k_values is None:
        k_values = (jnp.array([10, 20]), jnp.array([8, 3]))
    KXY = dyadic.fourier_meshgrid
    signal = 0
    for box_idx, k in zip(box_indices, k_values):
        frame_ft = compute_frames(dyadic, box_idx, k, KXY, wpt.redundancy, "none")
        signal += utils.unitary_ifft(frame_ft)
    return signal / jnp.max(jnp.abs(signal))


def _solver(sum_method, *, batch_size=64):
    return MSGBSolver(
        thr=200,
        thr_strat="top_n",
        batch_size=batch_size,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        tr_ode_solver=gb_solvers.solve_ODE_batch_t,
        sum_method=sum_method,
    )


# ============================================================================
# Forward — extend the existing 1D consistency check to 2D
# ============================================================================


@pytest.fixture(scope="module")
def domain_2d_periodic():
    def c(x):
        return 1500.0 + 0.0 * x[..., 0]

    return geometry.Domain(
        N=(64, 64),
        dx=(1e-4, 1e-4),
        c=c,
        cfl=0.354,
        periodic=(True, True),
    )


@pytest.fixture(scope="module")
def dyadic_2d():
    return DyadicDecomposition(
        num_levels=2,
        N=(64, 64),
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1, 1),
    )


@pytest.fixture(scope="module")
def wpt_2d(dyadic_2d):
    return MSWPT(dyadic_2d, redundancy=2, windowing="rectangular")


class TestForwardAggregationConsistency2D:
    """All forward `sum_method` variants must agree on a small 2D problem."""

    @pytest.mark.parametrize("use_complex", [False, True])
    def test_2d_all_methods_match(
        self, domain_2d_periodic, dyadic_2d, wpt_2d, use_complex
    ):
        sensors = geometry.Sensor(
            binary_mask=jnp.ones(domain_2d_periodic.N), domain=domain_2d_periodic
        )
        p0 = _make_test_signal_2d(dyadic_2d, wpt_2d)
        if not use_complex:
            p0 = p0.real
        ts = jnp.array([0.0])

        methods = (
            ["all_complex", "vmap_complex", "scan_complex"]
            if use_complex
            else ["all_real", "vmap_real", "scan_real"]
        )

        results = []
        for m in methods:
            out = _solver(m, batch_size=32).forward(
                p0,
                domain_2d_periodic,
                sensors,
                ts,
                wpt_2d,
            )
            results.append(np.asarray(out))

        for i, m in enumerate(methods[1:], 1):
            diff = np.max(np.abs(results[0] - results[i]))
            assert diff < 1e-12, (
                f"forward 2D: method {m!r} differs from {methods[0]!r} "
                f"by max abs diff {diff:.3e}"
            )


# ============================================================================
# Time reversal — cross-method consistency in 1D and 2D
# ============================================================================


def _build_data_domain(domain, ts, *, over_resolve=2):
    """Build a frequency-cropped (Nt', Ns) data domain and matching MSWPT.

    Mirrors what `examples/.../2d_time_reversal_and_adjoint.py` does. Kept here
    so the consistency test is self-contained.
    """
    nt_cropped = over_resolve * domain.N[0]
    if domain.ndim == 1:
        nt_data = nt_cropped
        domain_data = geometry.Domain(
            N=(nt_data,),
            dx=(float((ts[-1] - ts[0]) / (nt_data - 1)),),
            c=domain.c,
            periodic=(False,),
            cfl=domain.cfl,
        )
        dyadic = DyadicDecomposition(
            num_levels=2,
            N=(nt_data,),
            num_boxes_levels=(4, 8),
            box_aspect_ratio=(1,),
        )
    else:
        nt_data = nt_cropped
        ns = domain.N[1]
        domain_data = geometry.Domain(
            N=(nt_data, ns),
            dx=(float((ts[-1] - ts[0]) / (nt_data - 1)), float(domain.dx[1])),
            c=domain.c,
            periodic=(False, False),
            cfl=domain.cfl,
        )
        n_min = min(nt_data, ns)
        box_aspect = (nt_data // n_min, ns // n_min)
        dyadic = DyadicDecomposition(
            num_levels=2,
            N=(nt_data, ns),
            num_boxes_levels=(4, 8),
            box_aspect_ratio=box_aspect,
        )
    wpt = MSWPT(dyadic, redundancy=2, windowing="rectangular_mirror")
    return domain_data, wpt


def _synth_sensor_data(domain, ts, *, key_seed=0):
    """Cheap synthetic sensor record; we only care that solvers agree, not that
    the reconstruction looks like anything in particular."""
    key = jax.random.PRNGKey(key_seed)
    nt_data = 2 * domain.N[0]
    if domain.ndim == 1:
        return jax.random.normal(key, (nt_data,)) * 0.01
    return jax.random.normal(key, (nt_data, domain.N[1])) * 0.01


class TestTimeReversalAggregationConsistency:
    """TR sum_method variants must agree on tiny 1D + 2D problems."""

    def test_1d_all_methods_match(self):
        n = (64,)
        domain = geometry.Domain(
            N=n,
            dx=(1.0 / n[0],),
            c=lambda x: 1.0 + 0.0 * x[..., 0],
            cfl=0.3,
            periodic=(False,),
        )
        ts = domain.generate_time_domain()
        data_domain, data_wpt = _build_data_domain(domain, ts)
        sensor_mask = jnp.zeros(n).at[0].set(1.0)
        sources = geometry.Sensor(domain=domain, binary_mask=sensor_mask)
        eval_sensors = geometry.Sensor(domain=domain, binary_mask=jnp.ones(n))

        data = _synth_sensor_data(domain, ts)

        results = []
        for sm in ["all_real", "scan_real", "vmap_real"]:
            out = _solver(sm, batch_size=16).time_reversal(
                data=data,
                domain=domain,
                sensors=eval_sensors,
                sources=sources,
                ts=ts,
                data_domain=data_domain,
                data_wpt=data_wpt,
            )
            results.append(np.asarray(out))

        for i, sm in enumerate(["scan_real", "vmap_real"], 1):
            diff = np.max(np.abs(results[0] - results[i]))
            assert (
                diff < 1e-8
            ), f"TR 1D: {sm!r} differs from all_real by max abs diff {diff:.3e}"

    def test_2d_all_methods_match(self):
        n = (32, 32)
        domain = geometry.Domain(
            N=n,
            dx=(1e-4, 1e-4),
            c=lambda x: 1500.0 + 0.0 * x[..., 0],
            cfl=0.3,
            periodic=(False, False),
        )
        ts = domain.generate_time_domain()
        data_domain, data_wpt = _build_data_domain(domain, ts)
        sensor_mask = jnp.zeros(n).at[0, :].set(1.0)
        sources = geometry.Sensor(domain=domain, binary_mask=sensor_mask)
        eval_sensors = geometry.Sensor(domain=domain, binary_mask=jnp.ones(n))

        data = _synth_sensor_data(domain, ts)

        results = []
        for sm in ["all_real", "scan_real", "vmap_real"]:
            out = _solver(sm, batch_size=8).time_reversal(
                data=data,
                domain=domain,
                sensors=eval_sensors,
                sources=sources,
                ts=ts,
                data_domain=data_domain,
                data_wpt=data_wpt,
            )
            results.append(np.asarray(out))

        for i, sm in enumerate(["scan_real", "vmap_real"], 1):
            diff = np.max(np.abs(results[0] - results[i]))
            assert (
                diff < 1e-8
            ), f"TR 2D: {sm!r} differs from all_real by max abs diff {diff:.3e}"


# ============================================================================
# Adjoint — cross-method consistency in 2D (with the post-bug-fix indexing)
# ============================================================================


class TestAdjointAggregationConsistency:
    """Adjoint sum_method variants must agree on a tiny 2D problem."""

    def test_2d_all_methods_match(self):
        n = (32, 32)
        domain = geometry.Domain(
            N=n,
            dx=(1e-4, 1e-4),
            c=lambda x: 1500.0 + 0.0 * x[..., 0],
            cfl=0.3,
            periodic=(False, False),
        )
        ts = domain.generate_time_domain()
        data_domain, data_wpt = _build_data_domain(domain, ts)
        sensor_mask = jnp.zeros(n).at[0, :].set(1.0)
        sources = geometry.Sensor(domain=domain, binary_mask=sensor_mask)
        eval_sensors = geometry.Sensor(domain=domain, binary_mask=jnp.ones(n))

        data = _synth_sensor_data(domain, ts)

        results = []
        for sm in ["all_real", "scan_real", "vmap_real"]:
            out = _solver(sm, batch_size=8).adjoint(
                data=data,
                domain=domain,
                sensors=eval_sensors,
                sources=sources,
                ts=ts,
                data_domain=data_domain,
                data_wpt=data_wpt,
            )
            results.append(np.asarray(out))

        for i, sm in enumerate(["scan_real", "vmap_real"], 1):
            diff = np.max(np.abs(results[0] - results[i]))
            assert (
                diff < 1e-8
            ), f"adjoint 2D: {sm!r} differs from all_real by max abs diff {diff:.3e}"


# ============================================================================
# Regression: the _prepare_tr_params unconditional batch_data bug
# ============================================================================


def test_tr_2d_all_real_regression():
    """Regression for the _prepare_tr_params batching bug.

    `_prepare_tr_params` used to call ``utils.batch_data(...)`` unconditionally,
    leaving the TR params in ``(num_batches, batch_size, ...)`` form even for
    ``aggregate_method == 'all'``. Downstream, ``solve_ODE_batch_t``'s inner
    vmap stripped only one of the two batch axes, so ``mode`` arrived at
    ``coupled_rhs`` with shape ``(batch_size, 1)`` and tripped the broadcast
    against the 2x2 Hessian inside ``riccati_rhs``.

    Fixed by gating the ``batch_data`` call on
    ``self.aggregate_method in ["scan", "vmap"]`` (matching ``_prepare_adj_params``).
    """
    n = (32, 32)
    domain = geometry.Domain(
        N=n,
        dx=(1e-4, 1e-4),
        c=lambda x: 1500.0 + 0.0 * x[..., 0],
        cfl=0.3,
        periodic=(False, False),
    )
    ts = domain.generate_time_domain()
    data_domain, data_wpt = _build_data_domain(domain, ts)
    sensor_mask = jnp.zeros(n).at[0, :].set(1.0)
    sources = geometry.Sensor(domain=domain, binary_mask=sensor_mask)
    eval_sensors = geometry.Sensor(domain=domain, binary_mask=jnp.ones(n))

    data = _synth_sensor_data(domain, ts)
    out = _solver("all_real", batch_size=8).time_reversal(
        data=data,
        domain=domain,
        sensors=eval_sensors,
        sources=sources,
        ts=ts,
        data_domain=data_domain,
        data_wpt=data_wpt,
    )
    out_np = np.asarray(out)
    assert int(out_np.size) == n[0] * n[1]
    assert np.all(np.isfinite(out_np))


if __name__ == "__main__":
    pytest.main([__file__])
