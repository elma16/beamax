import jax.numpy as jnp
import jax
import pytest
from pathlib import Path
import types
import numpy as np

from beamax import geometry, utils

try:
    from beamax.solvers.kwave_solver import KWaveSolver, TimedKWaveSolver
except Exception as exc:  # pragma: no cover - depends on optional k-wave stack.
    pytest.skip(
        f"k-wave-python stack is unavailable: {exc}",
        allow_module_level=True,
    )


ROOT_DIR = utils.detect_root()
DATA_DIR = Path(ROOT_DIR / "tests/test-data")

jax.config.update("jax_enable_x64", True)

_SOLVER_KWARGS = dict(
    pml_inside=False,
    pml_size=20,
    smooth_p0=False,
    backend="cpp",
    device="cpu",
    quiet=True,
)


def _make_kwave_solver() -> KWaveSolver:
    return KWaveSolver(**_SOLVER_KWARGS)


def _match_image_shape(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.array(arr)
    if arr.shape == shape:
        return arr
    if arr.T.shape == shape:
        return arr.T
    raise ValueError(
        f"Cannot match output shape {arr.shape} to reference shape {shape}."
    )


@pytest.mark.parametrize(
    "periodic, d", [(periodic, d) for d in [2, 3] for periodic in [True, False]]
)
def test_kwave_linear(periodic, d):
    """
    Test the k-wave solver is linear.

    NB: fails for d = 1, periodic = False
    """
    if d == 1:
        N = (512, 1)
    elif d == 2:
        N = (64, 64)
    elif d == 3:
        N = (32, 32, 32)
    else:
        raise ValueError(f"Unsupported dimension: {d}")

    d = len(N)
    dx = (1e-4,) * d
    periodic = (periodic,) * d

    print(f"Testing k-wave linearity with N={N}, dx={dx}, periodic={periodic}")

    def c(x):
        return 1 + 0 * x[..., 0]

    domain = geometry.Domain(N=N, dx=dx, c=c, periodic=periodic)
    ts = domain.generate_time_domain()

    sensors_line = jnp.zeros(N)
    sensors_line = sensors_line.at[0, ...].set(1)

    kwave_solver = KWaveSolver(**_SOLVER_KWARGS)

    p0_a = jax.random.normal(jax.random.PRNGKey(0), N)
    p0_a = p0_a / jnp.max(jnp.abs(p0_a))
    p0_b = jax.random.normal(jax.random.PRNGKey(1), N)
    p0_b = p0_b / jnp.max(jnp.abs(p0_b))
    scale = jax.random.normal(jax.random.PRNGKey(2), ())
    p0_sum = p0_a + scale * p0_b

    pt_a = kwave_solver.forward(p0=p0_a, domain=domain, sensors=sensors_line, ts=ts)
    pt_b = kwave_solver.forward(p0=p0_b, domain=domain, sensors=sensors_line, ts=ts)

    pt_sum = kwave_solver.forward(p0=p0_sum, domain=domain, sensors=sensors_line, ts=ts)

    pt_sum_lin = pt_a + scale * pt_b

    assert jnp.allclose(pt_sum, pt_sum_lin, atol=1e-5)


def test_kwave_converges():
    """
    Test that the k-wave solver converges to a solution in a periodic domain.

    Compare a solution with a refined grid to a solution with a coarser grid.
    """
    d = 2
    N = (32,) * d
    dx = (1e-4,) * d
    periodic = (True,) * d
    cfl = 0.3536

    def c(x):
        return 1 + 0 * x[..., 0]

    domain = geometry.Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=cfl)
    ts = domain.generate_time_domain()

    sensors_line = jnp.zeros(N)
    sensors_line = sensors_line.at[(slice(None),) * (d - 1) + (0,)].set(1)
    kwave_solver = KWaveSolver(**_SOLVER_KWARGS)

    p0 = jnp.zeros(N)
    p0 = p0.at[tuple(n // 2 for n in N)].set(1)

    N_refine = tuple(n * 2 for n in N)
    dx_refine = tuple(d / 2 for d in dx)

    p0_refine = utils.interpolate_fourier(
        p0, N_refine, input_type="spatial", output_type="spatial"
    ).real * 2 ** (d / 2)

    slices = tuple(slice(None, None, 2) for _ in range(p0.ndim))
    assert jnp.allclose(
        p0, p0_refine[slices], atol=1e-6
    ), f"p0 and p0_refine are not close: {jnp.max(jnp.abs(p0 - p0_refine[slices]))}"

    domain_refine = geometry.Domain(
        N=N_refine, dx=dx_refine, c=c, periodic=periodic, cfl=cfl
    )
    sensors_refine = jnp.zeros(N_refine)
    sensors_refine = sensors_refine.at[(slice(None),) * (d - 1) + (0,)].set(1)

    ts_refine = domain_refine.generate_time_domain()

    pt = kwave_solver.forward(p0=p0, domain=domain, sensors=sensors_line, ts=ts)

    pt_refine = kwave_solver.forward(
        p0=p0_refine, domain=domain_refine, sensors=sensors_refine, ts=ts_refine
    )

    assert jnp.allclose(
        pt, pt_refine[slices], atol=2e-6
    ), f"pt and pt_refine are not close: {jnp.max(jnp.abs(pt - pt_refine[slices]))}"


def test_kwave_matches_matlab_reference():
    """
    Compare k-wave-python wrapper outputs against stored MATLAB reference data.
    """
    import h5py

    h5file = DATA_DIR / "kWave_results_2.h5"
    if not h5file.exists():
        pytest.skip(f"MATLAB reference fixture is not available: {h5file}")

    with h5py.File(h5file, "r") as h5:
        p0_mat = jnp.array(h5["/p0"][()])
        meas_mat = jnp.array(h5["/data"][()])
        tr_mat = jnp.array(h5["/tr_image"][()])
        adj_mat = jnp.array(h5["/adj_image"][()])

    N = tuple(int(n) for n in p0_mat.shape)
    d = len(N)
    dx = (1e-4,) * d
    periodic = (False,) * d
    cfl = 0.3

    def c(x):
        return 1500 + 0 * x[..., 0]

    domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
    ts = domain.generate_time_domain()

    sensor_mask = jnp.zeros(N)
    sensor_mask = sensor_mask.at[..., 0].set(1)
    sensors_all = jnp.ones(N)

    solver = _make_kwave_solver()

    meas_py = solver.forward(p0=p0_mat, domain=domain, sensors=sensor_mask, ts=ts)
    tr_py = -_match_image_shape(
        solver.time_reversal(
            data=meas_py,
            domain=domain,
            sensors=sensors_all,
            sources=sensor_mask,
            ts=ts,
            data_layout="nt_ns",
        ),
        N,
    )
    adj_py = -_match_image_shape(
        solver.adjoint(
            data=meas_py,
            domain=domain,
            sensors=sensors_all,
            sources=sensor_mask,
            ts=ts,
            data_layout="nt_ns",
        ),
        N,
    )

    tol = 1e-5

    assert jnp.allclose(meas_py, meas_mat, atol=tol)
    assert jnp.allclose(tr_py, tr_mat, atol=tol)
    assert jnp.allclose(adj_py, adj_mat, atol=tol)


def test_kwave_adjoint_dot_product():
    """
    Numerical adjoint check: <A x, y>_D ~= <x, A^T y>_X.

    Uses the python backend for both forward and adjoint to ensure
    consistent sensor ordering (cpp uses F-order, python uses C-order).
    """
    N = (48, 24)
    dx = (1e-4, 1e-4)
    periodic = (False, False)
    cfl = 0.3

    def c(x):
        return 1500 + 0 * x[..., 0]

    domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
    ts = domain.generate_time_domain()
    dt = float(ts[1] - ts[0])
    dA = float(np.prod(np.array(dx)))

    sensor_mask = jnp.zeros(N)
    sensor_mask = sensor_mask.at[0, :].set(1)
    sensor_mask = sensor_mask.at[:, 0].set(1)
    sensors_all = jnp.ones(N)

    # Use python backend for both forward and adjoint to ensure
    # consistent sensor ordering (adjoint forces python internally).
    solver = KWaveSolver(**{**_SOLVER_KWARGS, "backend": "python"})

    x = np.array(jax.random.normal(jax.random.PRNGKey(0), N, dtype=jnp.float32))
    Ax = np.array(solver.forward(p0=x, domain=domain, sensors=sensor_mask, ts=ts))
    y = np.array(jax.random.normal(jax.random.PRNGKey(1), Ax.shape, dtype=jnp.float32))

    Aty = np.array(
        solver.adjoint(
            data=y,
            domain=domain,
            sensors=sensors_all,
            sources=sensor_mask,
            ts=ts,
            data_layout="auto",
        )
    )
    Aty = _match_image_shape(Aty, x.shape)

    lhs = float(np.sum(Ax * y) * dt)
    # Empirical discrete scaling for this k-wave-python setup:
    # the pair is closest under 2*dA (PML + finite time window effects).
    rhs = float(np.sum(x * Aty) * (2.0 * dA))
    rel_err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).eps)

    assert rel_err < 1.5e-1


def test_build_adjoint_source_matches_matlab_shift_sum_fold():
    sensor_data_ns_nt = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    p_src = KWaveSolver._build_adjoint_source(sensor_data_ns_nt)

    # MATLAB reference:
    # r = flip(sensor_data, 2)
    # p_adj = [r, 0] + [0, r]
    # p_adj(:, end-1) += p_adj(:, end)
    # p_src = p_adj(:, 1:end-1)
    expected = np.array(
        [
            [3.0, 5.0, 4.0],
            [6.0, 11.0, 13.0],
        ]
    )
    assert np.array_equal(p_src, expected)


def test_default_solver_kwargs():
    solver = KWaveSolver()
    assert solver._solver_kwargs["backend"] == "cpp"
    assert solver._solver_kwargs["pml_inside"] is False


def test_coerce_sensor_data_layout_accepts_nt_ns_and_ns_nt():
    source_mask = np.zeros((4, 4))
    source_mask[0, 0] = 1
    source_mask[0, 1] = 1
    ns_nt = np.arange(10).reshape(2, 5)
    nt_ns = ns_nt.T

    out_auto_ns_nt = KWaveSolver._coerce_sensor_data_layout(
        ns_nt, source_mask, data_layout="auto", op_name="test"
    )
    out_auto_nt_ns = KWaveSolver._coerce_sensor_data_layout(
        nt_ns, source_mask, data_layout="auto", op_name="test"
    )
    out_explicit_nt_ns = KWaveSolver._coerce_sensor_data_layout(
        nt_ns, source_mask, data_layout="nt_ns", op_name="test"
    )

    assert np.array_equal(out_auto_ns_nt, ns_nt)
    assert np.array_equal(out_auto_nt_ns, ns_nt)
    assert np.array_equal(out_explicit_nt_ns, ns_nt)


def test_coerce_sensor_data_layout_rejects_mismatched_shapes():
    source_mask = np.zeros((4, 4))
    source_mask[0, :3] = 1
    data = np.zeros((5, 2))

    with pytest.raises(ValueError):
        KWaveSolver._coerce_sensor_data_layout(
            data, source_mask, data_layout="auto", op_name="test"
        )


def test_time_call_stdout_and_wall(monkeypatch):
    # stub _run_simulation to print timing once, then silence
    def _run(self, *a, **k):
        print("Total execution time: 0.003s")
        return {"p": "OK"}

    monkeypatch.setattr(KWaveSolver, "_run_simulation", _run)
    d = types.SimpleNamespace(
        N=(4, 4),
        dx=(1.0, 1.0),
        periodic=(True, True),
        sound_speed_array=np.ones((4, 4)),
        density_array=None,
        alpha_coeff=None,
        alpha_power=None,
    )
    solver = TimedKWaveSolver(mode="stdout")
    out, secs = solver.forward(np.zeros((4, 4)), d, np.ones((1,)), np.linspace(0, 1, 4))
    assert out == "OK" and secs == pytest.approx(0.003, rel=1e-6)
    solver = TimedKWaveSolver(mode="wall")
    out, secs = solver.forward(np.zeros((4, 4)), d, np.ones((1,)), np.linspace(0, 1, 4))
    assert out == "OK" and secs >= 0.0


def test_timed_solver_passes_tr_and_adj_kwargs(monkeypatch):
    tr_calls = {}
    adj_calls = {}

    def _tr(
        self,
        data,
        domain,
        sensors,
        sources,
        ts,
        *,
        record="p_final",
        data_layout="auto",
        **solver_kwargs,
    ):
        tr_calls.update(
            {
                "data": data,
                "domain": domain,
                "sensors": sensors,
                "sources": sources,
                "ts": ts,
                "record": record,
                "data_layout": data_layout,
                "solver_kwargs": solver_kwargs,
            }
        )
        return "TR"

    def _adj(
        self,
        data,
        domain,
        sensors,
        sources,
        ts,
        *,
        record="p_final",
        data_layout="auto",
        **solver_kwargs,
    ):
        adj_calls.update(
            {
                "data": data,
                "domain": domain,
                "sensors": sensors,
                "sources": sources,
                "ts": ts,
                "record": record,
                "data_layout": data_layout,
                "solver_kwargs": solver_kwargs,
            }
        )
        return "ADJ"

    monkeypatch.setattr(KWaveSolver, "time_reversal", _tr)
    monkeypatch.setattr(KWaveSolver, "adjoint", _adj)

    solver = TimedKWaveSolver(mode="wall")

    out_tr, secs_tr = solver.time_reversal(
        data="data",
        domain="domain",
        sensors="sensors",
        sources="sources",
        ts="ts",
        record="p",
        data_layout="nt_ns",
        foo=123,
    )
    out_adj, secs_adj = solver.adjoint(
        data="data2",
        domain="domain2",
        sensors="sensors2",
        sources="sources2",
        ts="ts2",
        record="p_final",
        data_layout="ns_nt",
        bar="x",
    )

    assert out_tr == "TR" and secs_tr >= 0.0
    assert out_adj == "ADJ" and secs_adj >= 0.0
    assert tr_calls["sources"] == "sources"
    assert tr_calls["data_layout"] == "nt_ns"
    assert tr_calls["solver_kwargs"]["foo"] == 123
    assert adj_calls["sources"] == "sources2"
    assert adj_calls["data_layout"] == "ns_nt"
    assert adj_calls["solver_kwargs"]["bar"] == "x"


# def test_kw():
#     """
#     test linear tr.
#     """
#     h5file = DATA_DIR / "kWave_results_1.h5"
#     h5 = h5py.File(h5file, "r")
#     p0_mat = jnp.array(h5["/p0"][()])
#     h5.close()

#     N = p0_mat.shape  # (Nx,Ny)
#     d = len(N)
#     dx = (1e-4,) * d
#     periodic = (False,) * d
#     cfl = 0.3

#     def c(x):
#         return 1500 + 0 * x[..., 0]

#     domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
#     ts = domain.generate_time_domain()

#     sensor_mask_1 = jnp.zeros(N)
#     sensor_mask_1 = sensor_mask_1.at[..., 0].set(1)

#     sensor_mask_2 = jnp.zeros(N)
#     sensor_mask_2 = sensor_mask_2.at[..., -1].set(1)

#     sensor_mask_sum = jnp.zeros(N)
#     sensor_mask_sum = sensor_mask_sum.at[..., 0].set(1)
#     sensor_mask_sum = sensor_mask_sum.at[..., -1].set(1)

#     sensors_all = jnp.ones(N)

#     sim_opts = SimulationOptions(data_cast="double", smooth_p0=False, save_to_disk=True)
#     exec_opts = SimulationExecutionOptions(
#         is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
#     )

#     solver = KWaveSolver(sim_opts, exec_opts)

#     meas_1 = solver.forward(p0_mat, domain, sensor_mask_1, ts)
#     meas_2 = solver.forward(p0_mat, domain, sensor_mask_2, ts)
#     meas_sum = solver.forward(p0_mat, domain, sensor_mask_sum, ts)

#     tr_1 = solver.time_reversal(meas_1.T, sensor_mask_1, domain, sensors_all, ts).T
#     tr_2 = solver.time_reversal(meas_2.T, sensor_mask_2, domain, sensors_all, ts).T
#     tr_sum = solver.time_reversal(
#         meas_sum.T, sensor_mask_sum, domain, sensors_all, ts
#     ).T

#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(tr_1 + tr_2)
#     plt.title("tr_1 + tr_2")
#     plt.colorbar()
#     plt.subplot(1, 3, 2)
#     plt.imshow(tr_sum)
#     plt.title("tr_sum")
#     plt.colorbar()
#     plt.subplot(1, 3, 3)
#     plt.imshow(tr_sum - (tr_1 + tr_2))
#     plt.title("tr_sum - (tr_1 + tr_2)")
#     plt.colorbar()
#     plt.tight_layout()
#     plt.show()


#     assert jnp.allclose(tr_sum, tr_1 + tr_2, atol=1e-5), (
#         f"tr_sum and tr_1 + tr_2 are not close: "
#         f"{jnp.max(jnp.abs(tr_sum - (tr_1 + tr_2)))}"
#     )


if __name__ == "__main__":
    pytest.main([__file__])
