import os
import subprocess
from pathlib import Path

import jax
import numpy as np
import pytest

from beamax.geometry import Domain
from beamax.install_kwave_metal import metal_binary_path, metal_supported

try:
    import kwave
    import h5py
    from beamax.solvers import kwave_solver
    from beamax.solvers.kwave_solver import KWaveSolver
except ModuleNotFoundError as exc:
    if exc.name != "kwave":
        raise
    pytest.skip("k-wave-python is unavailable", allow_module_level=True)

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    ("operation", "shape"), [("time_reversal", (8, 6)), ("adjoint", (8, 6, 7))]
)
def test_inverse_native_file_contract(tmp_path, monkeypatch, operation, shape):
    """Exercise k-wave-python's real HDF5 writer with a simulated native output."""
    ndim = len(shape)
    ix = np.indices(shape)
    c = 1440.0 + 5 * ix[0] + ix[1]
    rho = 995.0 + ix[1] + ix[-1]
    domain = Domain(
        N=shape, dx=(1e-4,) * ndim, c=c, density=rho, periodic=(False,) * ndim
    )
    mask = np.zeros(shape, dtype=bool)
    points = np.array([(1, 3, 1), (2, 5, 2), (4, 1, 0)])[:, :ndim]
    mask[tuple(points.T)] = True
    data = np.array([[1, 2, 3, 4], [4, 6, 5, 8], [3, 1, 2, 9]], dtype=float)
    ts = np.arange(4) * 1e-8
    binary = tmp_path / "solver"
    binary.touch()
    pml_size = tuple(range(1, ndim + 1))
    padded_shape = tuple(n + 2 * p for n, p in zip(shape, pml_size))
    full_field = np.arange(np.prod(padded_shape), dtype=np.float32).reshape(
        padded_shape
    )

    if operation == "time_reversal":
        expected_rows = data[:, ::-1] / (ndim * c[mask, None] ** 2)
        expected_mode = 0  # Dirichlet, assigning each split density.
    else:
        beta = np.array([[4, 7, 5, 4], [8, 13, 11, 14], [9, 11, 3, 7]])
        expected_rows = rho[mask, None] * beta / (2 * ndim)
        expected_mode = 1  # Additive, without the cosine source correction.
    native_order = np.argsort(np.ravel_multi_index(points.T, shape, order="F"))
    padded_mask = np.pad(mask, [(p, p) for p in pml_size])
    calls = []

    def run(command, **options):
        args = list(map(str, command))
        calls.append(args[0])
        assert "--p_final" in args
        with h5py.File(args[args.index("-i") + 1]) as h5:
            assert h5["p_source_mode"][()].item() == expected_mode
            np.testing.assert_array_equal(
                h5["p_source_index"][:].ravel(),
                1 + np.flatnonzero(padded_mask.ravel(order="F")),
            )
            np.testing.assert_allclose(
                h5["p_source_input"][0].T, expected_rows[native_order], rtol=2e-6
            )
        with h5py.File(args[args.index("-o") + 1], "w") as h5:
            raw = full_field.T if ndim == 3 else full_field.T[None, ...]
            h5.create_dataset("p_final", data=raw)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(subprocess, "run", run)
    solver = KWaveSolver(
        backend="cpp", binary_path=binary, pml_size=pml_size, quiet=True
    )
    field = getattr(solver, operation)(
        data, domain, mask, mask, ts, data_layout="ns_nt"
    )
    expected_field = full_field[tuple(slice(p, -p) for p in pml_size)]
    if operation == "adjoint":
        expected_field = expected_field / (rho * c**2)
    np.testing.assert_allclose(field, expected_field, rtol=2e-6)
    assert calls == [str(binary)]


@pytest.mark.parametrize(
    ("record", "ns"),
    [("p", 1), ("p", 3), ("p", 4), ("p_min", 3), ("p_max", 3), ("p_rms", 3)],
)
def test_native_time_reversal_returns_pressure_in_sensor_mask_order(
    tmp_path, monkeypatch, record, ns
):
    shape = (8, 6)
    domain = Domain(N=shape, dx=(1e-4,) * 2, c=1500.0, periodic=(True, True))
    ts = np.arange(4) * 1e-8
    source_mask = np.zeros(shape, dtype=bool)
    source_mask[3, 3] = True
    sensor_mask = np.zeros_like(source_mask)
    points = np.array([(1, 3), (2, 5), (4, 1), (5, 0)])[:ns]
    sensor_mask[tuple(points.T)] = True
    order = np.argsort(np.ravel_multi_index(points.T, shape, order="F"))
    codes = 100 * points[:, 0] + points[:, 1]
    traces = codes[:, None] + 1000 * np.arange(ts.size)
    binary = tmp_path / "solver"
    binary.touch()

    def run(command, **options):
        flag = "--p_raw" if record == "p" else f"--{record}"
        assert flag in command
        with h5py.File(command[command.index("-o") + 1], "w") as output:
            native = (
                traces[order].T[None, ...]
                if record == "p"
                else codes[order][None, None, :]
            )
            output.create_dataset(record, data=native)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    field = KWaveSolver(backend="cpp", binary_path=binary).time_reversal(
        np.ones((1, ts.size)), domain, sensor_mask, source_mask, ts, record=record
    )
    np.testing.assert_array_equal(field, traces if record == "p" else codes)


@pytest.mark.parametrize("record", ["p_final", "p_min_all", "p_max_all"])
def test_native_forward_preserves_grid_even_when_shape_matches_traces(
    tmp_path, monkeypatch, record
):
    shape = (4, 6)
    domain = Domain(N=shape, dx=(1e-4,) * 2, c=1500.0, periodic=(True, True))
    mask = np.zeros(shape, dtype=bool)
    mask[:, 1] = True
    ts = np.arange(shape[1]) * 1e-8
    p0 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    final = 3 * p0 + 2
    binary = tmp_path / "solver"
    binary.touch()

    def run(command, **options):
        assert f"--{record}" in command
        with h5py.File(command[command.index("-i") + 1]) as inputs:
            np.testing.assert_array_equal(inputs["p0_source_input"][0].T, p0)
        with h5py.File(command[command.index("-o") + 1], "w") as output:
            output.create_dataset(record, data=final.T[None, ...])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    result = KWaveSolver(backend="cpp", binary_path=binary).forward(
        p0, domain, mask, ts, record=record
    )
    np.testing.assert_array_equal(result, final)


@pytest.fixture(params=["cpp", "mac"])
def native_engine(request):
    """Exercise each installed native engine independently without downloading in tests."""
    if os.environ.get("BEAMAX_RUN_KWAVE_CPP_TESTS") != "1":
        pytest.skip(
            "Set BEAMAX_RUN_KWAVE_CPP_TESTS=1 to run native inverse comparisons."
        )
    if request.param == "cpp":
        binary = Path(kwave.BINARY_PATH) / "kspaceFirstOrder-OMP"
    else:
        if not metal_supported():
            pytest.skip("Apple silicon is required for the Metal binary.")
        override = os.environ.get("BEAMAX_KWAVE_BINARY_PATH")
        binary = (
            kwave_solver._normalize_kwave_binary_path(override, device="metal")
            if override
            else metal_binary_path()
        )
    if not binary.is_file():
        pytest.skip(f"Native k-Wave executable unavailable: {binary}")
    return request.param, binary


@pytest.mark.parametrize(("shape", "pml_size"), [((16, 12), 0), ((12, 10, 8), 2)])
def test_native_inverse_against_python_and_forward(native_engine, shape, pml_size):
    ndim = len(shape)
    ix = np.indices(shape)
    c = 1480.0 + 2 * ix[0] + ix[-1]
    # A smooth periodic density avoids a wraparound jump on the unpadded FFT grid.
    rho = 995.0 + 3 * np.cos(2 * np.pi * ix[1] / shape[1])
    domain = Domain(
        N=shape, dx=(1e-4,) * ndim, c=c, density=rho, periodic=(False,) * ndim
    )
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(np.array([(3, 6, 2), (4, 2, 3), (6, 3, 1)])[:, :ndim].T)] = True
    kwargs = dict(pml_size=pml_size, pml_inside=False, quiet=True)
    cpu = KWaveSolver(backend="python", device="cpu", **kwargs)
    backend, binary = native_engine
    native = KWaveSolver(backend=backend, binary_path=binary, **kwargs)
    ts = np.arange(19) * 1.5e-8
    rng = np.random.default_rng(17)
    x = rng.normal(size=shape)
    y = rng.normal(size=(ts.size, int(mask.sum())))
    y[[0, -1]] *= 3  # Probe both endpoints, including the terminal fold.
    forward = native.forward(x, domain, mask, ts)
    cpu_forward = cpu.forward(x, domain, mask, ts)
    assert np.linalg.norm(forward - cpu_forward) / np.linalg.norm(cpu_forward) < 1e-4
    receivers = np.roll(mask, 1, axis=0)
    args = (y, domain, receivers, mask, ts)
    cpu_pressure = cpu.time_reversal(*args, record="p", data_layout="nt_ns")
    native_pressure = native.time_reversal(*args, record="p", data_layout="nt_ns")
    assert native_pressure.shape == (int(receivers.sum()), ts.size)
    assert (
        np.linalg.norm(native_pressure - cpu_pressure) / np.linalg.norm(cpu_pressure)
        < 1e-4
    )
    images = {}
    for operation in ("time_reversal", "adjoint"):
        args = (y, domain, np.ones(shape, dtype=bool), mask, ts)
        reference = getattr(cpu, operation)(*args, data_layout="nt_ns")
        images[operation] = getattr(native, operation)(*args, data_layout="nt_ns")
        assert images[operation].shape == shape
        relative_error = np.linalg.norm(images[operation] - reference) / np.linalg.norm(
            reference
        )
        assert relative_error < 1e-4

    if pml_size == 0:
        # No PML damping, so compare directly against the discrete forward operator.
        lhs = np.vdot(forward, y)
        rhs = np.vdot(x, images["adjoint"])
        np.testing.assert_allclose(lhs, rhs, rtol=3e-5, atol=1e-6)
    else:
        # Isolate the terminal sample with the exterior absorbing layer enabled.
        terminal = np.zeros_like(y)
        terminal[-1] = y[-1]
        args = (terminal, domain, np.ones(shape, dtype=bool), mask, ts)
        reference = cpu.adjoint(*args, data_layout="nt_ns")
        image = native.adjoint(*args, data_layout="nt_ns")
        assert np.linalg.norm(image - reference) / np.linalg.norm(reference) < 1e-4


def test_native_surface_order_and_pairing(native_engine):
    from beamax.surfaces import DetectorSurface

    backend, binary = native_engine
    kw = KWaveSolver(
        backend=backend, binary_path=binary, pml_size=0, smooth_p0=False, quiet=True
    )
    domain = Domain((16, 16), (0.125, 0.125), (False, False), c=1.0)
    surface = DetectorSurface.circle((1.0, 0.875), 0.5, 4)
    mapping = surface.to_mask(domain)
    ts = np.arange(4) * 0.025
    rng = np.random.default_rng(810)
    p0 = rng.normal(size=domain.N)
    pressure = kw.forward(p0, domain, detectors=surface, ts=ts)
    reference = kw.forward(p0, domain, mapping.binary_mask, ts)
    np.testing.assert_allclose(
        pressure, reference[:, mapping.mask_to_chart], rtol=2e-5, atol=2e-6
    )
    data = rng.normal(size=(4, 4))
    args = (
        data[:, mapping.chart_to_mask],
        domain,
        np.ones(domain.N, bool),
        mapping.binary_mask,
        ts,
    )
    for operation in ("time_reversal", "adjoint"):
        solve = getattr(kw, operation)
        kwargs = {"pairing": "euclidean"} if operation == "adjoint" else {}
        np.testing.assert_allclose(
            solve(data, domain, detectors=surface, ts=ts, **kwargs),
            solve(*args, data_layout="nt_ns"),
            rtol=2e-5,
            atol=2e-6,
        )
