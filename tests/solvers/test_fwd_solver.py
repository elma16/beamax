"""
Comprehensive tests for MSGB forward solvers.

Test organization:
- TestMSGBSolverBasics: Core MSGB solver functionality
- TestMSGBSolverAggregation: Different aggregation strategies
- TestMSGBSolverAccuracy: Accuracy and convergence tests
- TestHybridSolverBasics: Basic hybrid solver functionality
- TestHybridSolverAdvanced: Advanced features (interpolation, windowing, etc.)
- TestSolverSharding: Multi-device parallelization
"""

import jax.numpy as jnp
import jax
import os
import pytest
import sys
import warnings
from pathlib import Path

from beamax import geometry, utils, transforms
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import gb_solvers, core
from beamax.solvers.msgb_solvers.msgb_solver import MSGBSolver
from beamax.solvers.msgb_solvers import forward_solver_utils
from beamax.solvers import HybridBackend, HybridSolver
from beamax.solvers.hybrid_solver import HybridSolverConfig

try:
    from beamax.solvers import KWaveSolver
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions
except Exception as exc:  # pragma: no cover - depends on optional k-wave stack.
    KWaveSolver = None
    SimulationExecutionOptions = None
    SimulationOptions = None
    KWAVE_IMPORT_ERROR = exc
else:
    KWAVE_IMPORT_ERROR = None

jax.config.update("jax_enable_x64", True)

requires_kwave = pytest.mark.skipif(
    KWAVE_IMPORT_ERROR is not None,
    reason=f"k-wave-python stack is unavailable: {KWAVE_IMPORT_ERROR}",
)


def _kwave_cpp_binary_available() -> bool:
    if KWAVE_IMPORT_ERROR is not None:
        return False
    if os.environ.get("CI") and os.environ.get("BEAMAX_RUN_KWAVE_CPP_TESTS") != "1":
        return False

    binary_name = (
        "kspaceFirstOrder-OMP.exe"
        if sys.platform == "win32"
        else "kspaceFirstOrder-OMP"
    )

    override = os.environ.get("BEAMAX_KWAVE_BINARY_PATH")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_dir():
            candidate = candidate / binary_name
        return candidate.exists()

    import kwave

    return (Path(kwave.BINARY_PATH) / binary_name).exists()


requires_kwave_cpp_binary = pytest.mark.skipif(
    not _kwave_cpp_binary_available(),
    reason=(
        "k-wave-python C++ OMP binary tests are disabled on CI unless "
        "BEAMAX_RUN_KWAVE_CPP_TESTS=1 is set, or the binary is unavailable."
    ),
)


# ============================================================================
# Fixtures and Utilities
# ============================================================================


# Pure data-object fixtures — kept module-scoped so identical (Domain, decomp,
# wpt) instances are reused across the many parameterised tests below. They
# hold no mutable JAX state, so reuse is safe and avoids redundant construction
# (and the JIT cache they prime stays warm across tests).
@pytest.fixture(scope="module")
def simple_domain_1d():
    """1D domain with homogeneous speed of sound."""

    def c(x):
        return 1.0 + 0 * x[..., 0]

    return geometry.Domain(N=(128,), dx=(1e-4,), c=c, periodic=(True,))


@pytest.fixture(scope="module")
def simple_domain_2d():
    """2D domain with homogeneous speed of sound."""

    def c(x):
        return 1500.0 + 0 * x[..., 0]

    return geometry.Domain(
        N=(128, 128), dx=(1e-4, 1e-4), c=c, cfl=0.354, periodic=(True, True)
    )


@pytest.fixture(scope="module")
def dyadic_decomp_1d():
    """1D dyadic decomposition."""
    return DyadicDecomposition(
        num_levels=2, N=(128,), num_boxes_levels=(4, 8), box_aspect_ratio=(1,)
    )


@pytest.fixture(scope="module")
def dyadic_decomp_2d():
    """2D dyadic decomposition."""
    return DyadicDecomposition(
        num_levels=2, N=(128, 128), num_boxes_levels=(4, 8), box_aspect_ratio=(1, 1)
    )


@pytest.fixture(scope="module")
def wpt_1d(dyadic_decomp_1d):
    """1D MSWPT."""
    return MSWPT(dyadic_decomp_1d, redundancy=2, windowing="rectangular")


@pytest.fixture(scope="module")
def wpt_2d(dyadic_decomp_2d):
    """2D MSWPT."""
    return MSWPT(dyadic_decomp_2d, redundancy=2, windowing="rectangular")


def create_test_signal(dyadic_decomp, wpt, box_indices=(34, 6), k_values=None):
    """
    Create a test signal from specific wavelet frames.

    Parameters
    ----------
    dyadic_decomp : DyadicDecomposition
    wpt : MSWPT
    box_indices : tuple
        Which boxes to use for signal
    k_values : tuple of arrays, optional
        Wave numbers for each box

    Returns
    -------
    jnp.ndarray
        Test signal in spatial domain
    """
    if k_values is None:
        if dyadic_decomp.ndim == 1:
            k_values = (jnp.array([10]), jnp.array([8]))
        else:
            k_values = (jnp.array([10, 20]), jnp.array([8, 3]))

    KXY = dyadic_decomp.fourier_meshgrid

    signal = 0
    for box_idx, k in zip(box_indices, k_values):
        frame_ft = transforms.compute_frames(
            dyadic_decomp, box_idx, k, KXY, wpt.redundancy, "none"
        )
        signal += utils.unitary_ifft(frame_ft)

    # Normalize
    signal = signal / jnp.max(jnp.abs(signal))
    return signal


@pytest.mark.parametrize(
    "override, message",
    [
        ({"thr_strat": "missing"}, "thr_strat"),
        ({"input_type": "time"}, "input_type"),
        ({"batch_size": 0}, "batch_size"),
        ({"sum_method": "all"}, "sum_method"),
    ],
)
def test_msgb_solver_validates_configuration(override, message):
    kwargs = dict(
        thr=10,
        thr_strat="top_n",
        batch_size=8,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        sum_method="scan_real",
    )
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        MSGBSolver(**kwargs)


# ============================================================================
# Test MSGB Solver - Aggregation Methods
# ============================================================================


class TestMSGBSolverAggregation:
    """Test that different aggregation methods produce identical results."""

    @pytest.mark.parametrize("periodic", [True, False])
    @pytest.mark.parametrize("use_complex", [False, True])
    def test_aggregation_methods_consistency(
        self, simple_domain_1d, dyadic_decomp_1d, wpt_1d, periodic, use_complex
    ):
        """Verify all aggregation methods (scan/vmap/all) give identical results."""
        # Update domain periodicity
        domain = geometry.Domain(
            N=simple_domain_1d.N,
            dx=simple_domain_1d.dx,
            c=simple_domain_1d.c,
            periodic=(periodic,),
        )

        sensors = geometry.Sensor(binary_mask=jnp.ones(domain.N), domain=domain)

        # Create test signal
        p0 = create_test_signal(dyadic_decomp_1d, wpt_1d)
        if not use_complex:
            p0 = p0.real
        jnp.zeros_like(p0)

        # Select methods based on dtype
        if use_complex:
            methods = ["all_complex", "vmap_complex", "scan_complex"]
        else:
            methods = ["all_real", "vmap_real", "scan_real"]

        # Run all methods
        results = []
        ts = jnp.array([0.0])

        for method in methods:
            solver = MSGBSolver(
                thr=100,
                thr_strat="top_n",
                batch_size=26,
                input_type="spatial",
                ode_solver=gb_solvers.solve_ODE_base,
                tr_ode_solver=gb_solvers.solve_ODE_batch_t,
                sum_method=method,
            )
            result = solver.forward(p0, domain, sensors, ts, wpt_1d)
            results.append(result)

        # Verify all results match
        for i, method in enumerate(methods[1:], 1):
            assert jnp.allclose(results[0], results[i], atol=1e-14), (
                f"Method {methods[i]} differs from {methods[0]}:\n"
                f"  Max diff: {jnp.max(jnp.abs(results[0] - results[i]))}\n"
                f"  Rel L2: {jnp.linalg.norm(results[0] - results[i]) / jnp.linalg.norm(results[0])}"
            )


# ============================================================================
# Test MSGB Solver - Frame Equivalence
# ============================================================================


class TestMSGBSolverFrameEquivalence:
    """Test that GB solver correctly implements frame decomposition."""

    def test_frame_reconstruction_matches_gb_solver(
        self, simple_domain_2d, dyadic_decomp_2d, wpt_2d
    ):
        """
        Verify that manual frame reconstruction equals GB solver output.

        This tests that:
        1. MSWPT forward transform
        2. Truncated inverse with windowing="none"
        3. GB solver forward
        All produce consistent results.
        """
        domain = simple_domain_2d

        # Create sparse coefficient vector
        total_coeffs = wpt_2d.total_coeffs
        coeffs = jnp.zeros(total_coeffs)

        # Add a few random coefficients
        num_nonzero = 2
        for i in range(num_nonzero):
            key = jax.random.PRNGKey(i)
            idx = jax.random.randint(key, (), 0, total_coeffs)
            coeffs = coeffs.at[idx].set(1.0)

        # Manual reconstruction from frames
        shapes = utils.compute_coeff_shapes(
            dyadic_decomp_2d, wpt_2d.redundancy, jnp.arange(dyadic_decomp_2d.num_levels)
        )

        nonzero_indices = jnp.where(coeffs != 0)[0]
        nn_level, nn_indices = utils.find_tensor_and_multiindex(nonzero_indices, shapes)

        cumsum_boxes = jnp.r_[0, jnp.cumsum(dyadic_decomp_2d.num_boxes_ndim)]
        box_indices = nn_indices[0, :] + cumsum_boxes[nn_level]
        k_values = nn_indices[1:, :]

        KXY = dyadic_decomp_2d.fourier_meshgrid
        manual_recon = 0
        for i in range(len(nonzero_indices)):
            frame_ft = transforms.compute_frames(
                dyadic_decomp_2d,
                box_indices[i],
                k_values[:, i],
                KXY,
                wpt_2d.redundancy,
                "none",
            )
            manual_recon += utils.unitary_ifft(frame_ft)

        # GB solver reconstruction
        p0s, M0s, x0s, ωs, a0s, modes = forward_solver_utils.compute_forward_parameters(
            nonzero_indices, wpt_2d, domain
        )
        a0s = a0s * coeffs[nonzero_indices]

        gb_recon = jnp.sum(
            core.compute_gaussian_beam(
                x0=x0s,
                p0=p0s,
                M0=M0s,
                a0=a0s,
                omega0=ωs,
                mode=modes,
                c=domain.c,
                lam=0,
                ts=jnp.array([0.0]),
                sensors=domain.grid,
                domain_size=domain.grid_size,
                periodic=jnp.array(domain.periodic),
                ode_solver=gb_solvers.solve_ODE_base,
                solver_config=None,
            ),
            axis=-1,
        )[0, ...]

        # Check equivalence
        max_error = jnp.max(jnp.abs(gb_recon - manual_recon))
        rel_error = jnp.linalg.norm(gb_recon - manual_recon) / jnp.linalg.norm(
            manual_recon
        )

        assert jnp.allclose(gb_recon, manual_recon, atol=1e-14), (
            f"GB solver doesn't match manual frame reconstruction:\n"
            f"  Max error: {max_error}\n"
            f"  Rel error: {rel_error}"
        )


# ============================================================================
# Test MSGB Solver - Accuracy
# ============================================================================


class TestMSGBSolverAccuracy:
    """Test accuracy and convergence properties of MSGB solver."""

    @pytest.mark.parametrize(
        "N,threshold_list",
        [
            # Two thresholds are enough to verify the monotonic-improvement
            # property in each dimension; previously this swept four
            # thresholds in 2D, which was the single slowest non-fixture call.
            ((128,), [100, 400]),
            ((64, 128), [1000, 4000]),
        ],
    )
    def test_accuracy_improves_with_threshold(self, N, threshold_list):
        """Verify that increasing threshold improves approximation accuracy."""
        d = len(N)
        domain = geometry.Domain(
            N=N, dx=(1e-4,) * d, c=lambda x: 1 + 0 * x[..., 0], periodic=(True,) * d
        )
        sensors = geometry.Sensor(binary_mask=jnp.ones(N), domain=domain)

        dyadic = DyadicDecomposition(2, N, (4, 8), (1,) * d)
        wpt = MSWPT(dyadic, 2, "rectangular")

        # Create test signal
        p0 = create_test_signal(dyadic, wpt)
        jnp.zeros_like(p0)

        # Test increasing thresholds
        errors = []
        ts = jnp.array([0.0])

        for threshold in threshold_list:
            solver = MSGBSolver(
                thr=threshold,
                thr_strat="top_n",
                batch_size=100,
                input_type="spatial",
                ode_solver=gb_solvers.solve_hom_diag,
                tr_ode_solver=gb_solvers.solve_ODE_batch_t,
                sum_method="scan_real",
            )

            gb_result = solver.forward(p0, domain, sensors, ts, wpt)[0, ...].reshape(N)
            error = jnp.linalg.norm(gb_result - p0.real) / jnp.linalg.norm(p0.real)
            errors.append(error)

        # Verify monotonic improvement
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i - 1] * 1.1, (  # Allow 10% tolerance
                f"Error increased from {errors[i - 1]:.4e} to {errors[i]:.4e} "
                f"at threshold {threshold_list[i]}"
            )

        # Final error should be reasonable
        # assert errors[-1] < 0.9, f"Final error {errors[-1]} too high"


# ============================================================================
# Test Hybrid Solver - Basic Functionality
# ============================================================================


@requires_kwave
@requires_kwave_cpp_binary
class TestHybridSolverBasics:
    """Test basic hybrid solver functionality."""

    @pytest.fixture
    def kwave_solver(self):
        """k-Wave solver for testing."""
        sim_opts = SimulationOptions(
            data_cast="double",
            smooth_p0=False,
            save_to_disk=True,
        )
        exec_opts = SimulationExecutionOptions(
            is_gpu_simulation=False,
            delete_data=False,
            verbose_level=0,
            show_sim_log=False,
        )
        return KWaveSolver(sim_opts, exec_opts)

    # `dt_oversample` is forced to 0 when `downsample=False` inside the test
    # below, so the (downsample=False, dt_oversample=30) combination is
    # identical to (downsample=False, dt_oversample=0). We enumerate the
    # three distinct cases explicitly instead of the 2x2 Cartesian product.
    @pytest.mark.parametrize(
        "downsample, dt_oversample",
        [(True, 0), (True, 30), (False, 0)],
    )
    def test_hybrid_matches_kwave(
        self,
        simple_domain_2d,
        dyadic_decomp_2d,
        kwave_solver,
        downsample,
        dt_oversample,
    ):
        """Verify hybrid solver matches k-Wave reference."""
        domain = simple_domain_2d
        ts = domain.generate_time_domain()

        wpt = MSWPT(dyadic_decomp_2d, 2, "rectangular_mirror")

        binary_mask = jnp.zeros(domain.N)
        binary_mask = binary_mask.at[0, :].set(1)
        sensors = geometry.Sensor(binary_mask=binary_mask, domain=domain)

        p0 = jnp.zeros(domain.N)
        p0 = p0.at[30, 40].set(100.0)

        hybrid = HybridSolver(
            hf_solver=kwave_solver,
            lf_backend=HybridBackend.from_beamax_solver(kwave_solver),
            box_corners=jnp.array([0, 15]),
            downsample=downsample,
            dt_oversample=dt_oversample if downsample else 0,
            interp_method="fourier",
        )

        kwave_result = kwave_solver.forward(p0, domain, binary_mask, ts)
        hybrid_result = hybrid.forward(p0, domain, sensors, ts, wpt)

        assert jnp.allclose(kwave_result, hybrid_result, atol=3e-5), (
            f"Hybrid (downsample={downsample}, dt_oversample={dt_oversample}) "
            f"doesn't match k-Wave:\n"
            f"  Max diff: {jnp.max(jnp.abs(kwave_result - hybrid_result))}\n"
            f"  Rel L2: {jnp.linalg.norm(kwave_result - hybrid_result) / jnp.linalg.norm(kwave_result)}"
        )


# ============================================================================
# Test Hybrid Solver - Advanced Features
# ============================================================================


class TestHybridSolverAdvanced:
    """Test advanced hybrid solver features."""

    @pytest.fixture
    def dummy_solver(self):
        """Dummy solver for testing configuration."""

        class DummySolver:
            def forward(self, *args, **kwargs):
                return jnp.zeros((10, 128))

            def time_reversal(self, *args, **kwargs):
                return jnp.zeros((128, 128))

        return DummySolver()

    def test_config_object_initialization(self, dummy_solver):
        """Test initialization with HybridSolverConfig object."""
        config = HybridSolverConfig(
            box_corners=jnp.array([0, 15]),
            downsample=True,
            interp_method="zoom",
            order=5,
        )

        solver = HybridSolver(
            hf_solver=dummy_solver,
            lf_backend=HybridBackend.from_beamax_solver(dummy_solver),
            config=config,
        )

        assert solver.config.downsample
        assert solver.config.interp_method == "zoom"
        assert solver.config.order == 5

    def test_kwargs_initialization(self, dummy_solver):
        """Test initialization with kwargs."""
        solver = HybridSolver(
            hf_solver=dummy_solver,
            lf_backend=HybridBackend.from_beamax_solver(dummy_solver),
            box_corners=jnp.array([0, 15]),
            downsample=False,
            interp_method="fourier",
        )

        assert not solver.config.downsample
        assert solver.config.interp_method == "fourier"

    def test_factory_method_periodic_domain(self, dummy_solver):
        """Test factory method auto-selects Fourier for periodic domain."""
        domain = geometry.Domain(
            N=(64, 64), dx=(1e-4, 1e-4), c=1500.0, periodic=(True, True)
        )

        solver = HybridSolver.create_with_domain(
            hf_solver=dummy_solver,
            lf_backend=HybridBackend.from_beamax_solver(dummy_solver),
            domain=domain,
            box_corners=jnp.array([0, 15]),
        )

        assert solver.config.interp_method == "fourier"

    def test_factory_method_nonperiodic_domain(self, dummy_solver):
        """Test factory method auto-selects zoom for non-periodic domain."""
        domain = geometry.Domain(
            N=(64, 64), dx=(1e-4, 1e-4), c=1500.0, periodic=(False, False)
        )

        solver = HybridSolver.create_with_domain(
            hf_solver=dummy_solver,
            lf_backend=HybridBackend.from_beamax_solver(dummy_solver),
            domain=domain,
            box_corners=jnp.array([0, 15]),
        )

        assert solver.config.interp_method == "zoom"
        assert solver.config.order == 3  # Default cubic

    def test_validation_warning_fourier_nonperiodic(
        self, dummy_solver, simple_domain_2d
    ):
        """Test that warning is issued for Fourier + non-periodic."""
        # Make non-periodic
        domain = geometry.Domain(
            N=simple_domain_2d.N,
            dx=simple_domain_2d.dx,
            c=simple_domain_2d.c,
            periodic=(False, False),
        )

        solver = HybridSolver(
            hf_solver=dummy_solver,
            lf_backend=HybridBackend.from_beamax_solver(dummy_solver),
            box_corners=jnp.array([0, 15]),
            downsample=True,
            interp_method="fourier",  # Explicitly use Fourier
        )

        # Should warn when forward is called
        dyadic = DyadicDecomposition(2, domain.N, (4, 8), (1, 1))
        wpt = MSWPT(dyadic, 2, "rectangular")
        sensors = geometry.Sensor(binary_mask=jnp.ones(domain.N), domain=domain)
        p0 = jnp.zeros(domain.N)
        ts = jnp.array([0.0, 1e-4])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.forward(p0, domain, sensors, ts, wpt)
            assert len(w) == 1
            assert "non-periodic domain" in str(w[0].message).lower()

    @pytest.mark.parametrize("window_type", ["kaiser", "tukey"])
    def test_window_types(self, dummy_solver, window_type):
        """Test different windowing functions."""
        solver = HybridSolver(
            hf_solver=dummy_solver,
            lf_backend=HybridBackend.from_beamax_solver(dummy_solver),
            box_corners=jnp.array([0, 15]),
            window_type=window_type,
            dt_oversample=20,
        )

        assert solver.config.window_type == window_type

        # Test windowing actually works
        data = jnp.ones((50, 10))
        windowed = solver._apply_window(data)

        # Window should taper at the end
        assert jnp.all(windowed[-20:, 0] <= 1.0)
        assert jnp.all(windowed[:-20, 0] == 1.0)

    @pytest.mark.parametrize(
        "interp_method,order",
        [
            ("fourier", 3),
            ("zoom", 1),
            ("zoom", 3),
            ("zoom", 5),
        ],
    )
    def test_interpolation_methods(self, dummy_solver, interp_method, order):
        """Test different interpolation methods."""
        solver = HybridSolver(
            hf_solver=dummy_solver,
            lf_backend=HybridBackend.from_beamax_solver(dummy_solver),
            box_corners=jnp.array([0, 15]),
            interp_method=interp_method,
            order=order,
        )

        assert solver.config.interp_method == interp_method
        assert solver.config.order == order


@requires_kwave
@requires_kwave_cpp_binary
def test_hybrid_downsample():
    """
    Check that Fourier-interpolated downsampling preserves hybrid solver output.
    """
    d = 2
    N = (128,) * d
    dx = (1e-4,) * d
    periodic = (True,) * d
    box_aspect_ratio = (1,) * d
    num_levels = 3
    num_boxes_levels = tuple([2 ** (i + 2) for i in range(num_levels)])

    def c(x):
        return 1500 + 0 * x[..., 0]

    windowing = "rectangular_mirror"
    input_type = "spatial"
    redundancy = 2

    cfl = (jnp.sqrt(2) / 4).round(3)
    domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

    ts = domain.generate_time_domain()

    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_levels, box_aspect_ratio
    )
    wpt = MSWPT(dyadic_decomp, redundancy, windowing)

    binary_mask = jnp.zeros(N)
    binary_mask = binary_mask.at[0, ...].set(1)
    sensors = geometry.Sensor(binary_mask=binary_mask, domain=domain)

    kxy = dyadic_decomp.fourier_meshgrid

    boxhf = 44
    boxlf = 10
    khf = jnp.array([10, 12])
    klf = jnp.array([10, 3])
    kerft_hf = transforms.compute_frames(
        dyadic_decomp, boxhf, khf, kxy, redundancy, "none"
    )
    kerft_lf = transforms.compute_frames(
        dyadic_decomp, boxlf, klf, kxy, redundancy, "none"
    )
    p0 = utils.unitary_ifft(kerft_hf) + utils.unitary_ifft(kerft_lf)
    p0 = p0 / jnp.max(jnp.abs(p0))
    p0 = p0.T.real

    msgb_solver = MSGBSolver(
        thr=1000,
        thr_strat="top_n",
        batch_size=100,
        input_type=input_type,
        ode_solver=gb_solvers.solve_ODE_base,
        tr_ode_solver=gb_solvers.solve_ODE_batch_t,
        sum_method="scan_real",
    )

    simulation_options = SimulationOptions(
        data_cast="double",
        smooth_p0=False,
        save_to_disk=True,
    )

    execution_options = SimulationExecutionOptions(
        is_gpu_simulation=False, delete_data=False, verbose_level=0, show_sim_log=False
    )

    kwave_solver = KWaveSolver(simulation_options, execution_options)

    box_corners = jnp.array([0, 15])
    cutoff_freq = None

    hybrid_solver = HybridSolver(
        hf_solver=msgb_solver,
        lf_backend=HybridBackend.from_beamax_solver(kwave_solver),
        downsample=False,
        box_corners=box_corners,
        cutoff_freq=cutoff_freq,
        input_type=input_type,
        interp_method="fourier",
        dt_oversample=0,
    )

    hybrid_solver_downsample = HybridSolver(
        hf_solver=msgb_solver,
        lf_backend=HybridBackend.from_beamax_solver(kwave_solver),
        downsample=True,
        box_corners=box_corners,
        cutoff_freq=cutoff_freq,
        input_type=input_type,
        interp_method="fourier",
        dt_oversample=50,
    )

    hybrid_data = hybrid_solver.forward(p0, domain, sensors, ts, wpt)
    hybrid_data_downsample = hybrid_solver_downsample.forward(
        p0, domain, sensors, ts, wpt
    )

    assert jnp.allclose(hybrid_data_downsample, hybrid_data, atol=1e-5)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
