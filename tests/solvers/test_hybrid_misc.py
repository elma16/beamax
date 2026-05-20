"""
Unit tests for thin validation / configuration / dispatch paths in
``hybrid_solver`` and ``hybrid_solver_utils`` that the existing integration
tests don't reach.

These are deliberately small and fast: no k-Wave, no MSGB, no real forward
solves. The goal is to lock in the error/validation behaviour and the
analytic-helper edge cases so the heavier integration tests don't have to
re-cover them.
"""

import warnings

import jax.numpy as jnp
import pytest

from beamax import geometry
from beamax.solvers.hybrid_solver import (
    FourierInterpolation,
    HybridSolver,
    HybridSolverConfig,
    ZoomInterpolation,
)
from beamax.solvers.hybrid_solver_utils import (
    are_opposing,
    find_bounding_corner_indices,
    get_indices_between_two_opposing_corners,
    get_indices_with_norm_less_than,
    split_frequency_components,
)


# ---------------------------------------------------------------------------
# HybridSolverConfig validation
# ---------------------------------------------------------------------------


class TestHybridSolverConfigValidation:
    def test_missing_both_split_specs_raises(self):
        with pytest.raises(ValueError, match="box_corners or cutoff_freq"):
            HybridSolverConfig()

    def test_both_split_specs_raises(self):
        with pytest.raises(ValueError, match="only one"):
            HybridSolverConfig(box_corners=jnp.array([0, 1]), cutoff_freq=0.5)

    def test_order_out_of_range_raises(self):
        with pytest.raises(ValueError, match="order must be 0-5"):
            HybridSolverConfig(box_corners=jnp.array([0, 1]), order=99)

    def test_unknown_window_type_raises(self):
        with pytest.raises(ValueError, match="window_type"):
            HybridSolverConfig(
                box_corners=jnp.array([0, 1]),
                window_type="hamming",
            )


class _DummySolver:
    """No-op solver used to construct HybridSolver instances without a real backend."""

    def forward(self, *args, **kwargs):
        return jnp.zeros((4, 4))

    def time_reversal(self, *args, **kwargs):
        return jnp.zeros((4, 4))

    def adjoint(self, *args, **kwargs):
        return jnp.zeros((4, 4))


def test_hybrid_solver_unknown_interp_method_raises():
    """The constructor must reject an unrecognised interp_method."""
    with pytest.raises(ValueError, match="interp_method"):
        HybridSolver(
            _DummySolver(),
            _DummySolver(),
            box_corners=jnp.array([0, 1]),
            interp_method="bilinear",  # not 'fourier' or 'zoom'
        )


# ---------------------------------------------------------------------------
# Interpolation strategies (the public abstract classes both have concrete
# implementations that are simple enough to test directly).
# ---------------------------------------------------------------------------


def test_fourier_interpolation_upsamples_constant_array():
    """A constant input stays spatially flat after Fourier resampling.

    The unitary FFT used internally rescales the DC term by ``N/N_target`` so
    the pointwise value is not preserved, but the output is still spatially
    uniform — that's the property we care about for interpolation correctness.
    """
    arr = jnp.ones((4, 4))
    out = FourierInterpolation().interpolate(arr, (8, 8))
    assert out.shape == (8, 8)
    assert float(jnp.std(out)) < 1e-8


def test_zoom_interpolation_returns_target_shape():
    """ZoomInterpolation must return exactly the requested shape."""
    arr = jnp.arange(16.0).reshape(4, 4)
    out = ZoomInterpolation(order=3).interpolate(arr, (8, 8))
    assert out.shape == (8, 8)


# ---------------------------------------------------------------------------
# Window helpers — edge cases that bypass the kaiser/tukey tapering.
# ---------------------------------------------------------------------------


def test_apply_kaiser_window_zero_oversample_is_passthrough():
    """dt_oversample=0 must short-circuit the Kaiser window."""
    solver = HybridSolver(
        _DummySolver(),
        _DummySolver(),
        box_corners=jnp.array([0, 1]),
        window_type="kaiser",
        dt_oversample=0,
    )
    data = jnp.ones((10, 3))
    out = solver._apply_kaiser_window(data)
    assert jnp.array_equal(out, data)


def test_apply_tukey_window_zero_oversample_is_passthrough():
    """dt_oversample=0 must short-circuit the Tukey window."""
    solver = HybridSolver(
        _DummySolver(),
        _DummySolver(),
        box_corners=jnp.array([0, 1]),
        window_type="tukey",
        dt_oversample=0,
    )
    data = jnp.ones((10, 3))
    out = solver._apply_tukey_window(data)
    assert jnp.array_equal(out, data)


def test_apply_kaiser_window_1d_input():
    """The Kaiser path must also handle 1D (time-only) inputs."""
    solver = HybridSolver(
        _DummySolver(),
        _DummySolver(),
        box_corners=jnp.array([0, 1]),
        window_type="kaiser",
        dt_oversample=4,
    )
    data = jnp.ones((20,))
    out = solver._apply_kaiser_window(data)
    assert out.shape == data.shape
    assert float(out[-1]) < 1.0  # taper kicks in at the end


def test_apply_tukey_window_1d_input():
    """The Tukey path must also handle 1D (time-only) inputs."""
    solver = HybridSolver(
        _DummySolver(),
        _DummySolver(),
        box_corners=jnp.array([0, 1]),
        window_type="tukey",
        dt_oversample=10,
    )
    data = jnp.ones((30,))
    out = solver._apply_tukey_window(data)
    assert out.shape == data.shape
    assert float(out[-1]) < 1.0


# ---------------------------------------------------------------------------
# hybrid_solver_utils helpers
# ---------------------------------------------------------------------------


def test_find_bounding_corner_indices_empty_raises():
    """find_bounding_corner_indices must reject empty index sets."""
    centers = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="empty"):
        find_bounding_corner_indices(centers, jnp.array([], dtype=jnp.int32))


def test_find_bounding_corner_indices_single_point():
    """A single index returns it twice (caller is expected to handle this)."""
    centers = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    c1, c2 = find_bounding_corner_indices(centers, jnp.array([1]))
    assert c1 == c2 == 1


def test_find_bounding_corner_indices_1d_falls_back_to_extremes():
    """In 1D, both 'min' and 'max' corners coincide; the fallback path picks the
    furthest point so the two corners actually differ.
    """
    centers = jnp.array([[-3.0], [-1.0], [2.0], [5.0]])
    c1, c2 = find_bounding_corner_indices(centers, jnp.arange(4))
    assert c1 != c2  # fallback fired
    assert {c1, c2} == {0, 3}


def test_are_opposing_smoke():
    """are_opposing should accept any two ints and return a bool."""
    assert isinstance(are_opposing(0, 1), bool) or isinstance(
        are_opposing(0, 1), (jnp.ndarray,)
    )


def test_get_indices_between_two_opposing_corners_smoke():
    """Lattice-like centres: the helper returns indices forming a bounding box."""
    centers = jnp.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]],
        dtype=jnp.float64,
    )
    # corner 0 = (0,0), corner 3 = (1,1); the bounding box covers {0,1,2,3}.
    idx = get_indices_between_two_opposing_corners(centers, 0, 3)
    idx_set = set(int(i) for i in idx)
    assert 0 in idx_set
    assert 3 in idx_set
    assert 4 not in idx_set  # outside the box


def test_get_indices_with_norm_less_than_inclusive_vs_exclusive():
    """Sanity-check the inclusive flag (used by split_frequency_components)."""
    centers = jnp.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    incl = set(
        int(i) for i in get_indices_with_norm_less_than(centers, 1.0, inclusive=True)
    )
    excl = set(
        int(i) for i in get_indices_with_norm_less_than(centers, 1.0, inclusive=False)
    )
    assert 2 in incl
    assert 2 not in excl


def test_split_frequency_components_requires_exactly_one_split_spec():
    """Both/neither cutoff_freq and box_corners → ValueError."""
    from beamax.decomposition import DyadicDecomposition
    from beamax.transforms import MSWPT

    dyadic = DyadicDecomposition(
        num_levels=2,
        N=(32, 32),
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1, 1),
    )
    wpt = MSWPT(dyadic, redundancy=2, windowing="rectangular")
    p0 = jnp.zeros((32, 32))
    sensors_mask = jnp.ones((32, 32))
    domain = geometry.Domain(
        N=(32, 32),
        dx=(1.0, 1.0),
        c=lambda x: 1.0 + 0.0 * x[..., 0],
        periodic=(True, True),
    )

    with pytest.raises(ValueError, match="Exactly one"):
        split_frequency_components(
            p0=p0,
            input_type="spatial",
            output_type="spatial",
            wpt=wpt,
            sensors_mask=sensors_mask,
            domain=domain,
            windowing="rectangular",
            cutoff_freq=None,
            box_corners=None,
        )


def test_split_frequency_components_empty_lf_warns_and_returns_zero_lf():
    """When the LF index set is empty, the helper warns and returns p0_LF = 0."""
    from beamax.decomposition import DyadicDecomposition
    from beamax.transforms import MSWPT

    dyadic = DyadicDecomposition(
        num_levels=2,
        N=(32, 32),
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1, 1),
    )
    wpt = MSWPT(dyadic, redundancy=2, windowing="rectangular")
    p0 = jnp.ones((32, 32))
    sensors_mask = jnp.ones((32, 32))
    domain = geometry.Domain(
        N=(32, 32),
        dx=(1.0, 1.0),
        c=lambda x: 1.0 + 0.0 * x[..., 0],
        periodic=(True, True),
    )
    # cutoff_freq below the smallest dyadic centre norm → empty idx_box.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p0_hf, p0_lf, mask_out, dom_out = split_frequency_components(
            p0=p0,
            input_type="spatial",
            output_type="spatial",
            wpt=wpt,
            sensors_mask=sensors_mask,
            domain=domain,
            windowing="rectangular",
            cutoff_freq=1e-12,
            box_corners=None,
        )
    assert any("empty" in str(w.message).lower() for w in caught)
    assert jnp.allclose(p0_lf, 0.0)
    assert mask_out is sensors_mask
    assert dom_out is domain


if __name__ == "__main__":
    pytest.main([__file__])
