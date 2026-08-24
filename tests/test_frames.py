from beamax import transforms, utils
from beamax.gb import core, gb_utils, gb_solvers
from beamax.geometry import Domain
from beamax.decomposition import DyadicDecomposition
import jax.numpy as jnp
import jax
import pytest
from einops import rearrange

jax.config.update("jax_enable_x64", True)

redundancy = 2

common_params = [
    (
        num_levels,
        N,
        tuple([2 ** (level + 2) for level in range(num_levels)]),
        (1,) * len(N),
    )
    for num_levels in range(1, 3)
    for N in [
        (128,),
        (128, 128),
        (256, 128),
        (128, 256),
    ]
]

representative_params = [
    (1, (128,), (4,), (1,)),
    (1, (128, 128), (4,), (1, 1)),
    (2, (128, 256), (4, 8), (1, 1)),
]

multiscale_params = [
    (2, (128,), (4, 8), (1,)),
    (2, (128, 256), (4, 8), (1, 1)),
]

single_coeff_params = [
    (1, (128,), (4,), (1,)),
    (2, (128, 128), (4, 8), (1, 1)),
]


def compute_frames_custom(
    dyadic_decomp: DyadicDecomposition,
    centre: jnp.ndarray,
    level: int,
    k: jnp.ndarray,
    fourier_space: jnp.ndarray,
    windowing: str,
) -> jnp.ndarray:
    box_length = dyadic_decomp.box_lengths[level]
    support_lengths = (
        box_length * redundancy * jnp.asarray(dyadic_decomp.box_aspect_ratio)
    )
    half_support = support_lengths // 2
    grid_shape = jnp.asarray(dyadic_decomp.N)
    parity_index = (centre + grid_shape // 2) // half_support
    parity_sign = 1 - 2 * (parity_index & 1)
    rolls = parity_sign * (half_support // 2)

    g = transforms.single_filter_coord(
        centre,
        level,
        dyadic_decomp.fourier_meshgrid,
        dyadic_decomp,
        redundancy,
        windowing,
    )

    scaled_phase = jnp.einsum(
        "...d,d->...", fourier_space - centre + rolls, k / support_lengths
    )

    return (
        jnp.prod(support_lengths) ** (-0.5)
        * jnp.exp(-2 * jnp.pi * 1j * scaled_phase)
        * g
    )


@pytest.mark.parametrize(
    "num_levels, N, num_boxes_outer_level, box_aspect_ratio", multiscale_params
)
def test_frame_cutoff_error(num_levels, N, num_boxes_outer_level, box_aspect_ratio):
    """
    Test that the error of the frames decreases as the scale increases (i.e the cutoff is scale dependent).
    Test that the error of the frames is the same for all boxes at a given scale (i.e the cutoff is not frequency dependent).
    """
    ndim = len(N)
    redundancy = 2

    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_outer_level, box_aspect_ratio
    )

    KXY = dyadic_decomp.fourier_meshgrid

    k = jnp.ones((ndim,))

    def compute_error(idx):
        idx = jnp.array([idx])
        phi_r = transforms.compute_frames(
            dyadic_decomp, idx, k, KXY, redundancy, "rectangular"
        )
        phi_r_x = utils.unitary_ifft(phi_r)
        phi_n = transforms.compute_frames(
            dyadic_decomp, idx, k, KXY, redundancy, "none"
        )
        phi_n_x = utils.unitary_ifft(phi_n)
        error = jnp.linalg.norm(phi_r_x - phi_n_x)
        return error

    indices = jnp.arange(jnp.sum(dyadic_decomp.num_boxes_ndim))

    errors = jax.vmap(compute_error)(indices)

    boxes_cumsum = jnp.concatenate(
        [jnp.array([0]), dyadic_decomp.num_boxes_ndim_cumsum]
    )

    for scale in range(num_levels):
        errors_scale = errors[boxes_cumsum[scale] : boxes_cumsum[scale + 1]]
        assert jnp.allclose(errors_scale, errors_scale[0], atol=1e-16), (
            f"cutoff error varies within level {scale}"
        )

    errors_allscales = errors[boxes_cumsum[:-1]]

    assert jnp.all(errors_allscales[:-1] >= errors_allscales[1:])


@pytest.mark.parametrize(
    "num_levels, N, num_boxes_outer_level, box_aspect_ratio", representative_params
)
def test_frames_conjugate_pairs_custom(
    num_levels, N, num_boxes_outer_level, box_aspect_ratio
):
    """Mirrored rectangular frames form conjugate pairs."""

    windowing = "rectangular_mirror"
    ndim = len(N)
    dx = (1e-4,) * ndim

    def c(x):
        return 1500 + 0 * x[..., 0]

    periodic = (False,) * ndim
    domain = Domain(N=N, dx=dx, c=c, periodic=periodic)
    _, fourier_space = domain.generate_meshgrid()

    KXY = jnp.stack(fourier_space, axis=-1)

    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_outer_level, box_aspect_ratio
    )

    limits = jnp.asarray(N) // 2
    trial_keys = jax.random.split(jax.random.PRNGKey(0), 2 * num_levels)

    for trial, trial_key in enumerate(trial_keys):
        x_key, k_key = jax.random.split(trial_key)
        level = trial // 2
        x = jax.random.randint(x_key, (ndim,), -limits, limits)
        k = jax.random.randint(k_key, (ndim,), -limits, limits)

        phi_x = utils.unitary_ifft(
            compute_frames_custom(dyadic_decomp, x, level, k, KXY, windowing)
        )
        phi_minus_x = utils.unitary_ifft(
            compute_frames_custom(dyadic_decomp, -x, level, k, KXY, windowing)
        )

        assert jnp.allclose(phi_x, jnp.conj(phi_minus_x), atol=1e-16), (
            f"conjugacy failed at level {level}, trial {trial % 2}"
        )


@pytest.mark.parametrize(
    "num_levels, N, num_boxes_outer_level, box_aspect_ratio", common_params
)
def test_frames_conjugate_pairs(num_levels, N, num_boxes_outer_level, box_aspect_ratio):
    """
    Test that the frames are conjugate pairs for the rectangular windowing function and no windowing.

    This works for both windowing == none and windowing == rectangular_mirror
    """

    windowing = "none"

    ndim = len(N)
    dx = (1e-4,) * ndim

    def c(x):
        return 1500 + 0 * x[..., 0]

    periodic = (False,) * ndim
    domain = Domain(N=N, dx=dx, c=c, periodic=periodic)

    _, fourier_space = domain.generate_meshgrid()

    KXY = jnp.stack(fourier_space, axis=-1)

    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_outer_level, box_aspect_ratio
    )
    num_boxes_level = dyadic_decomp.num_boxes_ndim_cumsum - 1
    num_boxes_level = jnp.concatenate([jnp.array([0]), num_boxes_level], axis=0)

    failures = []
    for idx in range(dyadic_decomp.total_num_boxes):
        k = jnp.array((3,) * ndim)
        level = utils.find_level(dyadic_decomp, idx)
        idx_conj = num_boxes_level[level + 1] - idx + num_boxes_level[level]
        if level > 0:
            idx_conj = idx_conj + 1

        phi_hat_x = transforms.compute_frames(
            dyadic_decomp, idx, k, KXY, redundancy, windowing
        )
        phi_hat_minus_x = transforms.compute_frames(
            dyadic_decomp, idx_conj, k, KXY, redundancy, windowing
        )
        shape = phi_hat_x.shape
        indices = jnp.indices(shape)
        sym_indices = tuple(
            (shape[axis] - indices[axis]) % shape[axis]
            for axis in range(phi_hat_x.ndim)
        )
        phi_hat_minus_x_flip = phi_hat_minus_x[sym_indices]

        phi_x = utils.unitary_ifft(phi_hat_x)
        phi_minus_x = utils.unitary_ifft(phi_hat_minus_x)
        spatial_ok = jnp.allclose(phi_x, jnp.conj(phi_minus_x), atol=1e-15)
        fourier_ok = jnp.allclose(phi_hat_x, jnp.conj(phi_hat_minus_x_flip), atol=1e-15)
        if not bool(spatial_ok and fourier_ok):
            failures.append((idx, int(idx_conj), bool(spatial_ok), bool(fourier_ok)))

    assert not failures, f"non-conjugate frame pairs: {failures[:10]}"


@pytest.mark.parametrize(
    "num_levels, N, num_boxes_outer_level, box_aspect_ratio", representative_params
)
def test_none_windowing_frame_eq_GB(
    num_levels, N, num_boxes_outer_level, box_aspect_ratio
):
    """An unwindowed frame equals its corresponding Gaussian beam."""
    windowing = "none"
    ndim = len(N)
    dx = (1e-4,) * ndim

    def c(x):
        return 1500 + 0 * x[..., 0]

    periodic = (False,) * ndim
    domain = Domain(N=N, dx=dx, c=c, periodic=periodic)
    space, fourier_space = domain.generate_meshgrid()
    XY = jnp.stack(space, axis=-1)
    KXY = jnp.stack(fourier_space, axis=-1)

    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_outer_level, box_aspect_ratio
    )

    level_starts = jnp.r_[0, dyadic_decomp.num_boxes_ndim_cumsum[:-1]]
    level_keys = jax.random.split(jax.random.PRNGKey(0), num_levels)
    limits = jnp.asarray(N) // 2

    for level, level_key in enumerate(level_keys):
        box_key, k_key = jax.random.split(level_key)
        boxidx = level_starts[level] + jax.random.randint(
            box_key, (), 0, dyadic_decomp.num_boxes_ndim[level]
        )
        k = (
            jax.random.randint(k_key, (ndim,), -limits, limits)
            % dyadic_decomp.box_lengths[level]
        )

        centres_ndim = dyadic_decomp.centres_ndim
        grid_sizes = domain.grid_size
        box_lengths = jnp.array(dyadic_decomp.box_lengths)
        box_aspect_ratio_array = jnp.array(dyadic_decomp.box_aspect_ratio)
        grid_shape = jnp.array(dyadic_decomp.N)

        phi_kx = transforms.compute_frames(
            dyadic_decomp, boxidx, k, KXY, redundancy, windowing
        )
        phi_x = utils.unitary_ifft(phi_kx)

        box_centres = centres_ndim[boxidx, :] / grid_sizes
        normxis = jnp.linalg.norm(box_centres, axis=-1, keepdims=True)

        bl = box_lengths[level] / grid_sizes * box_aspect_ratio_array
        Lls = bl * 2
        sigmas = bl / 2

        p0s = 2 * jnp.pi * box_centres / normxis
        alpha0 = 2j * (jnp.pi * sigmas) ** 2 / normxis
        a0s = jnp.prod(
            jnp.sqrt(
                (jnp.pi * rearrange(grid_sizes, "d -> 1 d"))
                / (Lls * rearrange(grid_shape, "d -> 1 d"))
            )
            * sigmas,
            axis=1,
        )

        x0 = k / Lls
        ts = jnp.linspace(0, 0.01, 10)

        p0s = rearrange(p0s, "d -> 1 d")
        alpha0 = rearrange(alpha0, "d -> 1 d")
        x0s = rearrange(x0, "d -> 1 d")

        M0s = gb_utils.prepare_M0(alpha0, None)
        gb_x = core.compute_gaussian_beam(
            x0s,
            p0s,
            M0s,
            a0s,
            normxis,
            jnp.ones((1,)),
            c,
            0,
            ts,
            XY,
            domain.grid_size,
            jnp.ones(ndim, dtype=bool),
            gb_solvers.solve_hom_diag,
            None,
        )

        atol = 1e-7 if num_levels == 1 else 1e-15
        assert jnp.allclose(phi_x, gb_x[0, ..., 0], atol=atol), (
            f"frame/beam mismatch at level {level}, box {int(boxidx)}"
        )


@pytest.mark.parametrize(
    "num_levels, N, num_boxes_outer_level, box_aspect_ratio", single_coeff_params
)
def test_single_coeff_eq_frame(num_levels, N, num_boxes_outer_level, box_aspect_ratio):
    """
    Test that a single coefficient is equal to a frame with certain parameters.
    """
    windowing = "rectangular_mirror"
    output_type = "fourier"
    ndim = len(N)
    dx = (1e-4,) * ndim

    def c(x):
        return 1500 + 0 * x[..., 0]

    periodic = (False,) * ndim

    domain = Domain(N=N, dx=dx, c=c, periodic=periodic)
    _, fourier_space = domain.generate_meshgrid()

    KXY = jnp.stack(fourier_space, axis=-1)

    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_outer_level, box_aspect_ratio
    )
    wpt = transforms.MSWPT(dyadic_decomp, redundancy, windowing)

    shapes = utils.compute_coeff_shapes(
        dyadic_decomp, redundancy, jnp.arange(dyadic_decomp.num_levels)
    )
    coeff_offsets = jnp.r_[0, jnp.cumsum(jnp.prod(shapes, axis=1))]
    cumsum_boxes = jnp.r_[0, jnp.cumsum(dyadic_decomp.num_boxes_ndim)]
    level_keys = jax.random.split(jax.random.PRNGKey(0), num_levels)

    for level, level_key in enumerate(level_keys):
        coeff_idx = jax.random.randint(
            level_key, (), coeff_offsets[level], coeff_offsets[level + 1]
        )
        coeffs = jnp.zeros(wpt.total_coeffs).at[coeff_idx].set(1.0)
        coeff_idx = jnp.array([coeff_idx])

        f_rect = wpt.inverse(coeffs, output_type)
        nn_level, nn_indices = utils.find_tensor_and_multiindex(coeff_idx, shapes)

        idx_guess = nn_indices[0, :] + cumsum_boxes[nn_level]
        k_guess = rearrange(nn_indices[1:, :], "d 1 -> d")

        phi_kx = transforms.compute_frames(
            dyadic_decomp, idx_guess, k_guess, KXY, redundancy, windowing
        )
        assert jnp.allclose(phi_kx, f_rect, atol=1e-16), (
            f"coefficient/frame mismatch at level {level}"
        )


@pytest.mark.parametrize(
    "N, aspect, frame_redundancy",
    [
        ((64, 32), (2, 1), 1),
        ((64, 32), (2, 1), 2),
        ((32, 64), (1, 2), 1),
        ((32, 64), (1, 2), 2),
    ],
)
def test_anisotropic_single_coeff_eq_frame(N, aspect, frame_redundancy):
    """Explicit frame atoms match the per-axis local-FFT reconstruction."""
    windowing = "rectangular" if frame_redundancy == 1 else "rectangular_mirror"
    decomp = DyadicDecomposition(
        num_levels=1,
        N=N,
        num_boxes_levels=(4,),
        box_aspect_ratio=aspect,
    )
    wpt = transforms.MSWPT(decomp, redundancy=frame_redundancy, windowing=windowing)

    coeff_shape = tuple(wpt.coeff_shapes[0])
    local_k = (coeff_shape[1] - 1, coeff_shape[2] - 1)
    coeff_idx = jnp.ravel_multi_index((0, *local_k), coeff_shape)
    coeffs = jnp.zeros(wpt.total_coeffs).at[coeff_idx].set(1.0)

    reconstructed = wpt.inverse(coeffs, "fourier")
    frame = transforms.compute_frames(
        decomp,
        0,
        jnp.asarray(local_k),
        decomp.fourier_meshgrid,
        frame_redundancy,
        windowing,
    )

    assert jnp.allclose(frame, reconstructed, atol=1e-16)


def test_isotropic_redundancy_two_phase_offsets_are_exactly_trivial():
    """The published rho=2 convention must not perturb packet amplitudes."""
    decomp = DyadicDecomposition(2, (64, 64), (4, 8), (1, 1))
    wpt = transforms.MSWPT(decomp, redundancy=2, windowing="rectangular")
    shapes = utils.compute_coeff_shapes(
        decomp, wpt.redundancy, jnp.arange(decomp.num_levels)
    )
    levels, indices = utils.find_tensor_and_multiindex(
        jnp.arange(wpt.total_coeffs), shapes
    )
    offsets = jnp.r_[0, jnp.cumsum(decomp.num_boxes_ndim)]
    boxes = indices[0] + offsets[levels]

    phases = transforms.compute_frame_phase(decomp, boxes, indices[1:].T, 2)

    assert jnp.array_equal(phases, jnp.ones_like(phases))
