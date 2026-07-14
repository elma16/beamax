# Pyright otherwise flags the MSWPT eqx.Module init-set fields that follow a
# field with a default; equinox supports this pattern at runtime via its custom
# __init__, but pyright applies plain dataclass ordering rules.
# pyright: reportGeneralTypeIssues=false
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax.lax import fori_loop
from jaxtyping import Array, Float, Int, Num
from typing import List, Tuple, Union

from beamax import utils
from beamax.decomposition import DyadicDecomposition


# Per-axis grid lengths can arrive as either a (d,) array or a plain tuple of
# ints. The function ``jnp.array(...)``s its argument anyway, so we accept both.
DomainLength = Union[Int[Array, " d"], Tuple[int, ...]]


def _validate_transform_configuration(redundancy: int, windowing: str) -> None:
    """Reject transform configurations that cannot define a finite dual."""
    if redundancy not in (1, 2):
        raise ValueError(f"redundancy must be 1 or 2; got {redundancy}.")
    if windowing not in ("none", "rectangular", "rectangular_mirror"):
        raise ValueError(f"Unknown windowing mode {windowing!r}.")
    if redundancy == 1 and windowing == "rectangular_mirror":
        raise ValueError(
            "redundancy=1 with windowing='rectangular_mirror' leaves "
            "uncovered Fourier bins, so the canonical-dual analysis is "
            "undefined; use windowing='rectangular' or redundancy=2."
        )


def compute_windowed_gaussian(
    centre: Int[Array, " d"],
    meshgrid: Int[Array, "*N d"],
    # Plain int when called directly; scalar Array when indexed from a vmap'd
    # box_lengths buffer.
    box_length,
    box_aspect_ratio: Union[Int[Array, " d"], Tuple[int, ...]],
    domain_length: DomainLength,
    redundancy: int,
    windowing: str,
) -> Float[Array, "*N"]:
    """
    Windowed N-D Gaussian in Fourier index space.

    Parameters
    ----------
    centre : jnp.ndarray, shape (d,)
        Tile centre in Fourier index units.
    meshgrid : jnp.ndarray, shape (*N, d)
        Integer-centred Fourier grid from `DyadicDecomposition.fourier_meshgrid`.
    box_length : int
        Smallest-axis tile length for this level.
    box_aspect_ratio : jnp.ndarray, shape (d,)
        Per-axis aspect multipliers. Values ≥ 1 with at least one 1.
    domain_length : array-like, shape (d,)
        Per-axis grid lengths. These define the periodic wrap independently on
        each Fourier axis.
    redundancy : int
        Supported translation-lattice redundancy (1 or 2). Redundancy 2 does
        not by itself make the Gaussian-window frame tight.
    windowing : {"none", "rectangular", "rectangular_mirror"}
        Windowing function applied to the Gaussian.

    Returns
    -------
    jnp.ndarray, shape (*N,)
        Windowed Gaussian weights. dtype = float32/float64.

    Notes
    -----
    Pure JAX, JIT-safe. Periodisation handled by modulo arithmetic.
    """
    box_aspect_ratio = jnp.array(box_aspect_ratio)
    domain_length = jnp.array(domain_length)

    sigma = (box_length // 2) * box_aspect_ratio
    diff = (meshgrid - centre + domain_length / 2) % domain_length - domain_length / 2
    gaussian_ndim = jnp.exp(-jnp.sum(jnp.square(diff / sigma), axis=-1))

    if windowing == "none":
        return gaussian_ndim

    window_size = box_length * box_aspect_ratio * (redundancy / 2)

    if windowing == "rectangular":
        window = jnp.all((-window_size <= diff) & (diff < window_size), axis=-1)
    elif windowing == "rectangular_mirror":
        window = jnp.all(jnp.abs(diff) < window_size, axis=-1)
    else:
        raise ValueError(f"Invalid windowing function: {windowing}")
    return gaussian_ndim * window.astype(float)


def single_filter_idx(
    # Accepts a plain int or an int array (the vmap'd call passes an array).
    centre_idx,
    meshgrid: Int[Array, "*N d"],
    dyadic_decomp: DyadicDecomposition,
    redundancy: int,
    windowing: str = "rectangular",
) -> Float[Array, "*N"]:
    """
    Filter for a single tile (by global box index).

    Parameters
    ----------
    centre_idx : int
        Global box index in `dyadic_decomp.centres_ndim`.
    meshgrid : jnp.ndarray, shape (*N, d)
        Fourier meshgrid.
    dyadic_decomp : DyadicDecomposition
        Dyadic parameters providing centres and per-level box lengths.
    redundancy : int
        Supported translation-lattice redundancy (1 or 2).
    windowing : str
        Windowing function (see `compute_windowed_gaussian`).

    Returns
    -------
    jnp.ndarray, shape (*N,)
        Filter values.
    """
    level = utils.find_level(dyadic_decomp, centre_idx)

    return compute_windowed_gaussian(
        dyadic_decomp.centres_ndim[centre_idx],
        meshgrid,
        dyadic_decomp.box_lengths[level],
        dyadic_decomp.box_aspect_ratio,
        dyadic_decomp.N,
        redundancy,
        windowing,
    )


vmap_filter_idx = vmap(single_filter_idx, in_axes=(0, None, None, None, None))


def single_filter_coord(
    centre: Int[Array, " d"],
    level: int,
    meshgrid: Int[Array, "*N d"],
    dyadic_decomp: DyadicDecomposition,
    redundancy: int,
    windowing: str = "rectangular",
) -> Float[Array, "*N"]:
    """
    Filter for a single tile (by `(centre, level)` pair).

    Parameters
    ----------
    centre : jnp.ndarray, shape (d,)
        Tile centre in Fourier indices.
    level : int
        Dyadic level (0..L-1).
    meshgrid : jnp.ndarray, shape (*N, d)
        Fourier meshgrid.
    dyadic_decomp : DyadicDecomposition
        Decomposition parameters.
    redundancy : int
        1 or 2.
    windowing : str
        Windowing function.

    Returns
    -------
    jnp.ndarray, shape (*N,)
        Filter values.
    """
    box_length = dyadic_decomp.box_lengths[level]
    box_aspect_ratio = dyadic_decomp.box_aspect_ratio
    domain_length = dyadic_decomp.N

    return compute_windowed_gaussian(
        centre,
        meshgrid,
        box_length,
        box_aspect_ratio,
        domain_length,
        redundancy,
        windowing,
    )


vmap_filter_coord = vmap(single_filter_coord, in_axes=(0, 0, None, None, None, None))


def compute_frame_phase(
    dyadic_decomp: DyadicDecomposition,
    boxidx: Union[int, Int[Array, "..."]],
    k: Int[Array, "... d"],
    redundancy: int,
) -> Num[Array, "..."]:
    """Unit-modulus phase induced by local-patch parity recentering.

    The fast transform parity-rolls each local Fourier patch before applying
    its FFT.  This factor converts the clean global modulation
    ``exp(-2 pi i m.k / S)`` into the exact coefficient convention used by the
    implemented transform, including rectangular supports and both supported
    redundancies.
    """
    level = utils.find_level(dyadic_decomp, boxidx)
    box_length = dyadic_decomp.box_lengths[level]
    if box_length.ndim > 0:
        box_length = box_length[..., None]
    support_lengths = (
        box_length * redundancy * jnp.asarray(dyadic_decomp.box_aspect_ratio)
    )
    half_support = support_lengths // 2
    centre = dyadic_decomp.centres_ndim[boxidx]
    grid_shape = jnp.asarray(dyadic_decomp.N)
    parity_index = (centre + grid_shape // 2) // half_support
    parity_sign = 1 - 2 * (parity_index & 1)
    rolls = parity_sign * (half_support // 2)
    phase_cycles = jnp.sum((centre - rolls) * k / support_lengths, axis=-1)
    # Reduce before evaluating the exponential.  Besides improving accuracy for
    # large indices, this makes mathematically integral offsets (including the
    # reported rho=2 isotropic configurations) evaluate as exactly zero cycles
    # instead of feeding a large multiple of an approximate float32 pi to exp.
    phase_cycles = jnp.remainder(phase_cycles, 1.0)
    return jnp.exp(2 * jnp.pi * 1j * phase_cycles)


def compute_sum_gsquare(
    dyadic_decomp: DyadicDecomposition,
    redundancy: int,
    windowing: str = "rectangular",
) -> Float[Array, "*N"]:
    """
    Sum of squares of all tile filters.

    Parameters
    ----------
    dyadic_decomp : DyadicDecomposition
        Frequency tiling.
    redundancy : int
        1 (basis) or 2 (frame).
    windowing : str
        Windowing function.

    Returns
    -------
    jnp.ndarray, shape (*N,)
        Σ_b g_b^2 over all boxes.

    Notes
    -----
    Implemented with `lax.fori_loop` to avoid large vmaps.
    """
    _validate_transform_configuration(redundancy, windowing)
    filters = vmap_filter_idx(
        jnp.arange(dyadic_decomp.total_num_boxes),
        dyadic_decomp.fourier_meshgrid,
        dyadic_decomp,
        redundancy,
        windowing,
    )
    return jnp.sum(jnp.square(filters), axis=0)


def compute_gh_filters(
    dyadic_decomp: DyadicDecomposition,
    redundancy: int,
    windowing: str = "rectangular",
) -> Tuple[Float[Array, "B *N"], Float[Array, "B *N"]]:
    """
    Compute `g` tiles and their dual `h = g / Σ g^2`.

    Parameters
    ----------
    dyadic_decomp : DyadicDecomposition
        Frequency tiling.
    redundancy : int
        1 or 2.
    windowing : str
        Windowing function.

    Returns
    -------
    (jnp.ndarray, jnp.ndarray)
        `(gfilt, hfilt)` each with shape (num_boxes, *N).
    """
    _validate_transform_configuration(redundancy, windowing)
    num_boxes_ndim = dyadic_decomp.num_boxes_ndim
    gfilt = vmap_filter_idx(
        jnp.arange(jnp.sum(num_boxes_ndim)),
        dyadic_decomp.fourier_meshgrid,
        dyadic_decomp,
        redundancy,
        windowing,
    )
    sum_gsquare = jnp.sum(gfilt**2, axis=0)
    if not bool(jnp.all(jnp.isfinite(sum_gsquare))) or not bool(
        jnp.all(sum_gsquare > 0)
    ):
        raise ValueError(
            "The dyadic decomposition leaves uncovered Fourier bins; "
            "the dual filters are undefined."
        )
    hfilt = gfilt / sum_gsquare
    return gfilt, hfilt


def compute_frames(
    dyadic_decomp: DyadicDecomposition,
    boxidx: int,
    k: Int[Array, " d"],
    fourier_space: Num[Array, "*N d"],
    redundancy: int,
    windowing: str = "rectangular",
) -> Num[Array, "*N"]:
    """
    Frame atom for box `boxidx` with plane-wave modulation.

    Parameters
    ----------
    dyadic_decomp : DyadicDecomposition
        Frequency tiling.
    boxidx : int
        Box index.
    k : jnp.ndarray, shape (d,)
        Local spatial-translation multi-index.
    fourier_space : jnp.ndarray, shape (*N, d)
        Integer Fourier-index coordinates.
    redundancy : int
        1 or 2.
    windowing : str
        Windowing function.

    Returns
    -------
    jnp.ndarray, shape (*N,)
        Complex atom, dtype = complex64/complex128.
    """
    level = utils.find_level(dyadic_decomp, boxidx)
    box_length = dyadic_decomp.box_lengths[level]
    support_lengths = (
        box_length * redundancy * jnp.asarray(dyadic_decomp.box_aspect_ratio)
    )

    g = single_filter_idx(
        boxidx, dyadic_decomp.fourier_meshgrid, dyadic_decomp, redundancy, windowing
    )

    scaled_phase = jnp.einsum("...d,d->...", fourier_space, k / support_lengths)
    normalisation = jnp.prod(support_lengths) ** (-0.5)
    phase_offset = compute_frame_phase(dyadic_decomp, boxidx, k, redundancy)

    return normalisation * phase_offset * jnp.exp(-2 * jnp.pi * 1j * scaled_phase) * g


class MSWPT(eqx.Module):
    """
    Multiscale Wave-Packet Transform.

    Parameters
    ----------
    dyadic_decomp : DyadicDecomposition
        Frequency tiling (centres, per-level box lengths).
    redundancy : int
        Supported translation-lattice redundancy (1 or 2). Static under JIT.
    windowing : {"rectangular", "rectangular_mirror", "none"}
        Windowing for tile filters. ``"none"`` is available for explicit
        synthesis/beam-comparison atoms only; the fast packed analysis rejects
        it because an untruncated Gaussian is not supported on the packed
        local patch. Static under JIT.

    Attributes
    ----------
    dyadic_decomp : DyadicDecomposition
    redundancy : int
    windowing : str
    complex_dtype : jnp.dtype
        complex64 unless JAX x64 enabled → complex128.
    sum_gsquare : jnp.ndarray, shape (*N,)
        Σ_b g_b^2 precomputed.
    boxes_cumsum : Tuple[int, ...]
        Cumulative number of boxes per level (static).
    coeff_shapes : Tuple[Tuple[int, ...], ...]
        Shape per level: (n_boxes_level, *support_shape).
    coeffs_cumsum : Tuple[int, ...]
        Flat coefficient offsets per level (static).
    total_coeffs : int
        Total number of flat coefficients (static).
    gfilts_packed : List[jnp.ndarray]
        Per-level tile `g` packed to minimal support boxes.
    _support_shapes : List[Tuple[int, ...]]
        Per-level support shapes (static).
    _box_shapes : List[Tuple[int, ...]]
        Per-level “box lengths” in each axis (static).
    _half_mask : jnp.ndarray
        Mask selecting one spatial-frequency representative from each
        conjugate pair.

    Notes
    -----
    - Constructor performs shape bookkeeping to keep runtime kernels static-JIT friendly.
    - All heavy transforms are pure JAX functions; no side effects.
    """

    dyadic_decomp: DyadicDecomposition = eqx.field()
    redundancy: int = eqx.field(default=2, static=True)
    windowing: str = eqx.field(default="rectangular", static=True)

    complex_dtype: jnp.dtype = eqx.field(static=True)
    sum_gsquare: jnp.ndarray

    boxes_cumsum: Tuple[int, ...] = eqx.field(static=True)
    coeff_shapes: Tuple[Tuple[int, ...], ...] = eqx.field(static=True)
    coeffs_cumsum: Tuple[int, ...] = eqx.field(static=True)
    max_box_length: int = eqx.field(static=True)
    total_coeffs: int = eqx.field(static=True)
    gfilts_packed: List
    _support_shapes: List[Tuple[int, ...]] = eqx.field(static=True)
    _box_shapes: List[Tuple[int, ...]] = eqx.field(static=True)
    _half_mask: jnp.ndarray

    def __init__(self, dyadic_decomp, redundancy, windowing):
        """
        Build a transform instance with precomputed static metadata.

        Parameters
        ----------
        dyadic_decomp : DyadicDecomposition
            Provides centres, per-level box lengths, and meshgrid.
        redundancy : int
            1 or 2. Controls per-level support sizes and total coeff count.
        windowing : str
            Window type for Gaussian tiles.

        Notes
        -----
        - Precomputes cumulative box/coeff offsets and per-level support shapes.
        - Chooses complex dtype from global JAX precision flag.
        - Packs representative per-level `g` filters for later slicing/rolls.
        """
        _validate_transform_configuration(redundancy, windowing)
        self.dyadic_decomp = dyadic_decomp
        self.redundancy = redundancy
        self.windowing = windowing
        sum_gsquare = self._compute_sum_gsquare()
        if not bool(jnp.all(jnp.isfinite(sum_gsquare))) or not bool(
            jnp.all(sum_gsquare > 0)
        ):
            raise ValueError(
                "The dyadic decomposition leaves uncovered Fourier bins; "
                "choose compatible grid, box counts, and aspect ratios."
            )
        self.sum_gsquare = sum_gsquare
        # `jax.config.x64_enabled` is a dynamically-attached attribute that
        # pyright cannot see; access it via getattr to keep the type-checker happy.
        x64_enabled = bool(getattr(jax.config, "x64_enabled", False))
        self.complex_dtype = jnp.complex128 if x64_enabled else jnp.complex64
        boxes_cumsum_arr = jnp.concatenate(
            [jnp.array([0]), self.dyadic_decomp.num_boxes_ndim_cumsum]
        )
        self.boxes_cumsum = tuple(boxes_cumsum_arr.astype(int).tolist())

        coeff_shapes_arr = utils.compute_coeff_shapes(
            self.dyadic_decomp,
            self.redundancy,
            jnp.arange(self.dyadic_decomp.num_levels),
        )
        # Convert JAX array to tuple of tuples (static structure)
        self.coeff_shapes = tuple(
            tuple(int(x) for x in row) for row in coeff_shapes_arr
        )

        coeffs_cumsum_list = [0]
        for shape in self.coeff_shapes:
            # Calculate product of shape dimensions
            prod = 1
            for dim in shape:
                prod *= dim
            coeffs_cumsum_list.append(coeffs_cumsum_list[-1] + prod)
        self.coeffs_cumsum = tuple(coeffs_cumsum_list)
        self.total_coeffs = int(self.coeffs_cumsum[-1])

        max_level = dyadic_decomp.num_levels - 1
        max_box_length = (
            dyadic_decomp.box_lengths[max_level]
            * max(dyadic_decomp.box_aspect_ratio)
            * (redundancy / 2)
        )
        self.max_box_length = int(max_box_length)
        self._support_shapes = []
        self._box_shapes = []
        for lvl in range(dyadic_decomp.num_levels):
            bl = (
                dyadic_decomp.box_lengths[lvl]
                * jnp.array(dyadic_decomp.box_aspect_ratio)
                * redundancy
                / 2
            ).astype(int)
            self._box_shapes.append(tuple(bl.tolist()))
            self._support_shapes.append(tuple((2 * bl).tolist()))
        seg = [
            self.coeffs_cumsum[i + 1] - self.coeffs_cumsum[i]
            for i in range(dyadic_decomp.num_levels)
        ]
        self._half_mask = jnp.concatenate(
            [jnp.concatenate([jnp.ones(L // 2), jnp.zeros(L - L // 2)]) for L in seg]
        )
        self.gfilts_packed = self._compute_all_g_packed()

    @eqx.filter_jit
    def _compute_sum_gsquare(self):
        """
        Compute Σ_b g_b^2 over all boxes.

        Returns
        -------
        jnp.ndarray, shape (*N,), dtype=float32/float64
            Sum of squared tile filters.

        Notes
        -----
        Uses `lax.fori_loop` to reduce memory and compile cost versus a single large `vmap`.
        Marked `@eqx.filter_jit(donate="all")`.
        """

        def body_fun(i, current_sum):
            """
            Add one squared tile filter to the running sum.

            Parameters
            ----------
            i : int
                Global box index.
            current_sum : jnp.ndarray, shape (*N,)
                Running sum of squared filters.

            Returns
            -------
            jnp.ndarray, shape (*N,)
                Updated running sum.
            """
            filter_i = single_filter_idx(
                i,
                self.dyadic_decomp.fourier_meshgrid,
                self.dyadic_decomp,
                self.redundancy,
                self.windowing,
            )
            return current_sum + jnp.square(filter_i)

        init_sum = jnp.zeros(self.dyadic_decomp.N)

        sum_gsq = fori_loop(0, self.dyadic_decomp.total_num_boxes, body_fun, init_sum)
        return sum_gsq

    def _compute_all_g_packed(self):
        """
        Precompute per-level packed `g` filters on minimal supports.

        Returns
        -------
        List[jnp.ndarray]
            For each level `ℓ`, an array of shape `support_shape[ℓ]` containing
            the centered, cropped `g` tile for the *first* box at that level.

        Notes
        -----
        - The packed tile is extracted with wrap-around at the level’s first centre.
        - Used to avoid recomputing or allocating full-size filters inside loops.
        """
        packs = []
        N = jnp.array(self.dyadic_decomp.N)
        centres = self.dyadic_decomp.centres_ndim + N // 2
        for lvl in range(self.dyadic_decomp.num_levels):
            start = self.boxes_cumsum[lvl]
            g = single_filter_idx(
                start,
                self.dyadic_decomp.fourier_meshgrid,
                self.dyadic_decomp,
                self.redundancy,
                self.windowing,
            )
            support = self._support_shapes[lvl]
            c = centres[start]
            packed = utils.extract_centered_box(g, support, c)
            packs.append(packed)

        return packs

    def _compute_coeffs(
        self, ft_sum_sq: Num[Array, "*N"]
    ) -> Num[Array, " total_coeffs"]:
        """
        Compute flat MSWPT coefficients level-by-level from `ft_sum_sq`.

        Parameters
        ----------
        ft_sum_sq : jnp.ndarray, shape (*N,), complex
            Fourier-domain data divided by Σ g^2 (pre-whitened).

        Returns
        -------
        jnp.ndarray, shape (total_coeffs,), complex
            Concatenated coefficients across all levels.

        Algorithm
        ---------
        For each level:
        1) Extract the Fourier patch around each centre with wrap-around.
        2) Multiply by packed `g`.
        3) Apply parity-preserving rolls to unwrap support.
        4) IFFT (unitary) to get coefficients for that box.
        5) Flatten and place into the global flat buffer.

        Notes
        -----
        - Uses `lax.fori_loop` within each level to keep memory bounded.
        - Axis rolls depend on centre and per-level box lengths (integer, static).
        """
        N = jnp.array(self.dyadic_decomp.N)
        num_levels = self.dyadic_decomp.num_levels
        d = self.dyadic_decomp.ndim
        centres_ndim = self.dyadic_decomp.centres_ndim
        axis = tuple(range(d))
        centres = centres_ndim + N // 2

        all_coeffs = jnp.zeros((self.total_coeffs,), dtype=self.complex_dtype)

        for level in range(num_levels):
            start_idx = self.boxes_cumsum[level]
            end_idx = self.boxes_cumsum[level + 1]
            coeff_idx_prev = self.coeffs_cumsum[level]

            centres_level = centres[start_idx:end_idx]
            gfilt_level_packed = self.gfilts_packed[level]

            box_length_level = self._box_shapes[level]
            support_shape_level = self._support_shapes[level]

            # This function processes a single box for the forward transform
            def loop_body(i, coeffs_for_level):
                """
                Compute and store coefficients for one box at the current level.

                Parameters
                ----------
                i : int
                    Local box index within the level.
                coeffs_for_level : jnp.ndarray
                    Running coefficient tensor for the level.

                Returns
                -------
                jnp.ndarray
                    Updated coefficient tensor for the level.
                """
                centre = centres_level[i]

                fft_patch = utils.extract_centered_box(
                    ft_sum_sq, support_shape_level, centre
                )
                support_filtered = gfilt_level_packed * fft_patch

                box_half_support = jnp.array(box_length_level)
                rolls_intermediate = (centre + N // 2) // box_half_support
                parity_sign = 1 - 2 * (rolls_intermediate & 1)
                # Parenthesise the half-length before applying the sign. For
                # an odd half-support, ``(-1 * length) // 2`` floors to -1,
                # whereas the inverse correctly uses ``-(length // 2) == 0``.
                # The old ordering broke rho=1 anisotropic round trips.
                rolls = parity_sign * (box_half_support // 2)
                support_filtered = jnp.roll(support_filtered, rolls, axis=axis)

                # Compute IFFT and update the coefficient array for this level
                coeff = utils.unitary_ifft(support_filtered)
                return coeffs_for_level.at[i].set(coeff)

            # Initialize an empty array for this level's coefficients
            initial_coeffs_level = jnp.zeros(
                self.coeff_shapes[level], dtype=self.complex_dtype
            )

            # Loop over all boxes in this level
            final_coeffs_level = lax.fori_loop(
                0, end_idx - start_idx, loop_body, initial_coeffs_level
            )

            # Update the full coefficient vector
            all_coeffs = lax.dynamic_update_slice(
                all_coeffs, jnp.ravel(final_coeffs_level), (coeff_idx_prev,)
            )

        return all_coeffs

    @eqx.filter_jit(donate="all-except-first")
    def forward(
        self, data: Num[Array, "*N"], input_type: str
    ) -> Num[Array, " total_coeffs"]:
        """
        Forward MSWPT.

        Parameters
        ----------
        data : jnp.ndarray, shape (*N,), real or complex
            Input field in spatial or Fourier domain.
        input_type : {"spatial", "fourier"}
            Declares the domain of `data`.

        Returns
        -------
        jnp.ndarray, shape (total_coeffs,), complex
            Flat coefficient vector.

        Notes
        -----
        - Converts to Fourier (`utils.unitary_fft`) if needed.
        - Divides by Σ g^2 to apply the pointwise canonical-dual analysis
          filters for this painless frame construction.
        - JIT-compiled while preserving the reusable transform state. The
          input data buffer may still be donated for memory efficiency.
        """
        if self.windowing == "none":
            raise ValueError(
                "MSWPT.forward does not support windowing='none': the fast "
                "local-patch implementation truncates an otherwise global "
                "Gaussian, so it cannot provide an exact analysis/synthesis "
                "pair. Use 'rectangular' or 'rectangular_mirror'."
            )
        ft_data = utils.convert_space(data, input_type, "fourier")
        ft_sum_gsq = ft_data / self.sum_gsquare
        coeffs = self._compute_coeffs(ft_sum_gsq)
        return coeffs

    @eqx.filter_jit(donate="all-except-first")
    def inverse(
        self, coeffs: Num[Array, " total_coeffs"], output_type: str
    ) -> Num[Array, "*N"]:
        """
        Fast synthesis MSWPT; the exact inverse of :meth:`forward` for its
        supported window configurations.

        Parameters
        ----------
        coeffs : jnp.ndarray, shape (total_coeffs,), complex
            Flat coefficient vector produced by :meth:`forward`.
        output_type : {"spatial", "fourier"}
            Domain of the returned array.

        Returns
        -------
        jnp.ndarray, shape (*N,), complex
            Reconstructed field in the requested domain.

        Notes
        -----
        With ``windowing="none"`` this remains a synthesis-only map for the
        Gaussian restricted to the transform's nominal packed support (the
        same packed atom as ``windowing="rectangular"``). It is *not* the
        global unwindowed atom returned by :func:`compute_frames`; there is
        deliberately no corresponding fast analysis call.

        The synthesis mirrors the analysis steps in :meth:`forward`::

            forward:   patch = extract(F / Σg², centre);   c = IFFT(roll(g*patch, +r))
            inverse:   tmp = FFT(c);                       add += g * roll(tmp, -r)

        where the periodic scatter-add happens at the true box centre.
        """
        N = jnp.array(self.dyadic_decomp.N, dtype=jnp.int32)
        L = self.dyadic_decomp.num_levels
        ft_out = jnp.zeros(self.dyadic_decomp.N, dtype=self.complex_dtype)

        centres_all = self.dyadic_decomp.centres_ndim.astype(jnp.int32)

        for level in range(L):
            # ----- static per-level data -----
            start = int(self.boxes_cumsum[level])
            end = int(self.boxes_cumsum[level + 1])
            nbox = end - start

            c_lo = int(self.coeffs_cumsum[level])
            c_hi = int(self.coeffs_cumsum[level + 1])

            gfilt = self.gfilts_packed[level]  # (*S,)
            S_tuple = tuple(int(s) for s in gfilt.shape)
            d = len(S_tuple)

            box_len = jnp.array(self._box_shapes[level], dtype=jnp.int32)  # (d,)

            # Precompute support indices/meshgrid once per level (static)
            aranges = tuple(jnp.arange(Sk, dtype=jnp.int32) for Sk in S_tuple)
            base_grids = jnp.meshgrid(*aranges, indexing="ij")
            S = jnp.array(S_tuple, dtype=jnp.int32)
            S_half = S // 2

            # Slice coeffs and reshape to (nbox, *S)
            coeffs_lvl = lax.dynamic_slice(coeffs, (c_lo,), (c_hi - c_lo,))
            coeffs_lvl = coeffs_lvl.reshape((nbox, *S_tuple))

            centres_lvl = centres_all[start:end]  # (nbox, d)

            def body(i, ft):
                """
                Scatter-add one inverse-transform box contribution.

                Parameters
                ----------
                i : int
                    Local box index within the level.
                ft : jnp.ndarray, shape (*N,)
                    Running Fourier-domain reconstruction.

                Returns
                -------
                jnp.ndarray, shape (*N,)
                    Updated Fourier-domain reconstruction.
                """
                centre = centres_lvl[i]  # (d,)

                # ----- compute parity rolls r (same as forward) -----
                ri = (centre + (N // 2)) // box_len
                sign = 1 - 2 * (ri & 1)  # even→+1, odd→-1
                rolls = sign * (box_len // 2)  # vector (d,)

                # ----- local synthesis on support -----
                cpatch = coeffs_lvl[i]  # (*S,)
                fh = utils.unitary_fft(cpatch)  # FFT(c)

                # exact roll(fh, -rolls) using modular indexing on support
                # jnp.roll(x, s) along axis k is x[take((arange - s) % S)]
                unrolled = fh
                for ax in range(d):
                    idx = (aranges[ax] + rolls[ax]) % S[ax]  # -(-rolls) == +rolls
                    unrolled = jnp.take(unrolled, idx, axis=ax)

                contrib = gfilt * unrolled  # g * roll(FFT(c), -r)

                # ----- periodic scatter-add at the true centre -----
                c0 = (centre + (N // 2)) % N  # [0,N)
                starts = (c0 - S_half) % N
                grids = tuple(((starts[k] + base_grids[k]) % N[k]) for k in range(d))

                ft = ft.at[grids].add(contrib)
                return ft

            ft_out = lax.fori_loop(0, nbox, body, ft_out)

        # back to requested domain
        return utils.convert_space(ft_out, "fourier", output_type)

    def convert_to_array(self, coeffs: Num[Array, " total_coeffs"]) -> Num[Array, "*M"]:
        """
        Reshape flat coefficients into a dense tensor arranged by spatial support.

        Parameters
        ----------
        coeffs : jnp.ndarray, shape (total_coeffs,), complex
            Flat vector returned by :meth:`forward`.

        Returns
        -------
        jnp.ndarray
            Dense coefficient tensor with per-level boxes unflattened and placed
            at their centred positions. **Shape:** `(redundancy * N1, redundancy * N2, …)`,
            **dtype:** complex.

        Notes
        -----
        - Intended for diagnostics/visualization; not required for forward/inverse.
        - Uses integer centres and per-level support shapes; pure JAX.
        """
        N = jnp.array(self.dyadic_decomp.N)
        num_levels = self.dyadic_decomp.num_levels
        centres_ndim = self.dyadic_decomp.centres_ndim
        centres = centres_ndim * 2 + N
        coeffs_array = jnp.zeros((self.redundancy * N), dtype=self.complex_dtype)

        for level in range(num_levels):
            start_idx = self.boxes_cumsum[level]
            end_idx = self.boxes_cumsum[level + 1]

            coeff_idx_prev = self.coeffs_cumsum[level]
            coeff_idx_next = self.coeffs_cumsum[level + 1]

            box_length_level = self._box_shapes[level]

            centres_level = centres[start_idx:end_idx]

            support_shape_level = self._support_shapes[level]

            length = jnp.prod(jnp.array(support_shape_level))
            coeffs_level = coeffs[coeff_idx_prev:coeff_idx_next]

            # unflatten coefficients, place it in the correct position
            coeff_offset = 0
            for boxidx in range(self.dyadic_decomp.num_boxes_ndim[level]):
                center = centres_level[boxidx]

                # Calculate box boundaries
                half_length = jnp.array(box_length_level)
                starts = center - half_length
                ends = center + half_length

                # Handle multi-dimensional slicing
                slices = tuple(slice(start, end) for start, end in zip(starts, ends))

                box = coeffs_level[coeff_offset : coeff_offset + length]
                box = jnp.reshape(box, support_shape_level)
                coeffs_array = coeffs_array.at[slices].set(box)
                coeff_offset += length
        return coeffs_array
