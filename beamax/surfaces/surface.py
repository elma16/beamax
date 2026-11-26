"""Detector hypersurfaces for curved MSGB boundary matching in 1D, 2D and 3D.

A ``DetectorSurface`` is a codimension-one sensor set ``Gamma`` in ``R^d`` described by a
smooth parametrisation ``X : R^m -> R^d`` (``m = d - 1``) sampled on the sensor axes of the
boundary-data grid, together with a *front* orientation: the unit normal ``n`` points to
the side holding the sources (the reconstruction side).  In 1D the surface is a point
(``m = 0``) and ``n = +-1``.

Data coordinates are ``(t, vth_1, ..., vth_m)``; sample ``j`` of sensor axis ``i`` sits at
``origin[i] + j * spacing[i]``.  Sensors are ordered row-major over the sensor axes, the
same layout as the trailing axes of the boundary data array.

"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def _check_samples(n, minimum):
    if (
        isinstance(n, (bool, np.bool_))
        or not isinstance(n, (int, np.integer))
        or n < minimum
    ):
        raise ValueError(f"Expected at least {minimum} samples on the chart axis.")


def _raw_normal(J):
    """Unnormalised normal from tangent columns, before orientation, for d = 1, 2, 3."""
    d = J.shape[0]
    if d == 1:
        return jnp.ones((1,), dtype=J.dtype)
    if d == 2:
        return jnp.stack([-J[1, 0], J[0, 0]])
    return jnp.cross(J[:, 0], J[:, 1])


@dataclass(frozen=True, eq=False)
class DetectorSurface:
    """Sampled detector hypersurface ``vth -> X(vth)`` with a front-pointing normal."""

    X: Callable[[jnp.ndarray], jnp.ndarray]
    ndim: int
    origin: tuple
    spacing: tuple
    shape: tuple
    periodic: tuple
    front_sign: float = 1.0
    name: str = "surface"
    active_bounds: tuple[tuple[float, float] | None, ...] | None = None

    # ------------------------------------------------------------------ validation
    def __post_init__(self):
        d = self.ndim
        if d not in (1, 2, 3):
            raise ValueError(f"ndim must be 1, 2 or 3; got {d}.")
        m = d - 1
        for label, value in (
            ("origin", self.origin),
            ("spacing", self.spacing),
            ("shape", self.shape),
            ("periodic", self.periodic),
        ):
            if not isinstance(value, tuple) or len(value) != m:
                raise ValueError(
                    f"{label} must be a tuple of length ndim - 1 = {m}; got {value!r}."
                )
        if any((not np.isfinite(s)) or s <= 0 for s in self.spacing):
            raise ValueError(
                f"spacing must be positive and finite; got {self.spacing}."
            )
        if not np.all(np.isfinite(self.origin)):
            raise ValueError("Chart origin must be finite.")
        if any(not isinstance(p, (bool, np.bool_)) for p in self.periodic):
            raise ValueError("periodic must contain booleans.")
        if any(
            (not isinstance(n, (int, np.integer))) or isinstance(n, bool) or n < 1
            for n in self.shape
        ):
            raise ValueError(f"shape must contain positive integers; got {self.shape}.")
        if self.front_sign not in (1.0, -1.0, 1, -1):
            raise ValueError(f"front_sign must be +1 or -1; got {self.front_sign}.")
        if self.active_bounds is not None and len(self.active_bounds) != m:
            raise ValueError("active_bounds must have one entry per chart axis.")
        out = jax.eval_shape(
            self.X,
            jax.ShapeDtypeStruct(
                (m,), jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
            ),
        )
        if out.shape != (d,):
            raise ValueError(
                f"X must map shape ({m},) to ({d},); got output shape {out.shape}."
            )

    @property
    def m(self) -> int:
        return self.ndim - 1

    @property
    def num_sensors(self) -> int:
        return int(np.prod(self.shape)) if self.m else 1

    # ------------------------------------------------------------------ sampling
    def param_axes(self):
        return [
            o + s * np.arange(n)
            for o, s, n in zip(self.origin, self.spacing, self.shape)
        ]

    def params(self) -> jnp.ndarray:
        """Parameter grid, shape ``(*shape, m)`` (``(1, 0)`` in 1D)."""
        if self.m == 0:
            return jnp.zeros((1, 0))
        grids = np.meshgrid(*self.param_axes(), indexing="ij")
        return jnp.asarray(np.stack(grids, axis=-1))

    def params_flat(self) -> jnp.ndarray:
        return self.params().reshape(self.num_sensors, self.m)

    def positions(self) -> jnp.ndarray:
        """Sensor positions, shape ``(num_sensors, d)``, row-major over the sensor axes."""
        return self.points(self.params_flat())

    # ------------------------------------------------------------------ batched geometry
    def _batch(self, th):
        th = jnp.asarray(th, dtype=float)
        if th.ndim == 1 and th.shape[0] == self.m:
            th = th[None]
        if th.ndim != 2 or th.shape[1] != self.m:
            raise ValueError(
                f"parameters must have shape (B, {self.m}); got {th.shape}."
            )
        return th

    def points(self, th) -> jnp.ndarray:
        return jax.vmap(self.X)(self._batch(th))

    def jacobian(self, th) -> jnp.ndarray:
        """``J = dX/dvth``, shape ``(B, d, m)``."""
        th = self._batch(th)
        if self.m == 0:
            return jnp.zeros((th.shape[0], self.ndim, 0))
        return jax.vmap(jax.jacfwd(self.X))(th)

    def hessian(self, th) -> jnp.ndarray:
        """``D^2 X``, shape ``(B, d, m, m)``."""
        th = self._batch(th)
        if self.m == 0:
            return jnp.zeros((th.shape[0], self.ndim, 0, 0))
        return jax.vmap(jax.jacfwd(jax.jacfwd(self.X)))(th)

    def normal(self, th) -> jnp.ndarray:
        """Unit normal pointing to the front (source) side, shape ``(B, d)``."""
        J = self.jacobian(th)
        if self.m == 0:
            return jnp.full((J.shape[0], 1), float(self.front_sign))
        n = jax.vmap(_raw_normal)(J)
        return self.front_sign * n / jnp.linalg.norm(n, axis=-1, keepdims=True)

    def first_fundamental_form(self, th) -> jnp.ndarray:
        J = self.jacobian(th)
        return jnp.einsum("bki,bkj->bij", J, J)

    def second_fundamental_form(self, th) -> jnp.ndarray:
        """``II_ab = n . d_a d_b X`` with respect to the front normal."""
        return jnp.einsum("bk,bkij->bij", self.normal(th), self.hessian(th))

    def shape_operator(self, th) -> jnp.ndarray:
        return jnp.linalg.solve(
            self.first_fundamental_form(th), self.second_fundamental_form(th)
        )

    def principal_curvatures(self, th) -> jnp.ndarray:
        """Eigenvalues of the shape operator, ascending; positive where Gamma bends towards the front."""
        if self.m == 0:
            return jnp.zeros((self._batch(th).shape[0], 0))
        I1, II = self.first_fundamental_form(th), self.second_fundamental_form(th)
        L = jnp.linalg.cholesky(I1)
        Linv = jnp.linalg.inv(L)
        S = jnp.einsum("bij,bjk,blk->bil", Linv, II, Linv)
        return jnp.linalg.eigvalsh(0.5 * (S + jnp.swapaxes(S, 1, 2)))

    def christoffel(self, th) -> jnp.ndarray:
        """Christoffel symbols ``Y[b, mu, a, c]`` of the induced metric: ``d_a d_c X = Y^mu_ac d_mu X + II_ac n``."""
        J, D2 = self.jacobian(th), self.hessian(th)
        B, m = J.shape[0], self.m
        I1 = jnp.einsum("bki,bkj->bij", J, J)
        rhs = jnp.einsum("bki,bkac->biac", J, D2).reshape(B, m, m * m)
        return jnp.linalg.solve(I1, rhs).reshape(B, m, m, m)

    def area_element(self, th) -> jnp.ndarray:
        """``sqrt(det I)``; 1 in 1D."""
        if self.m == 0:
            return jnp.ones((self._batch(th).shape[0],))
        return jnp.sqrt(jnp.linalg.det(self.first_fundamental_form(th)))

    # ------------------------------------------------------------------ interop
    def check_data_domain(self, data_domain, rtol=1e-9):
        """Raise unless ``data_domain`` is a ``(t, vth)`` grid matching this surface's sampling."""
        if data_domain.ndim != self.ndim:
            raise ValueError(
                f"data domain has {data_domain.ndim} axes; the surface needs 1 + {self.m}."
            )
        if tuple(data_domain.N[1:]) != tuple(self.shape):
            raise ValueError(
                f"data sensor axes {tuple(data_domain.N[1:])} do not match surface shape {self.shape}."
            )
        if not np.allclose(
            np.asarray(data_domain.dx[1:], float),
            np.asarray(self.spacing, float),
            rtol=rtol,
            atol=0,
        ):
            raise ValueError(
                f"data sensor spacing {data_domain.dx[1:]} does not match surface spacing {self.spacing}."
            )

    def to_sensor(self, domain):
        """Construct a nearest-voxel mask. Coincident mask cells raise ValueError."""
        from beamax.geometry import Sensor

        return Sensor(domain, positions=self.positions())

    def mask_to_chart(self, domain):
        """Return indices for `kwave_traces[..., indices]` in flat chart order.

        The inverse permutation maps chart channels to C-order mask channels.
        """
        sensor = self.to_sensor(domain)
        mask = np.asarray(sensor.binary_mask, dtype=bool)
        # Use exactly the rounding and clamping used to construct Sensor's mask.
        cells = jnp.rint(sensor.positions / jnp.asarray(domain.dx)).astype(int)
        cells = np.asarray(jnp.clip(cells, 0, jnp.asarray(domain.N) - 1))
        linear = np.ravel_multi_index(tuple(cells.T), domain.N)
        return np.searchsorted(np.flatnonzero(mask), linear)

    def prepare_open_data(self, data, *, padding, taper):
        """Taper open chart edges and zero-pad each side in sensor samples.

        Return the extended data and chart. Build the acquisition domain and
        transform on their new shape. Packets centred in padding are excluded.
        """
        data = jnp.asarray(data)
        if data.ndim != self.ndim or data.shape[1:] != self.shape:
            raise ValueError("data must have shape (Nt, *surface.shape).")
        if self.active_bounds is not None:
            raise ValueError("The surface already has padded chart bounds.")
        for widths in (padding, taper):
            if len(widths) != self.m or any(
                isinstance(w, (bool, np.bool_))
                or not isinstance(w, (int, np.integer))
                or w < 0
                for w in widths
            ):
                raise ValueError(
                    "Specify one nonnegative integer width per chart axis."
                )
        bounds = []
        for i, (n, h, origin, periodic, pad, width) in enumerate(
            zip(self.shape, self.spacing, self.origin, self.periodic, padding, taper)
        ):
            if periodic and (width or pad):
                raise ValueError(
                    "Periodic chart axes must have zero padding and taper."
                )
            if width and 2 * width >= n:
                raise ValueError(
                    "Taper width must leave an interior on each chart axis."
                )
            if width:
                distance = jnp.minimum(jnp.arange(n), n - 1 - jnp.arange(n))
                weight = jnp.sin(0.5 * jnp.pi * jnp.minimum(distance / width, 1)) ** 2
                shape = [1] * data.ndim
                shape[i + 1] = n
                data = data * weight.reshape(shape)
            bounds.append((origin, origin + (n - 1) * h) if pad else None)
        data = jnp.pad(data, ((0, 0), *((p, p) for p in padding)))
        extended = replace(
            self,
            origin=tuple(
                o - p * h for o, p, h in zip(self.origin, padding, self.spacing)
            ),
            shape=data.shape[1:],
            active_bounds=tuple(bounds),
        )
        return data, extended

    def active_parameters(self, vth):
        """Clip padded packet centres to the chart and return their validity."""
        inside = jnp.ones(vth.shape[0], dtype=bool)
        for i, bounds in enumerate(self.active_bounds or ()):
            if bounds is not None:
                lo, hi = bounds
                v = vth[:, i]
                tol = (
                    8 * jnp.finfo(v.dtype).eps * max(abs(lo), abs(hi), self.spacing[i])
                )
                inside &= (v >= lo - tol) & (v <= hi + tol)
                vth = vth.at[:, i].set(jnp.clip(v, lo, hi))
        return vth, inside

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def _oriented(
        X, d, origin, spacing, shape, periodic, name, toward=None, front_vector=None
    ):
        """Choose ``front_sign`` so the normal at the middle sample ``th0`` points towards ``toward``
        (a point, or a callable ``th0 -> point``) or along ``front_vector``."""
        m = d - 1
        th0 = jnp.asarray(
            [o + s * (n // 2) for o, s, n in zip(origin, spacing, shape)], dtype=float
        )
        J = jax.jacfwd(X)(th0) if m else jnp.zeros((d, 0))
        raw = _raw_normal(J) if m else jnp.ones((1,))
        if toward is not None:
            point = toward(th0) if callable(toward) else jnp.asarray(toward, float)
            target = point - X(th0)
        else:
            target = jnp.asarray(front_vector, float)
        inner = float(jnp.dot(raw, target))
        if not np.isfinite(inner) or inner == 0:
            raise ValueError(
                "The orientation must have a nonzero normal component at the middle sample."
            )
        sign = 1.0 if inner > 0 else -1.0
        return DetectorSurface(
            X,
            d,
            tuple(map(float, origin)),
            tuple(map(float, spacing)),
            tuple(shape),
            tuple(periodic),
            sign,
            name,
        )

    @classmethod
    def parametric(
        cls,
        X,
        ndim,
        origin,
        spacing,
        shape,
        periodic=None,
        toward=None,
        front_vector=None,
        name="surface",
    ):
        """General surface from a parametrisation ``X``; orient by a front point/callable or a front vector."""
        if (toward is None) == (front_vector is None):
            raise ValueError(
                "give exactly one of toward and front_vector to orient the surface."
            )
        periodic = tuple(periodic) if periodic is not None else (False,) * (ndim - 1)
        return cls._oriented(
            X,
            ndim,
            tuple(origin),
            tuple(spacing),
            tuple(shape),
            periodic,
            name,
            toward,
            front_vector,
        )

    @classmethod
    def point(cls, x0=0.0, front=+1, name="point"):
        """1D sensor at ``x0``; ``front = +1`` means sources at ``x > x0``."""
        x0 = float(x0)
        return cls(
            lambda th: jnp.asarray([x0]) + 0.0 * jnp.sum(th),
            1,
            (),
            (),
            (),
            (),
            float(np.sign(front)),
            name,
        )

    @classmethod
    def plane(
        cls,
        d,
        axis=0,
        value=0.0,
        origin=None,
        spacing=None,
        shape=None,
        front=+1,
        periodic=None,
        name="plane",
    ):
        """Axis-aligned plane ``x_axis = value``; tangential axes in ascending order (beamax planar layout)."""
        if d == 1:
            return cls.point(value, front, name)
        if spacing is None or shape is None or not 0 <= axis < d:
            raise ValueError("Give a valid plane axis, spacing and chart shape.")
        m = d - 1
        tan = [a for a in range(d) if a != axis]
        origin = tuple(origin) if origin is not None else (0.0,) * m
        periodic = tuple(periodic) if periodic is not None else (False,) * m
        value = float(value)

        def X(th):
            x = jnp.zeros((d,), dtype=th.dtype) + value * (jnp.arange(d) == axis)
            return x.at[jnp.asarray(tan)].set(th)

        fv = np.zeros(d)
        fv[axis] = np.sign(front)
        return cls._oriented(
            X, d, origin, tuple(spacing), tuple(shape), periodic, name, front_vector=fv
        )

    @classmethod
    def circle(cls, centre, radius, n, front="inside", name=None):
        _check_samples(n, 3)
        if not np.isfinite(radius) or radius <= 0 or front not in ("inside", "outside"):
            raise ValueError(
                "A circle needs a positive finite radius and front='inside' or 'outside'."
            )
        c = jnp.asarray(centre, float)

        def X(th):
            return c + radius * jnp.stack([jnp.cos(th[0]), jnp.sin(th[0])])

        s = cls._oriented(
            X,
            2,
            (0.0,),
            (2 * np.pi / n,),
            (n,),
            (True,),
            name or f"circle R={radius:g}",
            toward=centre,
        )
        return s if front == "inside" else _flip(s)

    @classmethod
    def arc(cls, centre, radius, theta0, theta1, n, name=None):
        """Open circular arc, sampled at ``n`` angles from ``theta0`` to ``theta1``; front = concave side."""
        _check_samples(n, 2)
        if (
            not np.isfinite(radius)
            or radius <= 0
            or not 0 < theta1 - theta0 < 2 * np.pi
        ):
            raise ValueError(
                "An arc needs a positive radius and an angular extent strictly between 0 and 2π."
            )
        c = jnp.asarray(centre, float)

        def X(th):
            return c + radius * jnp.stack([jnp.cos(th[0]), jnp.sin(th[0])])

        return cls._oriented(
            X,
            2,
            (theta0,),
            ((theta1 - theta0) / (n - 1),),
            (n,),
            (False,),
            name or "arc",
            toward=centre,
        )

    @classmethod
    def ellipse(cls, centre, a, b, n, name=None):
        _check_samples(n, 3)
        if not all(np.isfinite(v) and v > 0 for v in (a, b)):
            raise ValueError("Ellipse semiaxes must be finite and positive.")
        c = jnp.asarray(centre, float)

        def X(th):
            return c + jnp.stack([a * jnp.cos(th[0]), b * jnp.sin(th[0])])

        return cls._oriented(
            X,
            2,
            (0.0,),
            (2 * np.pi / n,),
            (n,),
            (True,),
            name or f"ellipse {a:g}x{b:g}",
            toward=centre,
        )

    @classmethod
    def polar(cls, centre, r_fn, n, name="polar curve"):
        """Star-shaped closed curve ``r = r_fn(theta)`` about ``centre``; front = inside."""
        _check_samples(n, 3)
        c = jnp.asarray(centre, float)

        def X(th):
            return c + r_fn(th[0]) * jnp.stack([jnp.cos(th[0]), jnp.sin(th[0])])

        return cls._oriented(
            X, 2, (0.0,), (2 * np.pi / n,), (n,), (True,), name, toward=centre
        )

    @classmethod
    def sphere_band(cls, centre, radius, theta0, theta1, n_theta, n_phi, name=None):
        """Sphere minus polar caps, ``(theta, phi)`` polar/azimuthal; theta open, phi periodic; front = inside."""
        _check_samples(n_theta, 2)
        _check_samples(n_phi, 3)
        if not (0 < theta0 < theta1 < np.pi and np.isfinite(radius) and radius > 0):
            raise ValueError(
                "A sphere band needs positive radius, at least two polar samples, and 0 < theta0 < theta1 < pi."
            )
        c = jnp.asarray(centre, float)

        def X(th):
            t, p = th[0], th[1]
            return c + radius * jnp.stack(
                [jnp.sin(t) * jnp.cos(p), jnp.sin(t) * jnp.sin(p), jnp.cos(t)]
            )

        return cls._oriented(
            X,
            3,
            (theta0, 0.0),
            ((theta1 - theta0) / (n_theta - 1), 2 * np.pi / n_phi),
            (n_theta, n_phi),
            (False, True),
            name or f"sphere band R={radius:g}",
            toward=centre,
        )

    @classmethod
    def cylinder(cls, centre, radius, z0, z1, n_theta, n_z, name=None):
        """Cylinder about the axis-2 line through ``centre``; ``(theta, z)``; front = inside."""
        _check_samples(n_theta, 3)
        _check_samples(n_z, 2)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Cylinder radius must be finite and positive.")
        c = jnp.asarray(centre, float)

        def X(th):
            return c + jnp.stack(
                [radius * jnp.cos(th[0]), radius * jnp.sin(th[0]), th[1]]
            )

        def axis_point(th0):
            return c + jnp.stack([0.0, 0.0, th0[1]])

        return cls._oriented(
            X,
            3,
            (0.0, z0),
            (2 * np.pi / n_theta, (z1 - z0) / (n_z - 1)),
            (n_theta, n_z),
            (True, False),
            name or f"cylinder R={radius:g}",
            toward=axis_point,
        )

    @classmethod
    def torus(cls, centre, major, minor, n_u, n_v, name=None):
        """Torus ``((R + r cos v) cos u, (R + r cos v) sin u, r sin v)``; both axes periodic; front = inside the tube."""
        _check_samples(n_u, 3)
        _check_samples(n_v, 3)
        if not np.isfinite(major) or not 0 < minor < major:
            raise ValueError("Torus radii must satisfy 0 < minor < major.")
        c = jnp.asarray(centre, float)

        def X(th):
            u, v = th[0], th[1]
            rho = major + minor * jnp.cos(v)
            return c + jnp.stack(
                [rho * jnp.cos(u), rho * jnp.sin(u), minor * jnp.sin(v)]
            )

        def tube_centre(th0):
            return c + major * jnp.stack([jnp.cos(th0[0]), jnp.sin(th0[0]), 0.0])

        return cls._oriented(
            X,
            3,
            (0.0, 0.0),
            (2 * np.pi / n_u, 2 * np.pi / n_v),
            (n_u, n_v),
            (True, True),
            name or f"torus {major:g}/{minor:g}",
            toward=tube_centre,
        )


def _flip(s: DetectorSurface) -> DetectorSurface:
    return DetectorSurface(
        s.X, s.ndim, s.origin, s.spacing, s.shape, s.periodic, -s.front_sign, s.name
    )
