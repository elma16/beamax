"""Circular-array PAT boundary-source operator and approximate adjoint."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import hankel1

from .matching import surface_time_reversal
from .surface import DetectorSurface


def ring_dirichlet_from_source(F, dt, R, c, symbol="exact", guard=0.05):
    """Apply the advanced circular boundary-source inverse to real (time, angle) traces.

    The exact symbol uses cylindrical Hankel functions in a homogeneous medium.
    Both symbols discard evanescent and zero-frequency modes. The principal
    symbol also excludes normal frequency below `guard * abs(temporal frequency)`.
    FFT and special-function evaluation run on the host.
    """
    F = np.asarray(F)
    if F.ndim != 2 or min(F.shape) < 1 or np.iscomplexobj(F):
        raise ValueError("F must be real and have shape (num_times, num_elements).")
    if not all(np.isfinite(x) and x > 0 for x in (dt, R, c)):
        raise ValueError("dt, R and c must be finite and positive.")
    if not np.isfinite(guard) or not 0 <= guard < 1:
        raise ValueError("guard must be in [0, 1).")
    nt, ns = F.shape
    nfft = 2 * nt
    spectrum = np.fft.fft2(F, s=(nfft, ns))
    w = 2 * np.pi * np.fft.fftfreq(nfft, dt)
    m = np.abs(np.fft.fftfreq(ns, 1 / ns))
    W, M = np.meshgrid(w, m, indexing="ij")
    z = np.abs(W) * R / c
    valid = z > M
    multiplier = np.zeros_like(spectrum)
    if symbol == "exact":
        h2 = np.abs(hankel1(M[valid], z[valid])) ** 2
        multiplier[valid] = -np.sign(W[valid]) * np.pi * R * h2 / (4j * c**2)
    elif symbol == "principal":
        kappa = np.sqrt(np.maximum(W**2 - (c * M / R) ** 2, 0))
        valid &= kappa > guard * np.abs(W)
        multiplier[valid] = 1j * np.sign(W[valid]) / (2 * c * kappa[valid])
    else:
        raise ValueError("symbol must be 'exact' or 'principal'.")
    return np.fft.ifft2(multiplier * spectrum)[:nt].real


def ring_adjoint(
    data,
    data_domain,
    data_wpt,
    domain,
    top_n,
    *,
    centre,
    radius,
    pairing,
    boundary_speed=None,
    symbol="exact",
    guard=0.05,
    eval_points=None,
    **tr_options,
):
    """Approximate continuous 2D PAT adjoint on a uniformly sampled circular array.

    `element_sum` uses dt times a sum over elements. `arc_length` additionally
    weights that sum by 2πR/N. The image pairing uses area. A heterogeneous
    interior uses the circular symbol at the shared boundary sound speed.
    """
    if domain.lam != 0.0:
        raise NotImplementedError(
            "The ring adjoint requires a lossless domain (lam=0)."
        )
    data = np.asarray(data)
    if domain.ndim != 2 or data.ndim != 2 or data.shape != tuple(data_domain.N):
        raise ValueError(
            "The ring adjoint requires a 2D image and a (time, element) acquisition grid."
        )
    if np.iscomplexobj(data) or not np.all(np.isfinite(data)):
        raise ValueError("Ring data must be real and finite.")
    nt, ns = data.shape
    if nt < 2 or ns < 3:
        raise ValueError(
            "The ring adjoint needs at least two times and three elements."
        )
    surface = DetectorSurface.circle(centre, radius, ns)
    surface.check_data_domain(data_domain)
    speeds = np.asarray(jax.vmap(domain.c_fn)(surface.positions()))
    c0 = float(speeds.mean()) if boundary_speed is None else float(boundary_speed)
    if not np.allclose(speeds, c0, rtol=1e-5, atol=0):
        raise ValueError(
            "The circular symbol requires a common sound speed at all ring elements."
        )
    ds = 2 * np.pi * radius / ns
    if pairing not in ("arc_length", "element_sum"):
        raise ValueError("pairing must be 'arc_length' or 'element_sum'.")
    dt = data_domain.dx[0]
    w = 2 * np.pi * np.fft.rfftfreq(2 * nt, dt)
    derivative = np.fft.irfft(
        1j * w[:, None] * np.fft.rfft(data, 2 * nt, axis=0), 2 * nt, axis=0
    )[:nt]
    source = c0**2 * derivative / (ds if pairing == "element_sum" else 1.0)
    trace = ring_dirichlet_from_source(source, dt, radius, c0, symbol, guard)
    image, diag = surface_time_reversal(
        trace,
        data_domain,
        data_wpt,
        surface,
        domain,
        top_n,
        eval_points=eval_points,
        **tr_options,
    )
    points = domain.grid if eval_points is None else jnp.asarray(eval_points)
    c_image = jax.vmap(domain.c_fn)(points.reshape(-1, 2)).reshape(points.shape[:-1])
    return -image / c_image**2, diag
