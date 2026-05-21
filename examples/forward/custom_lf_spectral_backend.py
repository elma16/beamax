"""
Custom low-frequency backend for HybridSolver.

Example category: Forward propagation
Example smoke: true

This example keeps the high-frequency path in MSGB and supplies a tiny
forward-only low-frequency backend implemented with JAX FFTs. The backend is
deliberately simple: homogeneous, 1D, periodic, zero initial velocity.
"""

import jax
import jax.numpy as jnp

from beamax import Domain, DyadicDecomposition, MSWPT, Sensor
from beamax.gb import gb_solvers
from beamax.solvers import HybridBackend, HybridSolver, MSGBSolver


jax.config.update("jax_enable_x64", True)


def spectral_lf_forward(p0_lf, ctx):
    """
    Solve the 1D homogeneous wave equation on ``ctx.component_domain``.

    This is a teaching adapter, not a production LF solver. It demonstrates
    the only contract a custom forward backend must satisfy:
    ``callable(component_array, context) -> sensor_data``.
    """
    domain = ctx.component_domain
    if len(domain.N) != 1:
        raise ValueError("spectral_lf_forward is a 1D example backend.")

    n = domain.N[0]
    dx = domain.dx[0]
    c0 = float(jnp.max(domain.sound_speed_array))
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)

    p0_hat = jnp.fft.fft(jnp.asarray(p0_lf))
    phase = jnp.cos(ctx.ts[:, None] * c0 * jnp.abs(k)[None, :])
    fields = jnp.fft.ifft(phase * p0_hat[None, :], axis=-1).real

    sensor_mask = jnp.asarray(ctx.component_sensor_mask).astype(bool)
    return fields[:, sensor_mask]


def main():
    n = 64
    domain = Domain(N=(n,), dx=(1.0 / n,), c=1.0, periodic=(True,))
    ts = jnp.linspace(0.0, 0.08, 5)

    x = jnp.arange(n) * domain.dx[0]
    p0 = jnp.exp(-200.0 * (x - 0.35) ** 2) * jnp.cos(18.0 * jnp.pi * x)

    decomp = DyadicDecomposition(
        num_levels=2,
        N=domain.N,
        num_boxes_levels=(4, 8),
        box_aspect_ratio=(1,),
    )
    wpt = MSWPT(decomp, redundancy=2, windowing="rectangular")
    sensors = Sensor(domain=domain, binary_mask=jnp.ones(domain.N))

    msgb = MSGBSolver(
        thr=int(wpt.total_coeffs),
        thr_strat="top_n",
        batch_size=64,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        sum_method="all_real",
    )

    hybrid = HybridSolver(
        hf_solver=msgb,
        lf_backend=HybridBackend(
            forward=spectral_lf_forward,
            name="1D spectral LF example",
        ),
        box_corners=jnp.array([0, 1]),
        downsample=False,
        use_time_extension=False,
        dt_oversample=0,
    )

    sensor_data = hybrid.forward(p0, domain, sensors, ts, wpt)
    print(sensor_data.shape)


if __name__ == "__main__":
    main()
