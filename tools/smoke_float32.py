"""Small release smoke test for Beamax with JAX 64-bit mode disabled."""

import jax
import jax.numpy as jnp

from beamax import Domain, DyadicDecomposition, MSWPT, Sensor
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver


def main() -> None:
    if jax.config.x64_enabled:
        raise RuntimeError("smoke_float32.py must run with JAX_ENABLE_X64=0")

    domain = Domain(
        N=(16,),
        dx=(1.0 / 16,),
        c=jnp.full((16,), 1.0, dtype=jnp.float32),
        periodic=(True,),
    )
    transform = MSWPT(
        DyadicDecomposition(1, domain.N, (4,), (1,)),
        redundancy=2,
        windowing="rectangular",
    )
    sensors = Sensor(domain, binary_mask=jnp.ones(domain.N, dtype=jnp.int32))
    p0 = jnp.cos(2 * jnp.pi * jnp.arange(16, dtype=jnp.float32) / 16)
    ts = jnp.linspace(0.0, 0.01, 3, dtype=jnp.float32)
    solver = MSGBSolver(
        thr=2,
        thr_strat="top_n",
        batch_size=2,
        input_type="spatial",
        ode_solver=gb_solvers.solve_ODE_base,
        sum_method="all_real",
    )

    result = solver.forward(p0, domain, sensors, ts, transform)
    if result.dtype != jnp.float32 or not bool(jnp.all(jnp.isfinite(result))):
        raise RuntimeError(f"invalid float32 result: dtype={result.dtype}")


if __name__ == "__main__":
    main()
