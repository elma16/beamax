"""
Measure peak memory usage of an MSGB forward solve, annotated for CPU and GPU runs.
"""
# measure_msgb.py
import os
import gc
import resource
import sys
import jax
import jax.numpy as jnp
from time import time
from beamax import geometry
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver

# If you ever run on GPU and want to avoid preallocation, set these BEFORE importing jax:
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"


def _ru_maxrss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # bytes
        return ru / (1024 * 1024)
    else:  # Linux: kilobytes
        return ru / 1024


def run_case(d: int, n: int, b: int) -> dict:
    jax.config.update("jax_enable_x64", False)
    # --- construct inputs ---
    N = (n,) * d
    dx = (1e-3,) * d
    periodic = (True,) * d
    box_aspect_ratio = (1,) * d
    num_levels = 2
    num_boxes_levels = (4, 8)
    cfl = 0.3
    redundancy = 2
    windowing = "rectangular_mirror"

    # coeffs = 0
    sensor_data = 0
    aux = 0

    def c(x):  # homogeneous
        return 1.0 + 0.0 * x[..., 0]

    domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
    ts = domain.generate_time_domain()  # Nt depends on cfl & domain size
    Nt = len(ts)

    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_levels, box_aspect_ratio
    )
    wpt = MSWPT(dyadic_decomp, redundancy, windowing)

    binary_mask = jnp.zeros(N)

    # pos = tuple(jnp.zeros(d, dtype=bool))
    # binary_mask = binary_mask.at[pos].set(1)
    binary_mask = binary_mask.at[0, ...].set(1)
    sensors = geometry.Sensor(domain, binary_mask=binary_mask)

    p0 = jnp.zeros(N)
    p0 = p0.at[N[0] // 4].set(1.0)
    dpdt = jnp.zeros_like(p0)

    # coeffs = wpt.forward(p0, "spatial")

    solver = gb_solvers.solve_hom_diag
    msgb_solver = MSGBSolver(
        thr=b,
        thr_strat="top_n",
        batch_size=b,
        input_type="spatial",
        ode_solver=solver,
        sum_method="all_real",
    )

    t0 = time()
    sensor_data, aux = msgb_solver.forward(p0, domain, sensors, ts, wpt)
    sensor_data = jax.device_get(sensor_data)
    t1 = time()

    # Peak memory for this process:
    peak_mb = _ru_maxrss_mb()

    # Clean up to get accurate ru_maxrss and avoid side effects
    del (
        p0,
        dpdt,
        sensors,
        wpt,
        dyadic_decomp,
        ts,
        domain,
        sensor_data,
        aux,
    )
    gc.collect()
    try:
        jax.clear_caches()  # free JIT caches for this process if any
    except Exception:
        pass

    return {
        "d": d,
        "N": n,
        "b": b,
        "Nt": Nt,
        "elapsed_s": t1 - t0,
        "peak_rss_mb": peak_mb,
    }


if __name__ == "__main__":
    import multiprocessing as mp

    combos = [(d, n, b) for d in (2, 3) for n in (64, 128) for b in (1, 100)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(len(combos), os.cpu_count())) as pool:
        results = pool.starmap(run_case, combos)
    for r in results:
        print(r)
