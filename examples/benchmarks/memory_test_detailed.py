"""
Comprehensive profiling script that accounts for ALL time including JAX async ops.
"""

import os
import gc
import sys
import jax
import jax.numpy as jnp
from time import perf_counter

# Enable profiling
os.environ["BEAMAX_PROFILE"] = "1"

from beamax import geometry
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.gb import gb_solvers
from beamax.solvers import MSGBSolver


def maxrss_mb():
    """Get peak RSS memory in MB."""
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru / (1024 * 1024)
    else:
        return ru / 1024


class Timer:
    """Context manager for timing with JAX sync."""

    def __init__(self, name):
        self.name = name
        self.start = None
        self.end = None

    def __enter__(self):
        jax.block_until_ready(None)  # Sync before starting
        self.start = perf_counter()
        print(f"\n{'>' * 60}")
        print(f"TIMING START: {self.name}")
        return self

    def __exit__(self, *args):
        jax.block_until_ready(None)  # Sync before stopping
        self.end = perf_counter()
        elapsed = self.end - self.start
        print(f"TIMING END: {self.name} = {elapsed:.4f}s")
        print(f"{'<' * 60}")


def run_case_comprehensive(d: int, n: int, b: int) -> dict:
    """Run with comprehensive timing."""
    print(f"\n{'=' * 80}")
    print(f"COMPREHENSIVE TIMING: d={d}, N={n}, beams={b}")
    print(f"{'=' * 80}\n")

    jax.config.update("jax_enable_x64", False)

    timing = {}

    # Setup phase
    with Timer("0_total_setup") as t:
        N = (n,) * d
        dx = (1e-3,) * d
        periodic = (True,) * d
        box_aspect_ratio = (1,) * d
        num_levels = 2
        num_boxes_levels = (4, 8)
        cfl = 0.3
        redundancy = 2
        windowing = "rectangular_mirror"

        def c(x):
            return 1.0 + 0.0 * x[..., 0]

        with Timer("0a_domain"):
            domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
            ts = domain.generate_time_domain()
        timing["0a_domain"] = perf_counter()

        with Timer("0b_decomposition"):
            dyadic_decomp = DyadicDecomposition(
                num_levels, N, num_boxes_levels, box_aspect_ratio
            )
        timing["0b_decomposition"] = perf_counter()

        with Timer("0c_wpt_init"):
            wpt = MSWPT(dyadic_decomp, redundancy, windowing)
        timing["0c_wpt_init"] = perf_counter()

        with Timer("0d_sensors"):
            binary_mask = jnp.zeros(N)
            binary_mask = binary_mask.at[0, ...].set(1)
            sensors = geometry.Sensor(domain, binary_mask=binary_mask)
        timing["0d_sensors"] = perf_counter()

        with Timer("0e_initial_condition"):
            p0 = jnp.zeros(N)
            p0 = p0.at[N[0] // 4].set(1.0)
            dpdt = jnp.zeros_like(p0)
        timing["0e_initial_condition"] = perf_counter()

        with Timer("0f_solver_init"):
            solver = gb_solvers.solve_hom_diag
            msgb_solver = MSGBSolver(
                thr=b,
                thr_strat="top_n",
                batch_size=b,
                input_type="spatial",
                ode_solver=solver,
                sum_method="all_real",
            )
        timing["0f_solver_init"] = perf_counter()

    timing["0_total_setup"] = t.end - t.start

    print("\n" + "=" * 80)
    print("MAIN COMPUTATION")
    print("=" * 80)

    # Main forward pass
    with Timer("1_forward_call") as t:
        sensor_data, aux = msgb_solver.forward(p0, domain, sensors, ts, wpt)
    timing["1_forward_call"] = t.end - t.start

    # Device get (may trigger remaining computation)
    with Timer("2_device_get") as t:
        sensor_data_cpu = jax.device_get(sensor_data)
    timing["2_device_get"] = t.end - t.start

    # Cleanup
    with Timer("3_cleanup") as t:
        del p0, dpdt, sensors, wpt, dyadic_decomp, ts, domain, sensor_data, aux
        gc.collect()
        # Replace the try/except block with this:
        if hasattr(jax, "clear_caches") and callable(getattr(jax, "clear_caches")):
            jax.clear_caches()
    timing["3_cleanup"] = t.end - t.start

    # Summary
    peak_mb = maxrss_mb()

    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)

    total_accounted = sum(timing.values())

    for key in sorted(timing.keys()):
        pct = 100 * timing[key] / total_accounted if total_accounted > 0 else 0
        print(f"  {key:30s}: {timing[key]:8.4f}s ({pct:5.1f}%)")

    print(f"  {'─' * 45}")
    print(f"  {'TOTAL ACCOUNTED':30s}: {total_accounted:8.4f}s (100.0%)")
    print(f"\n  Peak RSS memory: {peak_mb:.2f} MB")
    print(f"  Output shape: {sensor_data_cpu.shape}")

    return {
        "d": d,
        "N": n,
        "b": b,
        "timing": timing,
        "total_s": total_accounted,
        "peak_rss_mb": peak_mb,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--b", type=int, default=128)
    parser.add_argument("--warmup", action="store_true", help="Run warmup pass")
    args = parser.parse_args()

    # Setup (done once)
    N = (args.n,) * args.d
    dx = (1e-3,) * args.d
    periodic = (True,) * args.d
    box_aspect_ratio = (1,) * args.d
    num_levels = 2
    num_boxes_levels = (4, 8)
    cfl = 0.3
    redundancy = 2
    windowing = "rectangular_mirror"

    def c(x):
        return 1.0 + 0.0 * x[..., 0]

    print("Setting up problem...")
    domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
    ts = domain.generate_time_domain()
    dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_levels, box_aspect_ratio
    )
    wpt = MSWPT(dyadic_decomp, redundancy, windowing)

    binary_mask = jnp.zeros(N)
    binary_mask = binary_mask.at[0, ...].set(1)
    sensors = geometry.Sensor(domain, binary_mask=binary_mask)

    p0 = jnp.zeros(N)
    p0 = p0.at[N[0] // 4].set(1.0)
    dpdt = jnp.zeros_like(p0)

    solver = gb_solvers.solve_hom_diag
    msgb_solver = MSGBSolver(
        thr=args.b,
        thr_strat="top_n",
        batch_size=args.b,
        input_type="spatial",
        ode_solver=solver,
        sum_method="all_real",
    )

    if args.warmup:
        print("\n" + "=" * 80)
        print("WARMUP RUN (short time series to compile)")
        print("=" * 80)

        ts_short = ts[:10]  # Just 10 timesteps
        t0 = perf_counter()
        _ = msgb_solver.forward(p0, domain, sensors, ts_short, wpt)
        jax.block_until_ready(_)
        t1 = perf_counter()
        print(f"Warmup completed in {t1 - t0:.2f}s")

        del _
        gc.collect()

    print("\n" + "=" * 80)
    print(f"MAIN RUN: d={args.d}, N={args.n}, beams={args.b}")
    print("=" * 80)

    # Disable profiling for cleaner timing
    os.environ["BEAMAX_PROFILE"] = "0"

    t0 = perf_counter()
    sensor_data, aux = msgb_solver.forward(p0, domain, sensors, ts, wpt)
    sensor_data = jax.block_until_ready(sensor_data)
    t1 = perf_counter()

    elapsed = t1 - t0
    peak_mb = maxrss_mb()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"  Total time: {elapsed:.4f}s")
    print(f"  Peak memory: {peak_mb:.2f} MB")
    print(f"  Output shape: {sensor_data.shape}")

    # Breakdown estimate
    Nt = len(ts)
    Ns = jnp.sum(binary_mask)
    n_beams = aux[0].shape[0]  # p0s shape

    print("\n  Problem size:")
    print(f"    Beams: {n_beams}")
    print(f"    Timesteps: {Nt}")
    print(f"    Sensors: {int(Ns)}")
    print(f"    Total evaluations: {n_beams * Nt * int(Ns):,}")
    print(f"    Evaluations/sec: {(n_beams * Nt * int(Ns)) / elapsed:,.0f}")

    # Memory analysis
    beam_memory_mb = (n_beams * Nt * int(Ns) * 4) / (1024**2)  # float32
    print(f"\n  Memory for full beam array: {beam_memory_mb:.2f} MB")

    if beam_memory_mb > 1000:
        print(f"  WARNING: Full vectorization would use {beam_memory_mb:.0f} MB")
        print("  Consider scan method or reduce problem size")

    print("\n" + "=" * 80)
