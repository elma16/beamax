#!/usr/bin/env python
# coding: utf-8

"""
Runtime comparison between the k-Wave forward solver and beamax's MSGB solver across a grid-size sweep. Requires `[kwave]` extra.
"""
# # Comparison of k-Wave and GB Solver.
#
# In this example, we compare the k-Wave forward solver with the propagation and summation of a collection of GBs.



# ## Imports

import csv
from pathlib import Path
from time import time
from datetime import datetime
import os

import jax
import jax.numpy as jnp
from beamax import geometry, utils
from beamax.gb import gb_utils, gb_solvers
from beamax.solvers.msgb_solvers import forward_solver_utils
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from beamax.solvers.kwave_solver import TimedKWaveSolver
from beamax.plotter import use_beamax_style

import numpy as np
from collections import defaultdict

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

out_csv = PLOT_DIR / "runtimes.csv"

import matplotlib.pyplot as plt

use_beamax_style()

# ## GB vs. K-Wave


def _write_row(path, header, row):
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def _gb_case(N, dx, periodic, c_val, b, ω0_val, lam, cfl):
    def c(x):
        return c_val + 0 * x[..., 0]

    d = len(N)
    domain = geometry.Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=cfl)
    ts = domain.generate_time_domain()
    binary_mask = jnp.zeros(N)
    binary_mask = binary_mask.at[0, ...].set(1)
    sensors = geometry.Sensor(domain=domain, binary_mask=binary_mask)
    mode = jnp.ones((b,))
    x0 = (domain.grid_size * 0.5).repeat(b).reshape(b, d)
    p0 = jnp.zeros((b, d))
    p0 = p0.at[:, 0].set(1)
    p0 = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)
    a0 = jnp.ones((b,))
    ω0 = jnp.ones((b,)) * ω0_val
    alpha0 = jnp.ones((b, d)) * 1j
    M0 = gb_utils.prepare_M0(alpha0, None)
    solver = gb_solvers.solve_ODE_base

    def fn(p0, M0, x0, ω0, a0, mode, spos, ts):
        params = (p0, M0, x0, ω0, a0, mode)
        return forward_solver_utils.compute_forward_result_all_real(
            params, c, lam, ts, solver, spos, domain.grid_size, jnp.array(periodic)
        )

    jit_fn = jax.jit(fn)
    t1 = time()
    _ = jit_fn(p0, M0, x0, ω0, a0, mode, sensors.positions, ts).block_until_ready()
    t2 = time()
    print(f"Warmup time: {t2 - t1:.3f} seconds")
    t3 = time()
    out = jit_fn(p0, M0, x0, ω0, a0, mode, sensors.positions, ts).block_until_ready()
    t4 = time()
    print(f"Runtime: {t4 - t3:.3f} seconds")
    del out, p0, M0, x0, ω0, a0, mode, alpha0
    return t2 - t1, t4 - t3


def _kw_case(N, dx, periodic, c_val, is_gpu, cfl):
    def c(x):
        return c_val + 0 * x[..., 0]

    # d = len(N)
    domain = geometry.Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=cfl)
    ts = domain.generate_time_domain()
    binary_mask = jnp.zeros(N)
    binary_mask = binary_mask.at[0, ...].set(1)
    sensors = geometry.Sensor(binary_mask=binary_mask, domain=domain)
    p0 = jnp.ones(N)
    sim_opts = SimulationOptions(data_cast="double", smooth_p0=False, save_to_disk=True)
    exec_opts = SimulationExecutionOptions(
        is_gpu_simulation=is_gpu, delete_data=False, verbose_level=0
    )
    solver = TimedKWaveSolver(sim_opts, exec_opts)
    t1 = time()
    out = solver.forward(p0, domain, sensors.binary_mask, ts)
    t2 = time()
    print(f"K-Wave runtime: {t2 - t1:.3f} seconds")
    del out, p0
    return t2 - t1


gpu_available = any(d.platform == "gpu" for d in jax.devices())
platform = ",".join(sorted(set(d.platform for d in jax.devices())))
header = [
    "timestamp",
    "backend",
    "d",
    "n",
    "b",
    "is_gpu",
    "warmup_time",
    "runtime",
    "platform",
    "notes",
]

full_benchmark = os.environ.get("BEAMAX_FULL_BENCHMARKS", "0") == "1"
if full_benchmark:
    sizes_by_dim = {1: range(6, 12), 2: range(6, 12), 3: range(6, 10)}
    b_list = [1, 10, 100, 1000]
else:
    print("Running reduced benchmark. Set BEAMAX_FULL_BENCHMARKS=1 for full sweep.")
    sizes_by_dim = {1: range(6, 7), 2: range(6, 7)}
    b_list = [1, 10]
dx_val = 1e-4
c_val_gb = 1500.0
c_val_kw = 1500.0
ω0_val = 100.0
lam = 0.0
cfl = 0.3

for d, exps in sizes_by_dim.items():
    for i in exps:
        n = 2**i
        N = (n,) * d
        dx = (dx_val,) * d
        periodic = (False,) * d
        for b in b_list:
            try:
                w, r = _gb_case(N, dx, periodic, c_val_gb, b, ω0_val, lam, cfl)
                _write_row(
                    out_csv,
                    header,
                    dict(
                        timestamp=datetime.utcnow().isoformat(),
                        backend="gb",
                        d=d,
                        n=n,
                        b=b,
                        is_gpu="",
                        warmup_time=w,
                        runtime=r,
                        platform=platform,
                        notes="",
                    ),
                )
            except Exception as e:
                _write_row(
                    out_csv,
                    header,
                    dict(
                        timestamp=datetime.utcnow().isoformat(),
                        backend="gb",
                        d=d,
                        n=n,
                        b=b,
                        is_gpu="",
                        warmup_time="",
                        runtime="",
                        platform=platform,
                        notes=f"error:{type(e).__name__}:{e}",
                    ),
                )
                jax.clear_caches()
                break
        try:
            for is_gpu in (False, True):
                if is_gpu and not gpu_available:
                    _write_row(
                        out_csv,
                        header,
                        dict(
                            timestamp=datetime.utcnow().isoformat(),
                            backend="kwave",
                            d=d,
                            n=n,
                            b="",
                            is_gpu=is_gpu,
                            warmup_time="",
                            runtime="",
                            platform=platform,
                            notes="gpu_unavailable",
                        ),
                    )
                    continue
                rt = _kw_case(N, dx, periodic, c_val_kw, is_gpu, cfl)
                _write_row(
                    out_csv,
                    header,
                    dict(
                        timestamp=datetime.utcnow().isoformat(),
                        backend="kwave",
                        d=d,
                        n=n,
                        b="",
                        is_gpu=is_gpu,
                        warmup_time="",
                        runtime=rt,
                        platform=platform,
                        notes="",
                    ),
                )
        except Exception as e:
            _write_row(
                out_csv,
                header,
                dict(
                    timestamp=datetime.utcnow().isoformat(),
                    backend="kwave",
                    d=d,
                    n=n,
                    b="",
                    is_gpu=is_gpu,
                    warmup_time="",
                    runtime="",
                    platform=platform,
                    notes=f"error:{type(e).__name__}:{e}",
                ),
            )
            jax.clear_caches()

print(f"Saved: {out_csv}")

# ## Construct table


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if not r["runtime"] or r["notes"]:
                continue
            rows.append(
                dict(
                    timestamp=r["timestamp"],
                    backend=r["backend"],
                    d=int(r["d"]),
                    n=int(r["n"]),
                    b=None if r["b"] == "" else int(r["b"]),
                    is_gpu=(str(r["is_gpu"]).lower() == "true"),
                    warmup_time=None
                    if r["warmup_time"] == ""
                    else float(r["warmup_time"]),
                    runtime=float(r["runtime"]),
                    platform=r["platform"],
                )
            )
    return rows


def agg_by(rows, key_fields, x_field, y_field="runtime", reducer=np.median):
    dct = defaultdict(lambda: defaultdict(list))
    for r in rows:
        dct[tuple(r[k] for k in key_fields)][r[x_field]].append(r[y_field])
    out = {}
    for k, v in dct.items():
        xs = sorted(v.keys())
        ys = [reducer(v[x]) for x in xs]
        out[k] = (np.array(xs), np.array(ys))
    return out


rows = load_rows(out_csv)

# ---- k-Wave: runtime vs grid size, split by CPU/GPU ----
for dim in (2, 3):
    data = agg_by(
        [r for r in rows if r["backend"] == "kwave" and r["d"] == dim], ["is_gpu"], "n"
    )
    if not data:
        continue
    plt.figure()
    for (is_gpu,), (xs, ys) in sorted(data.items(), key=lambda kv: kv[0]):
        lbl = "GPU" if is_gpu else "CPU"
        plt.loglog(xs, ys, "o-", label=lbl)
    plt.xlabel(r"Grid size $N$")
    plt.ylabel("Runtime (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / f"kwave_{dim}d_runtime_from_csv.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

# ---- GB: runtime vs grid size, separate lines per bundle size b ----
for dim in (1, 2, 3):
    subset = [r for r in rows if r["backend"] == "gb" and r["d"] == dim]
    if not subset:
        continue
    lines = agg_by(subset, ["b"], "n")
    plt.figure()
    for (b,), (xs, ys) in sorted(lines.items(), key=lambda kv: kv[0][0]):
        plt.loglog(xs, ys, "o-", label=f"b={b}")
    plt.xlabel(r"Grid size $N$")
    plt.ylabel("Runtime (s)")
    plt.legend(title="Bundles")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / f"gb_{dim}d_runtime_vs_n_by_b.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

# ---- GB: runtime vs bundles b, separate lines per grid size N (optional) ----
for dim in (1, 2, 3):
    subset = [r for r in rows if r["backend"] == "gb" and r["d"] == dim]
    if not subset:
        continue
    lines = agg_by(subset, ["n"], "b")
    plt.figure()
    for (n,), (xs, ys) in sorted(lines.items(), key=lambda kv: kv[0][0]):
        plt.loglog(xs, ys, "o-", label=f"N={n}")
    plt.xlabel(r"Bundles $b$")
    plt.ylabel("Runtime (s)")
    plt.legend(title="Grid size")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / f"gb_{dim}d_runtime_vs_b_by_n.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
