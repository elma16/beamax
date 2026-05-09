#!/usr/bin/env python
# coding: utf-8

# # Investigating Strong and Weak Scaling
#
# Gaussian beams are embarrassingly parallelisable. We investigate to what extent this is the case.
#
# In relation to the thesis, we do experiments on google colab with a 8 cores of a v2 TPU.


"""
Aim: Work out the scalability of the ODE solver across multiple devices.
Specifically, I want to investigate strong and weak scaling.
"""

import jax.numpy as jnp
import jax
from time import time
from functools import partial
from jax.sharding import NamedSharding, PartitionSpec

from beamax import geometry, utils
from beamax.gb import core, gb_utils, gb_solvers
from beamax.plotter import use_beamax_style
from pathlib import Path
from beamax.solvers.kwave_solver import TimedKWaveSolver

from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
CACHE_DIR = Path(ROOT_DIR / "cache")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt

use_beamax_style()

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

num_devices = jax.device_count()
print(f"Number of devices: {num_devices}")

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_CPU_DEVICE_COUNT"] = "8"

if num_devices < 2:
    print("Skipping example: requires at least 2 devices.")
    raise SystemExit(0)
mesh = jax.make_mesh((8, 1), ("x", "y"))

d = 2
periodic = (True,) * d
cfl = 0.3


def c(x):
    return 1 + 0 * x[..., 0]


simulation_options = SimulationOptions(
    data_cast="double",
    smooth_p0=False,
    save_to_disk=True,
)

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0
)
kwave_solver = TimedKWaveSolver(simulation_options, execution_options)

solver_ode = gb_solvers.solve_ODE_base
solver_config = None


@partial(jax.jit, static_argnums=(6, 11, 12))
def compute_gb_sum(
    x0, p0, M0, a0, ω0, mode, c, ts, XY, domain_size, periodic, solver, solver_config
):
    return jnp.sum(
        core.compute_gaussian_beam(
            x0,
            p0,
            M0,
            a0,
            ω0,
            mode,
            c,
            ts,
            XY,
            domain_size,
            jnp.array(periodic),
            solver,
            solver_config,
        ),
        axis=-1,
    ).real


gb_ode_runtimes = []
gb_hom_runtimes = []

start = 4
end = 9
num_points = end - start + 1
exponents = jnp.linspace(start, end, num_points)
powers_of_2 = 2**exponents
ns = powers_of_2.astype(int)

for b in [16]:
    ω0 = jnp.ones((b,))
    alpha0 = jnp.ones((b, d)) * 1j
    M0 = None
    M0 = gb_utils.prepare_M0(alpha0, M0)
    is_M0_diagonal = gb_utils.is_diagonal(M0)

    mode = jnp.ones((b,))
    p0 = jnp.zeros((b, d))
    p0 = p0.at[:, 0].set(1)
    p0 = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)
    a0 = jnp.ones((b,))

    gb_ode_runtimes = []

    for n in ns:
        print(f"n = {n}")
        N = (n,) * d
        dx = (1 / N[0],) * d
        domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
        ts = domain.generate_time_domain()
        ts_subsample = jnp.linspace(0, len(ts) - 1, 5).astype(int)
        ts_small = ts[ts_subsample]

        # sensors_all = jnp.ones(N)
        sensors_all = jnp.zeros(N)
        sensors_all = sensors_all.at[0, 0].set(1)
        sensors = geometry.Sensor(domain=domain, binary_mask=sensors_all)
        x0 = (domain.grid_size * 0.5).repeat(b).reshape(b, d)

        x0s = jax.device_put(x0, NamedSharding(mesh, PartitionSpec("x", "y")))
        p0s = jax.device_put(p0, NamedSharding(mesh, PartitionSpec("x", "y")))
        M0s = jax.device_put(M0, NamedSharding(mesh, PartitionSpec("x", "y")))
        a0s = jax.device_put(a0, NamedSharding(mesh, PartitionSpec("x")))
        ω0s = jax.device_put(ω0, NamedSharding(mesh, PartitionSpec("x")))
        modes = jax.device_put(mode, NamedSharding(mesh, PartitionSpec("x")))

        # a0s = jax.device_put(a0, NamedSharding(mesh, PartitionSpec("x", "y")))
        # ω0s = jax.device_put(ω0, NamedSharding(mesh, PartitionSpec("x", "y")))
        # modes = jax.device_put(mode, NamedSharding(mesh, PartitionSpec("x", "y")))
        # print("y shape", y.shape)
        # # jax.debug.visualize_array_sharding(y)
        # z = jnp.sin(y)
        # (xt, pt, mt, at) = solver(x0, p0, M0, a0, mode, ts, c, solver_config)

        t1 = time()
        u0_all = compute_gb_sum(
            x0,
            p0,
            M0,
            a0,
            ω0,
            mode,
            c,
            ts,
            sensors.positions,
            domain.grid_size,
            periodic,
            solver_ode,
            solver_config,
        ).block_until_ready()
        t2 = time()
        gb_ode_runtimes.append(t2 - t1)
        print(f"GB runtime: {t2 - t1}")

    plt.semilogy(ns, gb_ode_runtimes, "-o", label="GB ODE Solver, b = {}".format(b))
plt.xlabel("N")
plt.ylabel("log(Runtime (s))")
plt.legend()
plt.title("GB ODE Solver Runtime for {d}D".format(d=d))
plt.savefig(PLOT_DIR / "gb-ode-runtime-3d.png")
plt.show()

"""
Aim: Work out the scalability of the ODE solver across multiple devices.
Specifically, I want to investigate strong and weak scaling.
"""

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from time import time
from functools import partial
from jax.sharding import NamedSharding, PartitionSpec

from beamax import geometry
from beamax.gb import gb_utils, gb_solvers
from pathlib import Path
from beamax.solvers.kwave_solver import TimedKWaveSolver

from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

ROOT_DIR = utils.detect_root()
CACHE_DIR = ROOT_DIR / "cache"
PLOT_DIR = ROOT_DIR / "plots"

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


@partial(jax.jit, static_argnums=(6, 11, 12))
def compute_gb_sum(
    x0, p0, M0, a0, ω0, mode, c, ts, XY, domain_size, periodic, solver, solver_config
):
    return jnp.sum(
        core.compute_gaussian_beam(
            x0,
            p0,
            M0,
            a0,
            ω0,
            mode,
            c,
            ts,
            XY,
            domain_size,
            jnp.array(periodic),
            solver,
            solver_config,
        ),
        axis=-1,
    ).real


# # Strong Scaling

num_devices = jax.device_count()
print(f"Number of devices: {num_devices}")
if num_devices < 2:
    print("Skipping example: requires at least 2 devices.")
    raise SystemExit(0)
mesh = jax.make_mesh((8, 1), ("x", "y"))

d = 2
periodic = (True,) * d
cfl = 0.3


def c(x):
    return 1 + 0 * x[..., 0]


simulation_options = SimulationOptions(
    data_cast="double",
    smooth_p0=False,
    save_to_disk=True,
)

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=False, delete_data=False, verbose_level=0
)

# kwave_solver = TimedKWaveSolver(simulation_options, execution_options)

solver_ode = gb_solvers.solve_ODE_base
solver_config = None

gb_ode_runtimes = []
gb_hom_runtimes = []

start = 4
end = 9
num_points = end - start + 1
exponents = jnp.linspace(start, end, num_points)
powers_of_2 = 2**exponents
ns = powers_of_2.astype(int)

N = (128,) * d
dx = (1 / N[0],) * d
domain = geometry.Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)
ts = domain.generate_time_domain()
ts_subsample = jnp.linspace(0, len(ts) - 1, 5).astype(int)
ts_small = ts[ts_subsample]

for b in [16, 32, 64, 128, 256, 512]:
    ω0 = jnp.ones((b,))
    alpha0 = jnp.ones((b, d)) * 1j
    M0 = None
    M0 = gb_utils.prepare_M0(alpha0, M0)
    is_M0_diagonal = gb_utils.is_diagonal(M0)

    mode = jnp.ones((b,))
    p0 = jnp.zeros((b, d))
    p0 = p0.at[:, 0].set(1)
    p0 = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)
    a0 = jnp.ones((b,))

    gb_ode_runtimes = []

    sensors_all = jnp.zeros(N)
    sensors_all = sensors_all.at[0, :].set(1)
    sensors = geometry.Sensor(domain=domain, binary_mask=sensors_all)
    x0 = (domain.grid_size * 0.5).repeat(b).reshape(b, d)

    x0s = jax.device_put(x0, NamedSharding(mesh, PartitionSpec("x", "y")))
    p0s = jax.device_put(p0, NamedSharding(mesh, PartitionSpec("x", "y")))
    M0s = jax.device_put(M0, NamedSharding(mesh, PartitionSpec("x", "y")))
    a0s = jax.device_put(a0, NamedSharding(mesh, PartitionSpec("x")))
    ω0s = jax.device_put(ω0, NamedSharding(mesh, PartitionSpec("x")))
    modes = jax.device_put(mode, NamedSharding(mesh, PartitionSpec("x")))

    # a0s = jax.device_put(a0, NamedSharding(mesh, PartitionSpec("x", "y")))
    # ω0s = jax.device_put(ω0, NamedSharding(mesh, PartitionSpec("x", "y")))
    # modes = jax.device_put(mode, NamedSharding(mesh, PartitionSpec("x", "y")))
    # print("y shape", y.shape)
    # # jax.debug.visualize_array_sharding(y)
    # z = jnp.sin(y)
    # (xt, pt, mt, at) = solver(x0, p0, M0, a0, mode, ts, c, solver_config)

    t1 = time()
    u0_all = compute_gb_sum(
        x0,
        p0,
        M0,
        a0,
        ω0,
        mode,
        c,
        ts,
        sensors.positions,
        domain.grid_size,
        periodic,
        solver_ode,
        solver_config,
    ).block_until_ready()
    t2 = time()
    print("warmup", t2 - t1)
    t1 = time()
    u0_all = compute_gb_sum(
        x0,
        p0,
        M0,
        a0,
        ω0,
        mode,
        c,
        ts,
        sensors.positions,
        domain.grid_size,
        periodic,
        solver_ode,
        solver_config,
    ).block_until_ready()
    t2 = time()
    print(f"GB runtime: {t2 - t1}")

    gb_ode_runtimes.append(t2 - t1)

#     plt.semilogy(ns, gb_ode_runtimes, "-o", label="GB ODE Solver, b = {}".format(b))
# plt.xlabel("N")
# plt.ylabel("log(Runtime (s))")
# plt.legend()
# plt.title("GB ODE Solver Runtime for {d}D".format(d=d))
# plt.savefig(PLOT_DIR / "gb-ode-runtime-3d.png")
# plt.show()

# ––––––– STRONG SCALING (N & b_total fixed, vary P) –––––––
N = (128, 128)
b = 2400
for P in [1, 2, 4, 8]:
    mesh = jax.make_mesh((P, 1), ("x", "y"))

    ω0 = jnp.ones((b,))
    alpha0 = jnp.ones((b, d)) * 1j
    M0 = None
    M0 = gb_utils.prepare_M0(alpha0, M0)
    is_M0_diagonal = gb_utils.is_diagonal(M0)

    mode = jnp.ones((b,))
    p0 = jnp.zeros((b, d))
    p0 = p0.at[:, 0].set(1)
    p0 = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)
    a0 = jnp.ones((b,))

    gb_ode_runtimes = []

    sensors_all = jnp.zeros(N)
    sensors_all = sensors_all.at[0, :].set(1)
    sensors = geometry.Sensor(domain=domain, binary_mask=sensors_all)
    x0 = (domain.grid_size * 0.5).repeat(b).reshape(b, d)

    x0s = jax.device_put(x0, NamedSharding(mesh, PartitionSpec("x", None)))
    p0s = jax.device_put(p0, NamedSharding(mesh, PartitionSpec("x", None)))
    M0s = jax.device_put(M0, NamedSharding(mesh, PartitionSpec("x", None, None)))
    a0s = jax.device_put(a0, NamedSharding(mesh, PartitionSpec("x")))
    ω0s = jax.device_put(ω0, NamedSharding(mesh, PartitionSpec("x")))
    modes = jax.device_put(mode, NamedSharding(mesh, PartitionSpec("x")))

    # print(x0s.shape, p0s.shape, M0s.shape, a0s.shape, ω0s.shape, modes.shape)
    # jax.debug.visualize_array_sharding(x0s)

    # initialize x0, p0, M0, a0, ω0, mode for exactly b_total beams
    # device_put so each of the P devices holds ~ b_total/P beams
    u0_all = compute_gb_sum(
        x0s,
        p0s,
        M0s,
        a0s,
        ω0s,
        modes,
        c,
        ts,
        sensors.positions,
        domain.grid_size,
        periodic,
        solver_ode,
        solver_config,
    ).block_until_ready()

    t0 = time()
    u0_all = compute_gb_sum(
        x0s,
        p0s,
        M0s,
        a0s,
        ω0s,
        modes,
        c,
        ts,
        sensors.positions,
        domain.grid_size,
        periodic,
        solver_ode,
        solver_config,
    ).block_until_ready()
    t1 = time()
    print(f"Strong: P={P}, time={t1 - t0}s")

# # Strong Scaling

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import matplotlib.pyplot as plt
from time import time

# Assume d, c, periodic, cfl, solver_ode, solver_config, geometry, gb_utils are defined
# Also assume domain, ts, sensors are defined for N=(128,128)

b_total_list = [600, 1200, 2400]
P_list = [1, 2, 4, 8]
runtimes_strong = {b: {} for b in b_total_list}

for b in b_total_list:
    for P in P_list:
        mesh = jax.make_mesh((P, 1), ("x", "y"))
        ω0 = jnp.ones((b,))
        alpha0 = jnp.ones((b, d)) * 1j
        M0 = gb_utils.prepare_M0(alpha0, None)
        mode = jnp.ones((b,))
        p0 = jnp.zeros((b, d))
        p0 = p0.at[:, 0].set(1.0)
        p0 = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)
        a0 = jnp.ones((b,))
        x0 = (domain.grid_size * 0.5).repeat(b).reshape(b, d)
        x0s = jax.device_put(x0, NamedSharding(mesh, PartitionSpec("x", None)))
        p0s = jax.device_put(p0, NamedSharding(mesh, PartitionSpec("x", None)))
        M0s = jax.device_put(M0, NamedSharding(mesh, PartitionSpec("x", None, None)))
        a0s = jax.device_put(a0, NamedSharding(mesh, PartitionSpec("x")))
        ω0s = jax.device_put(ω0, NamedSharding(mesh, PartitionSpec("x")))
        modes = jax.device_put(mode, NamedSharding(mesh, PartitionSpec("x")))
        compute_gb_sum(
            x0s,
            p0s,
            M0s,
            a0s,
            ω0s,
            modes,
            c,
            ts,
            sensors.positions,
            domain.grid_size,
            periodic,
            solver_ode,
            solver_config,
        ).block_until_ready()
        t0 = time()
        compute_gb_sum(
            x0s,
            p0s,
            M0s,
            a0s,
            ω0s,
            modes,
            c,
            ts,
            sensors.positions,
            domain.grid_size,
            periodic,
            solver_ode,
            solver_config,
        ).block_until_ready()
        t1 = time()
        runtimes_strong[b][P] = t1 - t0

# Plot runtime vs P
plt.figure(figsize=(5, 4))
for b in b_total_list:
    y = [runtimes_strong[b][P] for P in P_list]
    plt.plot(P_list, y, "-o", label=f"b_total = {b}")
plt.xlabel("Number of devices (P)")
plt.ylabel("Runtime (s)")
plt.xticks(P_list)
plt.title("Strong Scaling: Runtime vs. P")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()

# Plot efficiency vs P
plt.figure(figsize=(5, 4))
for b in b_total_list:
    T1 = runtimes_strong[b][1]
    effs = [100.0 * T1 / (P * runtimes_strong[b][P]) for P in P_list]
    plt.plot(P_list, effs, "-o", label=f"b_total = {b}")
plt.hlines(
    100.0, P_list[0], P_list[-1], colors="gray", linestyles="--", label="Ideal (100%)"
)
plt.xlabel("Number of devices (P)")
plt.ylabel("Efficiency (%)")
plt.xticks(P_list)
plt.ylim(0, 110)
plt.title("Strong Scaling Efficiency vs. P")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()

# Print efficiency table
print("\nStrong-Scaling Efficiency (%)\n" + "-" * 34)
print(f"{'b_total':>8} | " + " | ".join(f"P={P:>2d}" for P in P_list))
print("-" * 8 + "-+-" + "-+-".join("------" for _ in P_list))
for b in b_total_list:
    T1 = runtimes_strong[b][1]
    effs = [100.0 * T1 / (P * runtimes_strong[b][P]) for P in P_list]
    effs_str = " | ".join(f"{eff:6.1f}" for eff in effs)
    print(f"{b:8d} | {effs_str}")

# # Weak Scaling

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import matplotlib.pyplot as plt
from time import time

# ———— PARAMETERS YOU’VE ALREADY DEFINED ABOVE ————
# d, c, periodic, cfl, solver_ode, solver_config, geometry, gb_utils, etc.
# domain, ts, sensors (for a fixed N), etc.
#
# In particular:
#   * N = (128,128)                                     # 2D grid
#   * domain = geometry.Domain(N, dx, c, periodic, cfl)
#   * ts     = domain.generate_time_domain()
#   * sensors = geometry.Sensor(domain=domain, binary_mask=some_mask)
#
# Make sure `compute_gb_sum(...)` is JIT‐compiled exactly as before
# (with static_argnums set appropriately).

# ———— WEAK‐SCALING SWEEP ————
N = (128, 128)

# List of “beams per device” that we want to test:
b_per_device_list = [100, 200, 400]

# We will sweep over P = [1,2,4,8]; for each b_per_device, record runtime for each P
P_list = [1, 2, 4, 8]

# Data structure to hold runtimes:
#   runtimes[bpd][P] = measured_time_in_seconds
runtimes = {bpd: {} for bpd in b_per_device_list}

for bpd in b_per_device_list:
    print(f"\n=== Sweeping for b_per_device = {bpd} ===")
    for P in P_list:
        mesh = jax.make_mesh((P, 1), ("x", "y"))
        b_total = bpd * P

        # ——— initialize per‐beam arrays of length b_total ———
        ω0 = jnp.ones((b_total,))
        alpha0 = jnp.ones((b_total, d)) * 1j
        M0 = gb_utils.prepare_M0(alpha0, None)
        mode = jnp.ones((b_total,))
        p0 = jnp.zeros((b_total, d))
        p0 = p0.at[:, 0].set(1.0)
        p0 = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)
        a0 = jnp.ones((b_total,))

        # ——— domain & sensors are held fixed (N = (128,128)) ———
        # (reuse the same domain, ts, sensors from above if you want)
        # If you truly need to rebuild domain each time, uncomment:
        # domain = geometry.Domain(N, (1/N[0], 1/N[1]), c, periodic, cfl)
        # ts     = domain.generate_time_domain()
        # sensors_all = jnp.zeros(N)
        # sensors_all = sensors_all.at[0, :].set(1)
        # sensors = geometry.Sensor(domain=domain, binary_mask=sensors_all)

        # ——— build x0 = (grid_center) repeated b_total times ———
        x0 = (domain.grid_size * 0.5).repeat(b_total).reshape(b_total, d)

        # ——— SHARD every array along axis 0 (“x”) ———
        x0s = jax.device_put(x0, NamedSharding(mesh, PartitionSpec("x", None)))
        p0s = jax.device_put(p0, NamedSharding(mesh, PartitionSpec("x", None)))
        M0s = jax.device_put(M0, NamedSharding(mesh, PartitionSpec("x", None, None)))
        a0s = jax.device_put(a0, NamedSharding(mesh, PartitionSpec("x")))
        ω0s = jax.device_put(ω0, NamedSharding(mesh, PartitionSpec("x")))
        modes = jax.device_put(mode, NamedSharding(mesh, PartitionSpec("x")))

        # ——— optional: double‐check each shard’s shape: ———
        # print(f"P={P}: x0s.shape → {x0s.shape},  p0s.shape → {p0s.shape}")

        # ——— WARM UP (compile + partition) ———
        compute_gb_sum(
            x0s,
            p0s,
            M0s,
            a0s,
            ω0s,
            modes,
            c,
            ts,
            sensors.positions,
            domain.grid_size,
            periodic,
            solver_ode,
            solver_config,
        ).block_until_ready()

        # ——— NOW TIME THE SECOND (pure run) CALL ———
        t_start = time()
        compute_gb_sum(
            x0s,
            p0s,
            M0s,
            a0s,
            ω0s,
            modes,
            c,
            ts,
            sensors.positions,
            domain.grid_size,
            periodic,
            solver_ode,
            solver_config,
        ).block_until_ready()
        t_end = time()

        elapsed = t_end - t_start
        runtimes[bpd][P] = elapsed
        print(f"  P={P:<2d}  →  runtime = {elapsed:.3f} s")

# ——— PLOTTING “runtime vs P” for each b_per_device ———
plt.figure(figsize=(5, 4))
for bpd in b_per_device_list:
    y = [runtimes[bpd][P] for P in P_list]
    plt.plot(P_list, y, "-o", label=f"bpd = {bpd}")
plt.xlabel("Number of devices (P)")
plt.ylabel("Runtime (s)")
plt.xticks(P_list)
plt.title("Weak‐Scaling: Runtime vs. P (N fixed)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()

# ——— PRINT WEAK‐SCALING EFFICIENCY TABLE ———
# Efficiency(P) = (T1 / T_P) * 100%
print("\nWeak‐Scaling Efficiency (%)\n" + "-" * 30)
print(f"{'bpd':>6} | " + " | ".join(f"P={P:>2d}" for P in P_list))
print("-" * 6 + "-+-" + "-+-".join("------" for _ in P_list))
for bpd in b_per_device_list:
    T1 = runtimes[bpd][1]
    effs = [100 * T1 / runtimes[bpd][P] for P in P_list]
    effs_str = " | ".join(f"{eff:6.1f}" for eff in effs)
    print(f"{bpd:6d} | {effs_str}")

import matplotlib.pyplot as plt

# ——— PLOTT “runtime vs P” for each b_per_device ———
plt.figure(figsize=(5, 4))
for bpd in b_per_device_list:
    y = [runtimes[bpd][P] for P in P_list]
    plt.plot(P_list, y, "-o", label=f"bpd = {bpd}")
plt.xlabel("Number of devices (P)")
plt.ylabel("Runtime (s)")
plt.xticks(P_list)
plt.title("Weak‐Scaling: Runtime vs. P (N fixed)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()

# ——— COMPUTE AND PLOT EFFICIENCY (%) vs P WITH IDEAL LINE ———
plt.figure(figsize=(5, 4))
for bpd in b_per_device_list:
    T1 = runtimes[bpd][1]
    effs = [100.0 * T1 / runtimes[bpd][P] for P in P_list]
    plt.plot(P_list, effs, "-o", label=f"bpd = {bpd}")

# dashed horizontal line at 100% efficiency
plt.hlines(
    100.0, P_list[0], P_list[-1], colors="gray", linestyles="--", label="Ideal (100%)"
)

plt.xlabel("Number of devices (P)")
plt.ylabel("Efficiency (%)")
plt.xticks(P_list)
plt.ylim(0, 110)
plt.title("Weak‐Scaling Efficiency vs. P")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5)
plt.show()
