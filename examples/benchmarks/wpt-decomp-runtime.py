#!/usr/bin/env python
# coding: utf-8

"""
Runtime of the MSWPT forward and inverse transforms as a function of grid size and decomposition depth.
"""
# # WPT runtime



import jax.numpy as jnp
import jax as jax
from time import perf_counter, time
import matplotlib.pyplot as plt
from pathlib import Path

from functools import partial
from beamax import plotter, utils
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT
from beamax.plotter import use_beamax_style

ROOT_DIR = utils.detect_root()
PLOT_DIR = Path(ROOT_DIR / "plots")
DATA_DIR = Path(ROOT_DIR / "data")
PLOT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

use_beamax_style()

jax.config.update("jax_enable_x64", False)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

pltgb = plotter.PlotHelper()

d = 2
dx = (1e-4,) * d
periodic = (True,) * d
box_aspect_ratio = (1,) * d
num_levels = 4
num_boxes_level = (4,) * num_levels
windowing = "rectangular_mirror"
input_type = "spatial"
output_type = "spatial"
redundancy = 2
cfl = (jnp.sqrt(2) / 4).round(3)


def c(x):
    return 1500 + 0 * x[..., 0]


N = (128,) * d
t1 = time()
dyadic_decomp = DyadicDecomposition(num_levels, N, num_boxes_level, box_aspect_ratio)
wpt = MSWPT(dyadic_decomp, redundancy, windowing)
# wptNone = MSWPT(dyadic_decomp, redundancy, "none")
t2 = time()
print("Time to create params", t2 - t1)

small_gfilts = wpt.gfilts_packed


@partial(jax.jit)
def fn(p0):
    return wpt.forward(p0, input_type="spatial")


q0 = jnp.zeros(N)
p0 = jnp.ones(N)

t1 = time()
_ = fn(q0).block_until_ready()
t2 = time()
print("Time to warmup", t2 - t1)

t1 = time()
coeffs = fn(p0).block_until_ready()
t2 = time()
print("Time to forward", t2 - t1)

coeffs_array = wpt.convert_to_array(coeffs)

plt.imshow(jnp.abs(coeffs_array))
plt.title("coeffs")
plt.colorbar()
plt.savefig(DATA_DIR / "coeffs.png", dpi=300, bbox_inches="tight")
plt.show()

import jax.tree_util as tree_util
from argparse import ArgumentParser
import numpy as np

pltgb = plotter.PlotHelper()


def block(x):
    [a.block_until_ready() for a in tree_util.tree_leaves(x)]


def time_fn(f, args, n):
    block(f(*args))
    ts = []
    for _ in range(n):
        t0 = perf_counter()
        out = f(*args)
        block(out)
        ts.append(perf_counter() - t0)
    return np.mean(ts), np.std(ts)


def main():
    p = ArgumentParser()
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--jit", action="store_true")
    args = p.parse_args()

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

    d = 2
    N = (128,) * d
    box_aspect = (1,) * d
    num_levels = 3
    num_boxes = (4,) * num_levels

    decomp = DyadicDecomposition(num_levels, N, num_boxes, box_aspect)
    wpt = MSWPT(decomp, 2, "rectangular_mirror")
    fn = wpt.forward
    if args.jit:
        fn = jax.jit(fn)

    p0 = jnp.ones(N)
    mean, sd = time_fn(fn, (p0, "spatial"), args.runs)
    print(
        f"forward ({'jit' if args.jit else 'eager'}) over {args.runs} runs: "
        f"{mean * 1e3:.3f} ms ± {sd * 1e3:.3f} ms"
    )


main()
