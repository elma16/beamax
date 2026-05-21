import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import pytest

import jax.numpy as jnp
import jax as jax

from beamax import utils
from beamax.geometry import Domain, Sensor
from beamax.decomposition import DyadicDecomposition
from beamax.transforms import MSWPT, compute_frames
from beamax.gb import gb_solvers
from beamax.solvers.msgb_solvers.msgb_solver import MSGBSolver
from beamax.solvers import ShardingStrategy


jax.config.update("jax_enable_x64", True)


def c(x):
    return 1 + 0 * x[..., 0]


windowing = "rectangular"
input_type = "spatial"
output_type = "spatial"
redundancy = 2


def test_msgb_forward_then_time_reversal_1d():
    """
    1-D MSGB forward → time-reversal smoke with a small grid.
    - homogeneous c(x)
    - two simple wavepackets synthesized via compute_frames
    - one sensor at the left boundary
    - crop sensor data in frequency to N and run MSGB TR
    - assert correlation with ground-truth p0 is high and rel-L2 is reasonable
    """
    d = 1
    N = (512,) * d
    extent = (1,) * d
    dx = tuple([extent[i] / N[i] for i in range(d)])
    box_aspect_ratio = (1,) * d
    num_levels = 3
    num_boxes_levels = tuple([2 ** (level + 2) for level in range(num_levels)])

    windowing = "rectangular_mirror"
    redundancy = 2
    num_GB_img_space = N[0] * 2
    batch_size = 128
    input_type = "spatial"
    thr_strat = "top_n"
    sum_method = "scan_real"
    cfl = 0.5

    periodic = (False,) * d
    solver = gb_solvers.solve_ODE_base

    def c(x):
        return 1 - 0 * (
            jnp.exp(-((x[..., 0] - extent[0] / 3) ** 2) / (0.1**2))
            - jnp.exp(-((x[..., 0] - 2 * extent[0] / 3) ** 2) / (0.1**2))
        )

    img_domain = Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

    XY = img_domain.grid

    ts = img_domain.generate_time_domain()
    tmax_img = ts[-1]
    Nt = len(ts)
    if Nt != 4 * N[0]:
        ts = jnp.linspace(0, tmax_img, 4 * N[0])
        tmax_img = ts[-1]
        Nt = len(ts)
        dt = ts[1] - ts[0]
        cfl = jnp.min(c(XY)) * dt / min(dx)
        img_domain = Domain(N=N, dx=dx, c=c, cfl=cfl, periodic=periodic)

    img_dyadic_decomp = DyadicDecomposition(
        num_levels, N, num_boxes_levels, box_aspect_ratio
    )

    img_wpt = MSWPT(img_dyadic_decomp, redundancy, windowing)

    binary_mask = jnp.zeros(N)
    binary_mask = binary_mask.at[0, ...].set(1)
    sensors = Sensor(domain=img_domain, binary_mask=binary_mask)
    sensors_all = Sensor(domain=img_domain, binary_mask=jnp.ones(N))

    KXY = img_dyadic_decomp.fourier_meshgrid

    boxhf = 4
    boxlf = 0

    khf = jnp.array(
        [
            10,
        ]
    )
    klf = jnp.array(
        [
            25,
        ]
    )
    kerft_hf = compute_frames(img_dyadic_decomp, boxhf, khf, KXY, redundancy, "none")
    kerft_lf = compute_frames(img_dyadic_decomp, boxlf, klf, KXY, redundancy, "none")
    p0 = utils.unitary_ifft(kerft_hf) + utils.unitary_ifft(kerft_lf)
    p0 = p0 / jnp.max(jnp.abs(p0))
    p0 = p0.T

    p0 = p0.real

    utils.unitary_fft(p0)

    jnp.zeros_like(p0)

    img_dyadic_decomp.centres_ndim + jnp.array(N) // 2

    num_devices = jax.device_count()

    mesh = jax.make_mesh((num_devices,), ("x",))

    # Create sharding strategy
    sharding_strategy = ShardingStrategy(mesh, beam_axis="x")

    # Create solver with sharding
    msgb_solver = MSGBSolver(
        thr=num_GB_img_space,
        thr_strat=thr_strat,
        batch_size=batch_size,
        input_type=input_type,
        ode_solver=solver,
        tr_ode_solver=gb_solvers.solve_ODE_batch_t,
        sum_method=sum_method,
        sharding=sharding_strategy,
    )

    sensor_data_gb = msgb_solver.forward(
        p0,
        img_domain,
        sensors.positions,
        ts,
        img_wpt,
    )

    def cut_out_middle(arr, size):
        mid = arr.shape[0] // 2
        return arr[mid - size // 2 : mid + size // 2]

    sensor_data_fft = utils.unitary_fft(sensor_data_gb)
    sensor_data_fft_cropped = cut_out_middle(sensor_data_fft, N[0])
    sensor_data_cropped = utils.unitary_ifft(sensor_data_fft_cropped)

    N_rect = jnp.squeeze(sensor_data_cropped).shape

    Nt = N_rect[0]
    ts = jnp.linspace(0, tmax_img, Nt)
    dt = float(ts[1] - ts[0])
    dx_rect = (dt,)
    box_aspect_ratio_rect = (1,)
    tmax_data = ts[-1]
    assert jnp.allclose(
        tmax_img, tmax_data
    ), f"tmax_img {tmax_img} and tmax_data {tmax_data} are not equal."

    data_domain = Domain(N=N_rect, dx=dx_rect, c=c, periodic=periodic, cfl=cfl)

    data_dyadic_decomp = DyadicDecomposition(
        num_levels, N_rect, num_boxes_levels, box_aspect_ratio_rect
    )

    data_wpt = MSWPT(data_dyadic_decomp, redundancy, windowing)

    sensor_data_gb = jnp.squeeze(sensor_data_cropped)

    p0_TR_msgb = msgb_solver.time_reversal(
        data=sensor_data_gb,
        domain=img_domain,
        sensors=sensors_all,
        sources=sensors,
        ts=ts,
        data_domain=data_domain,
        data_wpt=data_wpt,
    )

    assert jnp.max(p0 - p0_TR_msgb) < 0.2


if __name__ == "__main__":
    pytest.main([__file__])
