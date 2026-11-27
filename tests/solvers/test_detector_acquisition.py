"""Regression reference for the public MSGB forward and ring reconstructions.

Values were captured on CPU at 6c68fe9 with JAX 0.11.0 in float64.
The four pressure columns follow the binary mask's C order. Both images
contain signed pressure on the same region of interest, without rescaling.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from beamax import Domain, DyadicDecomposition, MSWPT, Sensor
from beamax.gb.gb_solvers import solve_hom_diag
from beamax.solvers import MSGBSolver
from beamax.surfaces import DetectorSurface, ring_adjoint, surface_time_reversal


MASK_PRESSURE = [
    [-0.12126262944, 0.07375919897, -0.01573595971, 0.01912693425],
    [-0.13133307974, 0.08355821481, -0.00940159360, 0.02244953466],
    [-0.14690135500, 0.11390184105, 0.00067341355, 0.04317504514],
    [-0.14566352233, 0.15765025799, -0.00258205272, 0.10000501970],
    [-0.11829461487, 0.19169993289, -0.02563747033, 0.18715178446],
    [-0.05851562329, 0.18927245989, -0.05193684961, 0.25632064512],
    [0.03636080016, 0.13583106868, -0.04924888121, 0.24602739376],
    [0.14967090956, 0.03808472705, -0.00004330704, 0.13854867203],
    [0.24595910496, -0.07384782924, 0.07520391244, -0.01516994283],
    [0.28518439665, -0.15692553460, 0.13094673627, -0.13514687018],
    [0.24188451016, -0.17873151933, 0.13399330589, -0.17738405346],
    [0.13121290156, -0.14066621337, 0.09160546589, -0.15456408491],
    [0.01328635765, -0.07802659187, 0.04094348142, -0.10427995333],
    [-0.05135887450, -0.03008517603, 0.00923269375, -0.05613615018],
    [-0.05185918881, -0.00996471347, -0.00496662620, -0.02419060522],
    [-0.02291371719, -0.00508432205, -0.01299976348, -0.01031267726],
]

TIME_REVERSAL_IMAGE = [
    [-0.04377939418, 0.01262604250, -0.09068079501],
    [-0.02458529000, 0.18527902467, 0.08424860354],
    [0.29028135757, 0.74192334635, 0.50984449734],
    [0.03776720527, 0.60226519660, 0.88578156971],
]

ADJOINT_IMAGE = [
    [-0.03152086455, 0.01144802961, -0.07517168621],
    [-0.02128373887, 0.08015675525, 0.03377271412],
    [0.13575694794, 0.34567146369, 0.18745091601],
    [0.01094507690, 0.29710437190, 0.44044245745],
]


@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [(np.float32, 2e-5, 1e-6), (np.float64, 1e-8, 1e-10)],
    ids=["float32", "float64"],
)
def test_asymmetric_ring_acquisition(dtype, rtol, atol):
    with jax.enable_x64(dtype is np.float64):
        ds, dt = 0.125, 0.0625
        domain = Domain((16, 16), (ds, ds), (False, False), c=1.0)
        centre, radius = (1.0, 0.875), 0.5
        ring = DetectorSurface.circle(centre, radius, n=4)
        # In mask order the receivers are left, bottom, top, right.
        mask = np.zeros(domain.N, dtype=bool)
        mask[[4, 8, 8, 12], [7, 3, 11, 7]] = True
        sensors = Sensor(domain, binary_mask=mask)
        chart_order = ring.mask_to_chart(domain)
        np.testing.assert_array_equal(chart_order, [3, 2, 0, 1])

        x, y = np.indices(domain.N)
        p0 = np.exp(-((x - 9) ** 2 / 2 + (y - 6) ** 2 / 4))
        p0 -= 0.65 * np.exp(-((x - 6) ** 2 / 3 + (y - 8) ** 2 / 2))
        ts = jnp.arange(16) * dt
        image_wpt = MSWPT(
            DyadicDecomposition(1, domain.N, (4,), (1, 1)), 2, "rectangular_mirror"
        )
        solver = MSGBSolver(
            thr=128,
            thr_strat="top_n",
            batch_size=32,
            input_type="spatial",
            ode_solver=solve_hom_diag,
            sum_method="scan_real",
        )
        pressure = solver.forward(
            jnp.asarray(p0, dtype=dtype), domain, sensors, ts, image_wpt
        )
        assert pressure.dtype == dtype
        np.testing.assert_allclose(pressure, MASK_PRESSURE, rtol=rtol, atol=atol)

        # Both inverse methods take samples in angular order starting at +x.
        traces = pressure[:, chart_order]
        acquisition = Domain(traces.shape, (dt, *ring.spacing), (False, True))
        trace_wpt = MSWPT(
            DyadicDecomposition(1, acquisition.N, (2,), (4, 1)), 2, "rectangular_mirror"
        )
        points = domain.grid[6:10, 5:8]
        tr_image, _ = surface_time_reversal(
            traces,
            acquisition,
            trace_wpt,
            ring,
            domain,
            80,
            eval_points=points,
            batch_size=32,
        )
        adj_image, _ = ring_adjoint(
            traces,
            acquisition,
            trace_wpt,
            domain,
            80,
            centre=centre,
            radius=radius,
            pairing="arc_length",
            eval_points=points,
            batch_size=32,
        )
        assert tr_image.dtype == adj_image.dtype == dtype
        np.testing.assert_allclose(tr_image, TIME_REVERSAL_IMAGE, rtol=rtol, atol=atol)
        np.testing.assert_allclose(adj_image, ADJOINT_IMAGE, rtol=rtol, atol=atol)
