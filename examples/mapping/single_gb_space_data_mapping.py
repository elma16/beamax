#!/usr/bin/env python3
r"""Visualise the space <-> data domain mapping for a single Gaussian beam.

This is the educational / paper figure script for the "how does a Gaussian beam
in the image domain map to detector data, and back?" experiment.  It mirrors the
local boundary analysis of Chapter 5inv (Lemma "Restriction of a Gaussian beam
to the boundary") and the MSGB time-reversal construction, and follows the
thesis-experiments figure conventions (see 6inv/TR_vis_roundtrip.py): viridis for
fields, RdBu_r for differences, panel labels (a)/(b)/(c), and a final difference
panel reporting ``max`` rather than a percentage.

Geometry (chosen for visualisation, *not* the real PAT boundary geometry):

    * 2D, homogeneous medium (c = const), non-periodic.
    * A detector LINE through the MIDDLE of the domain at x_1 = x1_det
      (axis 0 = normal coordinate x_1, axis 1 = tangential coordinate x_*).
    * A real pressure p_0 launched BELOW the detector at near-normal (17 deg
      oblique) incidence.  Its d'Alembert split has a forward mode p_0^+
      (+1, up, crossing Gamma) and a backward mode p_0^- (-1, down, leaving
      through the bottom).  The branch-level correspondence follows p_0^+;
      p_0^- is neither displayed nor reconstructed.  t_gamma is read off the
      solved forward beam-centre trajectory.

Two summary figures keep the two directions legible at thesis width:
    image -> data (3x2): p_0^+ | p^+(t_gamma) | g | coefficients | p_gamma |
                         p_gamma - g
    data -> image (2x2): g_K | recovered modes at t_gamma | p_{0,rec}^+ |
                         p_0^+ - p_{0,rec}^+

Individual panels (also saved to OUT_DIR):

    2d_01_p0.png             forward-going launch component p_0^+
    2d_02_pt.png             beam when its centre intersects the detector
    2d_03_g.png              recorded data g(x_*, t), whole signal captured
    2d_04_mswpt.png          log-magnitude MSWPT coefficients of the data
    2d_05_asymptotic.png         Lemma 5.1 asymptotic expansion at the detector (zoom)
    2d_06_asymptotic_minus_g.png asymptotic - g restriction error (zoom)
    2d_07_topk.png           data reconstructed from the top-K kept atoms
    2d_08_recovered.png      top-K data-GBs mapped into the image domain at t_gamma
    2d_09_p0_rec.png         back-propagated p_{0,rec}^+
    2d_10_diff.png           p_0^+ - p_{0,rec}^+

The Lemma 5.1 quadratic form M_tilde is exactly ``tr_solver_utils.mT_forward``.
The data->image map of the MAIN figures is the top-K BVP route
(``inverse="mswpt"``, the default): the ``Config.top_k`` most significant
data-frame atoms are each turned into a Gaussian beam by
``compute_TR_parameters`` (the Section 6 packet-beam matching), weighted by
their complex coefficients, and summed by ``compute_TR_result`` -- i.e. exactly
the production ``MSGBSolver`` time reversal restricted to its K dominant atoms.
There is no 1-to-1 beam<->frame-element mapping in this approximation.  Its
error combines top-K truncation, packet-to-beam matching, and amplitude
normalisation; redundancy by itself is not lossy.  The exact Hessian inverse
(``mT_inverse`` + ODE back-propagation, ``inverse="exact"``) is always computed
as a supplementary *shape* validation and saved as
``2d_13_p0_rec_exact.png`` / ``2d_14_diff_exact.png``.

On the reverse question -- redoing the asymptotic mapping from the data side -- see
``docs/guides/reverse-taylor-mapping.md``.

Run (from the beamax repo, public venv)::

    python examples/mapping/single_gb_space_data_mapping.py

The figures are written to ``outputs/`` next to this script.

Example smoke: false
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from beamax import geometry, plotter, utils
from beamax.decomposition import DyadicDecomposition
from beamax.gb import core, gb_solvers, gb_utils
from beamax.solvers.msgb_solvers import tr_solver_utils
from beamax.transforms import MSWPT

jax.config.update("jax_enable_x64", True)

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FIELD_CMAP = "viridis"  # fields / traces / packets
DIFF_CMAP = "RdBu_r"  # differences only

# Toggles for the spatial-panel annotations.
SHOW_CENTRE = True  # black dot at the beam centre x_gamma(t)
SHOW_ARROW = False  # orange momentum arrow on spatial panels

# Toggle: compute the "wavefield on the sensor" with a TRUE wave solve (k-Wave)
# instead of the Gaussian beam.  For a single beam the beam-approximation error
# is small, so this is optional -- it only reveals the space-asymptotic (beam) error
# in the "wavefield - asymptotic" panel.  Requires `pip install 'beamax[kwave]'`.
USE_KWAVE_FORWARD = False


def apply_thesis_style() -> None:
    """Replicate thesis_experiments.figures.apply_thesis_style rcParams."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 11.0,
            "mathtext.fontset": "dejavusans",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
        }
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """All tunable parameters for the experiment."""

    d: int = 2
    n: int = 512  # ~8 points / wavelength at omega0=40 (no aliasing)
    extent: float = 10.0
    c0: float = 1.0

    x1_det: float = 5.0  # detector at x_1 = x1_det (axis 0 normal)

    x0_normal: float = 2.5  # launch BELOW the detector, halfway to the
    x0_tangential: float = 3.0  # bottom of the domain (x_1 = 0)
    theta_deg: float = 17  # incidence angle to the detector NORMAL
    omega0: float = 100.0  # carrier resolved by the grid (ppw ~ 8)
    alpha0: float = 0.3  # ...and a wide waist => near-waist at Gamma
    a0: float = 1.0

    t_show: float = 1.5  # (unused) kept for backward compatibility
    t_max: float = 10.0  # crossing-time search horizon (data window = 2*t_gamma)
    nt: int = 256  # (unused) data Nt is now data_C * Nx (see below)

    num_levels: int = 3
    num_boxes_levels: tuple = (4, 8, 16)
    redundancy: int = 2
    windowing: str = "rectangular_mirror"
    # TR data domain (matching msgb): shape (data_C * Nx, Nx) with time-elongated
    # dyadic boxes of aspect (data_C, 1).  The time-frequency axis is the "long"
    # one (dispersion |tau| = c|k| >= c|k_*|), so the boxes stretch along time.
    data_C: int = 2

    top_k: int = 32  # data-frame atoms kept for the top-K TR recon
    # TR reconstruction batch size: the K beams are rendered in chunks of this
    # many (aggregate_method="scan"), so peak memory is O(batch_size * |grid|),
    # INDEPENDENT of top_k.  Lower this if you OOM; raise it for speed.
    batch_size: int = 1

    @property
    def dx(self) -> float:
        return self.extent / self.n


CFG = Config()


def sound_speed(x: jnp.ndarray) -> jnp.ndarray:
    """Homogeneous sound speed c(x) = c0."""
    return CFG.c0 + 0.0 * x[..., 0]


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of reference ``a`` not explained by the best scalar multiple of
    ``b``: min_alpha ||a - alpha b|| / ||a||.  Invariant to the scale of ``b``."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    alpha = float(np.dot(a, b) / (np.dot(b, b) + 1e-30))
    return float(np.linalg.norm(a - alpha * b) / (np.linalg.norm(a) + 1e-30))


# ---------------------------------------------------------------------------
# Forward beam set-up
# ---------------------------------------------------------------------------
def build_beam(cfg: Config):
    """Return launch parameters for the d'Alembert split (batch size 2).

    A single initial pressure ``p_0`` decomposes into two half-wave modes
    ``omega = +/- c|p|``.  Both beams share the launch point (at the detector,
    orthogonal incidence), waist, amplitude and frequency; they differ only in
    ``mode``: +1 propagates forward along +p_0, -1 backward along -p_0.
    """
    theta = np.deg2rad(cfg.theta_deg)
    p_one = [np.cos(theta), np.sin(theta)]  # |p| = 1, axis0 normal
    p_dir = jnp.array([p_one, p_one])  # (2, 2)
    x0 = jnp.array([[cfg.x0_normal, cfg.x0_tangential]] * 2)
    alpha0 = jnp.array([[cfg.alpha0, cfg.alpha0]] * 2) * 1j
    M0 = gb_utils.prepare_M0(alpha0, None)
    a0 = jnp.array([cfg.a0, cfg.a0])
    omega0 = jnp.array([cfg.omega0, cfg.omega0])
    mode = jnp.array([1.0, -1.0])  # forward / backward
    return x0, p_dir, M0, a0, omega0, mode


def crossing_time(x0, p0, cfg: Config, mode_sign: float = 1.0) -> float:
    """Analytic time at which the beam centre reaches x_1 = x1_det.

    Exact only for homogeneous c; kept as a fallback / sanity check for
    ``solver_crossing_time`` (which is the value actually used)."""
    p_hat = p0 / jnp.linalg.norm(p0, axis=-1, keepdims=True)
    v_normal = cfg.c0 * mode_sign * float(p_hat[0, 0])
    return float((cfg.x1_det - float(x0[0, 0])) / v_normal)


def solver_crossing_time(
    x0, p0, M0, a0, mode, cfg: Config, ode_solver, n_fine: int = 4000
) -> float:
    """Time the forward-mode beam CENTRE reaches x_1 = x1_det, read off the
    solved trajectory (sign-change + linear interpolation of the normal
    coordinate).  This is the robust, c-general way to get t_gamma -- the beam
    centre is wherever the gb solver puts it, not a closed-form guess."""
    ts = jnp.linspace(0.0, cfg.t_max, n_fine)
    xt, _, _, _ = ode_solver(
        x0[:1], p0[:1], M0[:1], a0[:1], mode[:1], ts, sound_speed, 0.0, None
    )
    x_normal = np.asarray(xt[0, :, 0])  # forward mode, normal coord
    ts_np = np.asarray(ts)
    s = np.sign(x_normal - cfg.x1_det)
    crossings = np.flatnonzero(s[:-1] * s[1:] < 0)
    if crossings.size == 0:  # never reaches the detector
        return crossing_time(x0, p0, cfg, mode_sign=float(mode[0]))
    i = int(crossings[0])
    x1, x2 = x_normal[i], x_normal[i + 1]
    t1, t2 = ts_np[i], ts_np[i + 1]
    return float(t1 + (cfg.x1_det - x1) * (t2 - t1) / (x2 - x1))


def beam_state_at(x0, p0, M0, a0, mode, t, ode_solver):
    """Evaluate the beam ray data (x, p, M, a) at a single time t."""
    ts = jnp.array([0.0, t])
    xt, pt, Mt, At = ode_solver(x0, p0, M0, a0, mode, ts, sound_speed, 0.0, None)
    return xt[:, -1], pt[:, -1], Mt[:, -1], At[:, -1, 0]


def boundary_trace_packet(xg, pg, Mg, ag, omega0, mode, t_gamma, t_grid, xstar_grid):
    """Analytic Lemma 5.1 packet p_gamma(t, 0, x_*) on the (t, x_*) grid.

    Phase = omega * ( L_gamma + Q_gamma ),
      L_gamma = p_*. dx_*  -  G dt           (linear)
      Q_gamma = 1/2 y^T M_tilde y,  y=(dt,dx_*) (quadratic)
    with M_tilde = mT_forward(xg, pg, Mg, mode, c)  [boundary Hessian].
    """
    M_tilde = tr_solver_utils.mT_forward(xg, pg, Mg, mode, sound_speed)[0]
    G_val = float(gb_utils.G(xg, pg, mode, sound_speed)[0])
    p_star = float(pg[0, 1])
    xstar_g = float(xg[0, 1])

    TT, XX = np.meshgrid(np.asarray(t_grid), np.asarray(xstar_grid), indexing="ij")
    dt = TT - t_gamma
    dxs = XX - xstar_g

    M = np.asarray(M_tilde)
    L = p_star * dxs - G_val * dt
    Q = 0.5 * (M[0, 0] * dt * dt + 2.0 * M[0, 1] * dt * dxs + M[1, 1] * dxs * dxs)
    # full complex amplitude; the real field is u + conj(u) = 2 Re(.), so the
    # caller takes 2 * Re(packet) to match compute_gaussian_beam_real.
    amp = complex(np.asarray(ag).ravel()[0])
    packet = amp * np.exp(1j * float(omega0[0]) * (L + Q))
    return packet, M_tilde, G_val, p_star


def scalar_match(ref: np.ndarray, x: np.ndarray) -> float:
    """alpha minimising ||ref - alpha x|| (used to normalise frame-convention
    amplitudes before differencing)."""
    ref = np.asarray(ref, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    return float(np.dot(ref, x) / (np.dot(x, x) + 1e-30))


def kwave_spatial_field(p0_field, domain, ts, t_gamma):
    """True spatial wavefield p(x_1, x_*) at t_gamma, via k-Wave (the toggle).

    Records the full grid and slices the time nearest ``t_gamma``.  Returns the
    (N1, N2) field, or ``None`` if k-Wave is unavailable.  NB: k-Wave's p0
    d'Alembert-splits 50/50; the caller scalar-matches to the GB beam.
    """
    try:
        from beamax.solvers.kwave_solver import KWaveSolver
    except Exception as exc:  # pragma: no cover - depends on optional extra
        print(
            f"[USE_KWAVE_FORWARD] k-Wave unavailable ({exc}); "
            f"falling back to the Gaussian beam."
        )
        return None
    solver = KWaveSolver()
    full_mask = np.ones(tuple(domain.N))
    rec = np.asarray(
        solver.forward(
            np.asarray(p0_field), domain, full_mask, np.asarray(ts), record="p"
        )
    )
    it = int(np.argmin(np.abs(np.asarray(ts) - t_gamma)))
    return rec[it].reshape(tuple(domain.N))


# ---------------------------------------------------------------------------
# Styled panel helpers (thesis-experiments conventions)
# ---------------------------------------------------------------------------
def _mark_centre(ax, centre, p_dir=None):
    """Optional black centre dot (+ orange momentum arrow) honouring the toggles."""
    if centre is not None and SHOW_CENTRE:
        ax.plot(centre[0], centre[1], "o", ms=6, mfc="k", mec="w", mew=0.8, zorder=5)
    if centre is not None and p_dir is not None and SHOW_ARROW:
        ax.quiver(
            centre[0],
            centre[1],
            p_dir[0],
            p_dir[1],
            color="tab:orange",
            angles="xy",
            scale=7,
            width=0.012,
            zorder=6,
        )


def _shared_norm(*fields):
    """Linear colour normalisation shared by directly comparable fields."""
    arrays = [np.asarray(field) for field in fields]
    vmin = float(min(np.min(field) for field in arrays))
    vmax = float(max(np.max(field) for field in arrays))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _add_colorbar(fig, ax, mappable):
    """Append a colourbar with exactly the same height as the image axes."""
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.06)
    return fig.colorbar(mappable, cax=cax)


def _spatial_panel(
    ax,
    fld,
    extent,
    title,
    *,
    centre=None,
    p_dir=None,
    zoom=None,
    norm=None,
    cfg=CFG,
):
    """viridis spatial field with detector line, beam centre, momentum arrow."""
    im = ax.imshow(
        np.asarray(fld),
        origin="lower",
        extent=extent,
        cmap=FIELD_CMAP,
        aspect="equal",
        norm=norm,
    )
    ax.axhline(cfg.x1_det, color="red", ls="--", lw=1.2)
    _mark_centre(ax, centre, p_dir)
    if zoom is not None:
        ax.set_xlim(*zoom[0])
        ax.set_ylim(*zoom[1])
    ax.set_xlabel(r"$x_*$")
    ax.set_ylabel(r"$x_1$")
    ax.set_title(title)
    return im


def _data_panel(ax, fld, extent, title, *, centre=None, zoom=None, norm=None):
    """viridis data-domain trace/packet."""
    im = ax.imshow(
        np.asarray(fld),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=FIELD_CMAP,
        norm=norm,
    )
    _mark_centre(ax, centre)
    if zoom is not None:
        ax.set_xlim(*zoom[0])
        ax.set_ylim(*zoom[1])
    ax.set_xlabel(r"$x_*$")
    ax.set_ylabel(r"$t$")
    ax.set_title(title)
    return im


def _diff_panel(ax, diff, extent, title, *, spatial, centre=None, zoom=None):
    """RdBu_r symmetric difference panel."""
    diff = np.asarray(diff)
    m = float(np.max(np.abs(diff))) or 1.0
    im = ax.imshow(
        diff,
        origin="lower",
        extent=extent,
        cmap=DIFF_CMAP,
        vmin=-m,
        vmax=m,
        aspect="equal" if spatial else "auto",
    )
    _mark_centre(ax, centre)
    if zoom is not None:
        ax.set_xlim(*zoom[0])
        ax.set_ylim(*zoom[1])
    ax.set_xlabel(r"$x_*$")
    ax.set_ylabel(r"$x_1$" if spatial else r"$t$")
    ax.set_title(title)
    return im


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(
    cfg: Config = CFG, save: bool = True, inverse: str = "mswpt"
) -> dict:
    """Run the full image->data->image pipeline for ``cfg``.

    ``inverse`` selects which data->image map the inverse summary uses:
    ``"mswpt"`` (the default; the dominant data-frame elements turned into
    beams by ``compute_TR_parameters`` -- the Section 6 packet-beam matching)
    or ``"exact"`` (Lemma 5.1 ``mT_inverse`` + ODE
    back-propagation).  BOTH routes are always computed and reported, so the
    exact roundtrip serves as a validation baseline either way.

    Returns a dict of relative-error metrics.  When ``save`` is True the two
    summary figures and the individual panels are also written to ``OUT_DIR``;
    the frequency sweep calls with ``save=False`` to skip plotting.
    """
    ode_solver = gb_solvers.solve_hom_general

    domain = geometry.Domain(
        N=(cfg.n, cfg.n),
        dx=(cfg.dx, cfg.dx),
        c=sound_speed,
        cfl=0.3,
        periodic=(False, False),
    )
    domain_size = domain.grid_size
    periodic = jnp.array([False, False])
    grid = domain.grid

    ndet = int(round(cfg.x1_det / cfg.dx))
    sensor_mask = jnp.zeros((cfg.n, cfg.n)).at[ndet, :].set(1.0)
    sources = geometry.Sensor(domain=domain, binary_mask=sensor_mask)
    det_positions = sources.positions
    xstar_axis = np.asarray(det_positions[:, 1])

    zoom_w = 1.2

    # --- forward beam: the two modes ----------------------------------------
    x0, p0, M0, a0, omega0, mode = build_beam(cfg)
    # Launch is BELOW the detector: the forward mode (+1) travels up and crosses
    # Gamma at t_gamma>0, while the backward mode (-1) radiates down and out the
    # bottom -- so only the forward mode is detected.  Record a one-sided
    # [0, t_max] window with the crossing comfortably inside it.
    t_gamma = solver_crossing_time(x0, p0, M0, a0, mode, cfg, ode_solver)
    # TR data domain (msgb convention): Nt = data_C * Nx, time-elongated.  The
    # window is centred on the crossing so different incidence angles compare
    # fairly rather than being clipped.
    nx_data = int(det_positions.shape[0])
    nt_data = cfg.data_C * nx_data
    ts_data = jnp.linspace(0.0, 2.0 * t_gamma, nt_data)
    ts_np = np.asarray(ts_data)

    xg, pg, Mg, ag = beam_state_at(x0, p0, M0, a0, mode, t_gamma, ode_solver)
    xstar_gamma = float(xg[0, 1])

    def field_at(times, x0_, p0_, M0_, a0_, w_, mode_, solver):
        return np.asarray(
            core.compute_gaussian_beam_real(
                x0_,
                p0_,
                M0_,
                a0_,
                w_,
                mode_,
                sound_speed,
                0.0,
                jnp.asarray(times),
                grid,
                domain_size,
                periodic,
                solver,
                None,
            )[0]
        )

    # The internal plane sees only the + branch.  Every displayed reference,
    # trace, reconstruction, and error therefore uses that same branch.
    field_init_plus = field_at(
        [0.0], x0[:1], p0[:1], M0[:1], a0[:1], omega0[:1], mode[:1], ode_solver
    )
    # snapshot at t_gamma: the forward-mode CENTRE is exactly on the detector
    field_gamma_plus = field_at(
        [t_gamma],
        x0[:1],
        p0[:1],
        M0[:1],
        a0[:1],
        omega0[:1],
        mode[:1],
        ode_solver,
    )
    data = np.asarray(
        core.compute_gaussian_beam_real(
            x0[:1],
            p0[:1],
            M0[:1],
            a0[:1],
            omega0[:1],
            mode[:1],
            sound_speed,
            0.0,
            ts_data,
            det_positions,
            domain_size,
            periodic,
            ode_solver,
            None,
        )
    )

    # whole-signal check: the data window must contain the full transit, i.e.
    # the trace must be ~0 in the first/last 5% of the recording window.
    k_edge = max(1, nt_data // 20)
    edge_frac = float(
        np.linalg.norm(np.concatenate([data[:k_edge].ravel(), data[-k_edge:].ravel()]))
        / (np.linalg.norm(data) + 1e-30)
    )
    if edge_frac > 1e-6:
        print(
            f"WARNING: signal not fully recorded -- edge energy fraction "
            f"{edge_frac:.2e} > 1e-6; enlarge the data window (t_max/data_C)."
        )

    if save:
        print("=" * 70)
        print(
            f"single GB d'Alembert split: launch x0={np.asarray(x0)[0]}, "
            f"p0={np.asarray(p0)[0]}, omega0={float(omega0[0])}"
        )
        print(
            f"detector x1_det={cfg.x1_det} (row {ndet}); forward mode crosses "
            f"t_gamma={t_gamma:.4f} at x_*={xstar_gamma:.4f}"
        )

    # asymptotic packet: analytic boundary trace.  ONLY the forward mode (index 0)
    # crosses the detector -- the backward mode never reaches Gamma, so adding
    # its boundary-trace packet would plant a spurious blob (badly so at oblique
    # incidence, where the two modes separate tangentially).
    asymptotic, M_tilde, G_val, p_star = boundary_trace_packet(
        xg[:1],
        pg[:1],
        Mg[:1],
        ag[:1],
        omega0[:1],
        mode[:1],
        t_gamma,
        ts_data,
        det_positions[:, 1],
    )
    asymptotic_real = 2.0 * np.real(asymptotic)  # u + conj(u)

    # Berra (2017)'s two asymptotic expansions, in their two domains:
    #  * time-asymptotic phi_gamma  -- restricted in SPACE (x_1=det), expanded in TIME.
    #    Its error vs the data g(x_*,t) is the DATA-domain panel  asymptotic - data.
    #  * space-asymptotic (the beam) -- restricted in TIME (t=t_gamma), expanded in
    #    SPACE.  The "wavefield" lives in the IMAGE domain: p(x_1,x_*) at t_gamma,
    #    zoomed onto the detector crossing.  By default it is the GB beam; with
    #    USE_KWAVE_FORWARD it is the TRUE wave solution and its departure from the
    #    smooth beam is the (small) space-asymptotic / beam-approximation error.
    diff_asymptotic_data = asymptotic_real - data  # time-asymptotic restriction error
    wavefield_spatial = field_gamma_plus  # image-domain p^+(x_1,x_*) at t_gamma
    if USE_KWAVE_FORWARD and save:
        kw = kwave_spatial_field(field_init_plus, domain, ts_data, t_gamma)
        if kw is not None:
            wavefield_spatial = scalar_match(field_gamma_plus, kw) * kw
    diff_wave_asymptotic = (
        wavefield_spatial - field_gamma_plus
    )  # space-asymptotic / beam error

    # data-domain MSWPT.  Shape (data_C * Nx, Nx) with time-elongated boxes of
    # aspect (data_C, 1) -- the msgb TR convention (boxes stretched along the
    # long, time-frequency axis), NOT square or space-elongated.
    data_shape = (nt_data, nx_data)
    data_decomp = DyadicDecomposition(
        cfg.num_levels, data_shape, cfg.num_boxes_levels, (cfg.data_C, 1)
    )
    data_wpt = MSWPT(data_decomp, cfg.redundancy, cfg.windowing)

    coeffs = data_wpt.forward(jnp.asarray(data), "spatial")
    coeffs_np = np.asarray(coeffs)
    coeffs_array = np.asarray(data_wpt.convert_to_array(coeffs))
    # top-K data-frame atoms (largest |coeff|), mirroring the production TR
    # selector ``forward_solver_utils._threshold_top_n``: keep the indices AND
    # their COMPLEX values -- the relative magnitudes/phases of the K atoms are
    # exactly what the superposition needs, and a single global scalar cannot
    # supply them.
    K = int(cfg.top_k)
    topk_idx = [int(i) for i in np.argsort(np.abs(coeffs_np))[::-1][:K]]
    topk_vals = coeffs_np[np.array(topk_idx)]  # (K,) complex

    # Inverse-summary panel (a): reconstruct the data from ONLY the K kept atoms
    # (inverse-transform of the truncated coefficient vector) -- the top-K
    # generalisation of the old single "dominant component".
    single = jnp.zeros_like(coeffs)
    for i in topk_idx:
        single = single.at[i].set(coeffs[i])
    topk_field = np.real(np.asarray(data_wpt.inverse(single, "spatial")))
    # This is the best-global-scale squared-correlation score.  It is not the
    # fraction of signal or frame-coefficient energy retained.
    topk_shape_score = 1.0 - relative_error(data, topk_field) ** 2
    topk_corr = float(
        np.dot(data.ravel(), topk_field.ravel())
        / (np.linalg.norm(data) * np.linalg.norm(topk_field) + 1e-30)
    )

    # ===== data -> image: the REAL beamax time reversal, top-K ===============
    # This follows MSGBSolver._prepare_tr_params / compute_TR_result exactly:
    #   1. compute_TR_parameters turns each kept atom into a beam (one per
    #      coefficient), carrying its own per-beam emission time ts[:, 0];
    #   2. each beam's geometric amplitude is weighted by 0.5 * its complex
    #      coefficient (the c_pos = 0.5 * forward d'Alembert half);
    #   3. compute_TR_result back-propagates every beam over its own ts to t=0
    #      and SUMS them; the field is doubled (full-field TR convention).
    # This is an approximate top-K packet-to-beam map.  A complete dual-frame
    # reconstruction is exact; the approximation here also changes packets to
    # beams and fits a global output scale.
    dt = float(ts_data[1] - ts_data[0])
    data_domain = geometry.Domain(
        N=data_shape, dx=(dt, cfg.dx), c=sound_speed, cfl=0.3, periodic=(False, False)
    )
    (
        pts,
        Mts,
        xts,
        omegas_tr,
        ats,
        signum,
        ts_tr,
    ) = tr_solver_utils.compute_TR_parameters(
        jnp.array(topk_idx), data_domain, data_wpt, sources
    )
    cpos_vals = 0.5 * jnp.asarray(topk_vals)  # c_pos half-weights
    a_w_1d = ats[:, 0] * cpos_vals  # (K,)    snapshot
    a_w_2d = ats * cpos_vals[:, None]  # (K, 1)  compute_TR_result
    # recovered modes ON Gamma: the K weighted beams at solver time 0 (== the
    # crossing for every beam), summed by compute_gaussian_beam_real, doubled.
    back_gamma_mswpt = 2.0 * field_at(
        [0.0],
        xts,
        pts,
        Mts,
        a_w_1d,
        omegas_tr,
        signum[:, 0],
        gb_solvers.solve_hom_general,
    )
    # p_{0,rec}^+: per-beam back-propagation to t=0 + sum, via compute_TR_result
    # (solve_ODE_batch_t handles each atom's variable emission time ts[:, 0]).
    # BATCHING (memory): with aggregate_method="all" the renderer materialises
    # the whole beam axis at once -- compute_gaussian_beam_real_TR returns
    # (Nt, *grid, K) -- so peak memory grows linearly in K and OOMs for large K.
    # Instead reshape the beams into (num_batches, batch_size, ...) with
    # utils.batch_data and use aggregate_method="scan": the scan accumulates one
    # batch at a time, so peak memory is O(batch_size * |grid|), INDEPENDENT of
    # K.  This is exactly MSGBSolver._prepare_tr_params + time_reversal; ats
    # (index 4) is zero-padded in the final batch so padding beams add nothing.
    tr_params = utils.batch_data(
        pts,
        Mts,
        xts,
        omegas_tr,
        a_w_2d,
        signum,
        ts_tr,
        batch_size=cfg.batch_size,
        zero_padded_args=(4,),
    )
    back_init_mswpt = 2.0 * np.asarray(
        tr_solver_utils.compute_TR_result(
            tr_params,
            sound_speed,
            0.0,
            grid,
            domain_size,
            periodic,
            ode_solver=gb_solvers.solve_ODE_batch_t,
            aggregate_method="scan",
        )
    )
    # (b) EXACT data->image (validation baseline): M_tilde = mT_forward(beam at
    # Gamma) is the data-domain boundary Hessian; mT_inverse recovers the spatial
    # Hessian (exact inverses); then the ORIGINAL forward-mode beam is propagated
    # from the crossing back to t=0 -- no quantisation.
    xg_f, pg_f, Mg_f, ag_f = xg[:1], pg[:1], Mg[:1], ag[:1]
    omega_f, mode_f = omega0[:1], mode[:1]
    M_tilde_f = tr_solver_utils.mT_forward(xg_f, pg_f, Mg_f, mode_f, sound_speed)
    M_rec = tr_solver_utils.mT_inverse(xg_f, pg_f, M_tilde_f, mode_f, sound_speed)
    back_init_exact = field_at(
        [-t_gamma], xg_f, pg_f, M_rec, ag_f, omega_f, mode_f, ode_solver
    )
    back_gamma_exact = field_at(
        [0.0], xg_f, pg_f, M_rec, ag_f, omega_f, mode_f, ode_solver
    )
    # amplitudes carry a frame / TR normalisation; rescale each field by one
    # global scalar to the forward mode for an honest shape comparison (the
    # relative weighting of the K atoms already comes from their coefficients).
    alpha_init = scalar_match(field_init_plus, back_init_mswpt)
    back_init_mswpt = alpha_init * back_init_mswpt
    back_gamma_mswpt = (
        scalar_match(field_gamma_plus, back_gamma_mswpt) * back_gamma_mswpt
    )
    back_init_exact = scalar_match(field_init_plus, back_init_exact) * back_init_exact
    back_gamma_exact = (
        scalar_match(field_gamma_plus, back_gamma_exact) * back_gamma_exact
    )
    if inverse == "mswpt":
        field_back_init, field_back_gamma = back_init_mswpt, back_gamma_mswpt
    else:
        field_back_init, field_back_gamma = back_init_exact, back_gamma_exact

    # geometry helpers for markers
    extent_xy = [0.0, float(domain_size[1]), 0.0, float(domain_size[0])]
    data_ext = [xstar_axis[0], xstar_axis[-1], ts_np[0], ts_np[-1]]
    zoom = (
        (xstar_gamma - zoom_w, xstar_gamma + zoom_w),
        (t_gamma - zoom_w, t_gamma + zoom_w),
    )
    p_plot = (float(p0[0, 1]), float(p0[0, 0]))  # (x_*, x_1) components
    launch_xy = (cfg.x0_tangential, cfg.x0_normal)
    # Local spatial views: a path-wide window makes this high-frequency beam
    # nearly invisible at thesis width.  The adjacent launch/crossing panels
    # already communicate the propagation, so centre each view on the field it
    # is intended to show.  Use the same windows again in the inverse figure so
    # the references and reconstructions remain visually comparable.
    zoom_launch = (
        (cfg.x0_tangential - zoom_w, cfg.x0_tangential + zoom_w),
        (cfg.x0_normal - zoom_w, cfg.x0_normal + zoom_w),
    )
    zoom_crossing = (
        (xstar_gamma - zoom_w, xstar_gamma + zoom_w),
        (cfg.x1_det - zoom_w, cfg.x1_det + zoom_w),
    )

    # Directly comparable field panels share colour limits, including across
    # the two summary figures.
    data_norm = _shared_norm(data, asymptotic_real, topk_field)
    launch_norm = _shared_norm(field_init_plus, field_back_init)
    crossing_norm = _shared_norm(field_gamma_plus, field_back_gamma)

    # ===== panel definitions (shared by summary and individual saves) =======
    def _mswpt_pf(ax):
        im = plotter.plot_mswpt_coeffs(ax, coeffs_array, data_decomp, log_scale=True)
        ax.set_title(r"$\log |c_{\ell,j,k}|$")
        ax.set_xlabel(r"$\xi_*$")
        ax.set_ylabel(r"$\tau$")
        return im

    forward_panels = [
        # row 1: forward-going image branch -> data
        (
            lambda ax: _spatial_panel(
                ax,
                field_init_plus,
                extent_xy,
                r"$p_0^+$",
                centre=launch_xy,
                p_dir=p_plot,
                zoom=zoom_launch,
                norm=launch_norm,
            ),
            (0, 0),
            "2d_01_p0.png",
        ),
        (
            lambda ax: _spatial_panel(
                ax,
                field_gamma_plus,
                extent_xy,
                r"$p^+(t_\gamma)$",
                centre=(xstar_gamma, cfg.x1_det),
                p_dir=p_plot,
                zoom=zoom_crossing,
                norm=crossing_norm,
            ),
            (0, 1),
            "2d_02_pt.png",
        ),
        (
            lambda ax: _data_panel(
                ax,
                data,
                data_ext,
                r"$g(x_*,t)$",
                centre=(xstar_gamma, t_gamma),
                zoom=zoom,
                norm=data_norm,
            ),
            (1, 0),
            "2d_03_g.png",
        ),
        # row 2: data-domain analysis, zoomed on the crossing
        (_mswpt_pf, (1, 1), "2d_04_mswpt.png"),
        (
            lambda ax: _data_panel(
                ax,
                asymptotic_real,
                data_ext,
                r"$p_\gamma(t,x_*)$",
                centre=(xstar_gamma, t_gamma),
                zoom=zoom,
                norm=data_norm,
            ),
            (2, 0),
            "2d_05_asymptotic.png",
        ),
        (
            lambda ax: _diff_panel(
                ax,
                diff_asymptotic_data,
                data_ext,
                r"$p_\gamma-g$",
                spatial=False,
                centre=(xstar_gamma, t_gamma),
                zoom=zoom,
            ),
            (2, 1),
            "2d_06_asymptotic_minus_g.png",
        ),
    ]

    inverse_panels = [
        # top-K data -> forward-going image branch
        (
            lambda ax: _data_panel(
                ax,
                topk_field,
                data_ext,
                rf"$g_K$ ($K={K}$)",
                centre=(xstar_gamma, t_gamma),
                zoom=zoom,
                norm=data_norm,
            ),
            (0, 0),
            "2d_07_topk.png",
        ),
        (
            lambda ax: _spatial_panel(
                ax,
                field_back_gamma,
                extent_xy,
                r"$p_K^+(t_\gamma)$",
                centre=(xstar_gamma, cfg.x1_det),
                zoom=zoom_crossing,
                norm=crossing_norm,
            ),
            (0, 1),
            "2d_08_recovered.png",
        ),
        (
            lambda ax: _spatial_panel(
                ax,
                field_back_init,
                extent_xy,
                r"$p_{0,\mathrm{rec}}^+$",
                centre=launch_xy,
                zoom=zoom_launch,
                norm=launch_norm,
            ),
            (1, 0),
            "2d_09_p0_rec.png",
        ),
        (
            lambda ax: _diff_panel(
                ax,
                field_init_plus - field_back_init,
                extent_xy,
                r"$p_0^+-p_{0,\mathrm{rec}}^+$",
                spatial=True,
                centre=launch_xy,
                zoom=zoom_launch,
            ),
            (1, 1),
            "2d_10_diff.png",
        ),
    ]
    panels = forward_panels + inverse_panels

    # ----- relative-error metrics (returned for the frequency sweep) --------
    # relative_error(reference, candidate): normalise by the fixed reference so a
    # near-zero (under-resolved) candidate scores ~1.0 rather than a false 0.
    errors = {
        "omega0": float(cfg.omega0),
        "asymptotic_vs_data": relative_error(data, asymptotic_real),
        "space_asymptotic_err": relative_error(field_gamma_plus, wavefield_spatial),
        "topk_vs_trace": relative_error(data, topk_field),
        "topk_shape_score": topk_shape_score,
        "topk_corr": topk_corr,
        "recon_vs_p0": relative_error(field_init_plus, field_back_init),
        "recon_vs_p0_mswpt": relative_error(field_init_plus, back_init_mswpt),
        "recon_vs_p0_exact": relative_error(field_init_plus, back_init_exact),
    }

    if not save:
        return errors

    # ===== Berra's two asymptotic expansions, each in its own domain =============
    src = (
        "k-Wave"
        if (USE_KWAVE_FORWARD and wavefield_spatial is not field_gamma_plus)
        else "GB beam"
    )
    cmark = (xstar_gamma, t_gamma)  # data-domain marker (x_*, t)
    cmark_s = (xstar_gamma, cfg.x1_det)  # image-domain marker (x_*, x_1)
    zw = zoom_w
    zoom_s = ((xstar_gamma - zw, xstar_gamma + zw), (cfg.x1_det - zw, cfg.x1_det + zw))

    def _zoomed_spatial(ax, fld, title):
        im = _spatial_panel(ax, fld, extent_xy, title, centre=cmark_s)
        ax.set_xlim(*zoom_s[0])
        ax.set_ylim(*zoom_s[1])
        return im

    figc, axc = plt.subplots(2, 3, figsize=(15.5, 9.5), constrained_layout=True)

    def cbarc(im, ax):
        figc.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # SPACE-asymptotic (image domain): the wavefield p(x_1,x_*) at t_gamma zoomed on
    # the detector crossing -- this is 2d_02_pt zoomed in, per Berra Type B.
    cbarc(
        _zoomed_spatial(
            axc[0, 0], wavefield_spatial, f"wavefield at $\\Gamma$ ({src}, zoom)"
        ),
        axc[0, 0],
    )
    cbarc(
        _diff_panel(
            axc[1, 0],
            diff_wave_asymptotic,
            extent_xy,
            r"wavefield $-$ space-asymptotic (beam err)",
            spatial=True,
            centre=cmark_s,
            zoom=zoom_s,
        ),
        axc[1, 0],
    )
    # TIME-asymptotic (data domain): the packet phi_gamma vs the recorded data.
    cbarc(
        _data_panel(
            axc[0, 1],
            asymptotic_real,
            data_ext,
            r"time-asymptotic $\phi_\gamma$",
            centre=cmark,
        ),
        axc[0, 1],
    )
    cbarc(
        _diff_panel(
            axc[1, 1],
            diff_asymptotic_data,
            data_ext,
            r"asymptotic $-$ data (restriction err)",
            spatial=False,
            centre=cmark,
        ),
        axc[1, 1],
    )
    # TOP-K reconstruction vs the asymptotic packet: the K kept atoms reconstruct the
    # data trace; differencing against the smooth asymptotic packet shows the
    # frame-quantisation departure (scalar-matched -- shape is what counts).
    topk_matched = scalar_match(asymptotic_real, topk_field) * topk_field
    diff_topk_asymptotic = topk_matched - asymptotic_real
    cbarc(
        _data_panel(
            axc[0, 2],
            topk_matched,
            data_ext,
            rf"top-$K$ atoms ($K={K}$)",
            centre=cmark,
            zoom=zoom,
        ),
        axc[0, 2],
    )
    cbarc(
        _diff_panel(
            axc[1, 2],
            diff_topk_asymptotic,
            data_ext,
            r"top-$K\ -$ asymptotic (quantisation err)",
            spatial=False,
            centre=cmark,
            zoom=zoom,
        ),
        axc[1, 2],
    )
    figc.savefig(OUT_DIR / "single_gb_asymptotic_comparison.png")
    plt.close(figc)

    # individual panel: the top-K vs asymptotic difference
    f, a = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    f.colorbar(
        _diff_panel(
            a,
            diff_topk_asymptotic,
            data_ext,
            r"top-$K\ -$ asymptotic",
            spatial=False,
            centre=cmark,
            zoom=zoom,
        ),
        ax=a,
        fraction=0.046,
        pad=0.04,
    )
    f.savefig(OUT_DIR / "2d_12_diff_topk_asymptotic.png")
    plt.close(f)

    # supplementary validation: the EXACT Lemma 5.1 inverse roundtrip
    f, a = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    f.colorbar(
        _spatial_panel(
            a,
            back_init_exact,
            extent_xy,
            r"$p_{0,\mathrm{rec}}^+$ (exact $\widetilde{M}$ inverse)",
            centre=launch_xy,
            zoom=zoom_launch,
        ),
        ax=a,
        fraction=0.046,
        pad=0.04,
    )
    f.savefig(OUT_DIR / "2d_13_p0_rec_exact.png")
    plt.close(f)
    f, a = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    f.colorbar(
        _diff_panel(
            a,
            field_init_plus - back_init_exact,
            extent_xy,
            r"$p_0^+ - p_{0,\mathrm{rec}}^+$ (exact)",
            spatial=True,
            centre=launch_xy,
            zoom=zoom_launch,
        ),
        ax=a,
        fraction=0.046,
        pad=0.04,
    )
    f.savefig(OUT_DIR / "2d_14_diff_exact.png")
    plt.close(f)

    # supplementary individual panels (the summaries own 2d_01..2d_10):
    #   2d_15 image-domain wavefield at Gamma (zoomed); 2d_16 the FULL-window
    #   asymptotic - g restriction error (the zoomed twin is summary panel 2d_06).
    f, a = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    f.colorbar(
        _zoomed_spatial(a, wavefield_spatial, rf"wavefield at $\Gamma$ ({src}, zoom)"),
        ax=a,
        fraction=0.046,
        pad=0.04,
    )
    f.savefig(OUT_DIR / "2d_15_wavefield_at_detector.png")
    plt.close(f)
    f, a = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
    f.colorbar(
        _diff_panel(
            a,
            diff_asymptotic_data,
            data_ext,
            r"asymptotic $-\ g$ (full)",
            spatial=False,
            centre=cmark,
        ),
        ax=a,
        fraction=0.046,
        pad=0.04,
    )
    f.savefig(OUT_DIR / "2d_16_asymptotic_minus_g_full.png")
    plt.close(f)

    # ===== thesis-width summary figures + individual panels =============
    def _save_summary(panel_specs, shape, figsize, filename):
        fig, axs = plt.subplots(*shape, figsize=figsize, constrained_layout=True)
        fig.set_constrained_layout_pads(
            w_pad=0.16,
            h_pad=0.08,
            wspace=0.32,
            hspace=0.08,
        )
        axs = np.asarray(axs).reshape(shape)
        colorbar_pairs = []
        for idx, (pf, (i, j), _) in enumerate(panel_specs):
            colorbar = _add_colorbar(fig, axs[i, j], pf(axs[i, j]))
            colorbar_pairs.append((axs[i, j], colorbar.ax))
            axs[i, j].text(
                0.035,
                0.965,
                f"({chr(97 + idx)})",
                transform=axs[i, j].transAxes,
                va="top",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.12",
                    fc="white",
                    ec="none",
                    alpha=0.75,
                ),
            )
        fig.canvas.draw()
        for ax, cax in colorbar_pairs:
            if not np.isclose(
                ax.get_position().height,
                cax.get_position().height,
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError("mapping-panel colourbar height mismatch")
        fig.savefig(OUT_DIR / filename)
        plt.close(fig)

    _save_summary(
        forward_panels,
        (3, 2),
        (5.45, 7.15),
        "single_gb_image_to_data.png",
    )
    _save_summary(
        inverse_panels,
        (2, 2),
        (5.45, 5.3),
        "single_gb_data_to_image.png",
    )

    for pf, _, fname in panels:
        f, a = plt.subplots(figsize=(4.8, 4.4), constrained_layout=True)
        _add_colorbar(f, a, pf(a))
        f.savefig(OUT_DIR / fname)
        plt.close(f)

    # ----- console summary (errors, not percentages in figures) -------------
    print("-" * 70)
    print("difference maxima (RdBu_r panels):")
    print(
        f"  wavefield - space-asymptotic  : {np.max(np.abs(diff_wave_asymptotic)):.3e}"
        f"  ({src}; beam err, ~0 unless k-Wave)"
    )
    print(
        f"  asymptotic - data             : {np.max(np.abs(diff_asymptotic_data)):.3e}"
        f"  (time-asymptotic restriction err)"
    )
    print(
        f"  top-K (matched) - asymptotic  : {np.max(np.abs(diff_topk_asymptotic)):.3e}"
    )
    print(
        f"  reconstruction - p_0^+    : {np.max(np.abs(field_back_init - field_init_plus)):.3e}"
    )
    print("relative L2 errors (best-scalar-matched):")
    print(f"  asymptotic trace vs data      : {errors['asymptotic_vs_data']:.3e}")
    print(f"  top-K atoms vs data trace : {errors['topk_vs_trace']:.3e}")
    print(f"top-K data-frame reconstruction (K={K}, redundant frame):")
    print(f"  best-scale shape score    : {100.0 * topk_shape_score:.1f} %")
    print(f"  correlation with trace    : {topk_corr:.3f}")
    print(f"  TR amplitude match alpha  : {alpha_init:.3f}  (frame normalisation)")
    print("data->image reconstructions vs p_0^+ (relative L2):")
    print(
        f"  top-K TR (compute_TR_result): {errors['recon_vs_p0_mswpt']:.3e}  (best-scale shape error)"
    )
    print(
        f"  exact mT_inverse + ODE      : {errors['recon_vs_p0_exact']:.3e}  (best-scale shape check)"
    )
    print(
        f"crossing x_g={np.asarray(xg)[0].tolist()} at t_gamma={t_gamma:.4f}; "
        f"main figures use inverse = {inverse}; "
        f"data window edge energy = {edge_frac:.1e}"
    )
    print("=" * 70)
    print(f"Figures written to {OUT_DIR}")
    print("=" * 70)
    return errors


def main() -> None:
    apply_thesis_style()
    run_experiment(CFG, save=True)


if __name__ == "__main__":
    main()
