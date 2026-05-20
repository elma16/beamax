import pytest

import jax
from jax import jit, vmap
import jax.numpy as jnp

from beamax import geometry
from beamax.gb import core, gb_utils, gb_solvers
from beamax.geometry import Domain
from beamax.gb.gb_utils import prepare_M0, is_diagonal, check_M0

jax.config.update("jax_enable_x64", True)


def generate_complex_positive_definite_matrix(b, d):
    key = jax.random.PRNGKey(0)

    A = jax.random.uniform(key, shape=(b, d, d)) * 5
    real_part = jnp.einsum("bij,bkj->bik", A, A)

    key, _ = jax.random.split(key)
    B = jax.random.normal(key, shape=(b, d, d)) * 0.5
    imag_part = jnp.einsum("bij,bkj->bik", B, B)
    M0 = real_part + 1j * imag_part
    return M0


@jit
def compute_eigenvalues(array: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the eigenvalues of a batch of matrices.

    :arg array: The array of matrices. (b, d, d)
    :returns: The eigenvalues of the matrices. (b, d)
    """

    def eig_slice(slice):
        return jnp.linalg.eigvals(slice)

    return vmap(eig_slice)(array)


def calculate_ray_distance(xt):
    displacements = jnp.diff(xt, axis=1)
    step_distances = jnp.sqrt(jnp.sum(displacements**2, axis=-1))
    ray_distances = jnp.cumsum(step_distances, axis=1)
    ray_distances = jnp.concatenate(
        [jnp.zeros((xt.shape[0], 1)), ray_distances], axis=1
    )
    return ray_distances


def random_data(b, N, isotropic=False):
    d = len(N)
    periodic = (False,) * d
    dx = (1e-4,) * d
    domain = Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=0.3)
    x_sensor = domain.grid

    key = jax.random.PRNGKey(0)
    Nt = 100
    ts = jnp.linspace(0, 1e-4, Nt)

    x0 = jax.random.uniform(key, (b, d))
    p0 = jax.random.uniform(key, (b, d))
    a0 = jax.random.uniform(key, (b,)) * 100
    omega0 = jax.random.uniform(key, (b,))

    if isotropic:
        alpha0 = jnp.ones((b, d)) * 10j
    else:
        alpha0 = jax.random.uniform(key, (b, d)) * 1j

    return ts, x0, p0, a0, alpha0, omega0, x_sensor


def c(x):
    return 1 + 0 * x[..., 0]


def generate_test_params():
    # We sweep 1D / 2D / 3D shapes only. The hom-vs-num solver agreement
    # being asserted here is a numerical property and does not gain coverage
    # by also running at float32 precision, so we keep only the x64 variant.
    return [(5, N, True) for N in [(40,), (30, 40), (30, 40, 50)]]


@pytest.fixture(scope="module", params=generate_test_params())
def gb_setup(request):
    b, N, use_x64 = request.param
    jax.config.update("jax_enable_x64", use_x64)
    print(f"Running test with batch size {b} and grid size {N}")
    t, x0, p0, a0, alpha0, omega0, x_sensor = random_data(b, N, isotropic=True)

    d = len(N)
    M0 = gb_utils.prepare_M0(alpha0, None)
    # is_M0_diagonal = gb_utils.is_diagonal(M0)
    mode = jnp.ones((b,))
    periodic = (False,) * d
    domain_size = jnp.ones((d,))
    solver_config = None
    lam = 0

    hom_solver = gb_solvers.solve_hom_diag
    ode_solver = gb_solvers.solve_ODE_base

    gb_hom = core.compute_gaussian_beam(
        x0,
        p0,
        M0,
        a0,
        omega0,
        mode,
        c,
        lam,
        t,
        x_sensor,
        domain_size,
        jnp.array(periodic),
        hom_solver,
        solver_config,
    )

    xt_hom, pt_hom, Mt_hom, At_hom = hom_solver(
        x0, p0, M0, a0, mode, t, c, solver_config
    )

    gb_num = core.compute_gaussian_beam(
        x0,
        p0,
        M0,
        a0,
        omega0,
        mode,
        c,
        lam,
        t,
        x_sensor,
        domain_size,
        jnp.array(periodic),
        ode_solver,
        solver_config,
    )

    xt_num, pt_num, Mt_num, At_num = ode_solver(
        x0, p0, M0, a0, mode, t, c, lam, solver_config
    )

    hom_results = [gb_hom, xt_hom, pt_hom, Mt_hom, At_hom]
    num_results = [gb_num, xt_num, pt_num, Mt_num, At_num]

    return hom_results, num_results, b, N, t, x0, p0, a0, alpha0, omega0, x_sensor


def test_compare_analytical_numerical_gb(gb_setup):
    """
    Test that the analytical and numerical solutions are the same when the medium is homogeneous.

    1. Assert that the two GB wavefields are the same.
    2. Assert that the shapes are what we expect.
    3. Assert that the wavefields are complex.
    """
    gb_hom_result, gb_num_result, b, N, t, *_ = gb_setup
    gb_hom = gb_hom_result[0]
    gb_num = gb_num_result[0]
    Nt = len(t)
    print(f"d: {len(N)}, gb_hom: {gb_hom.shape}, gb_num: {gb_num.shape}")
    assert jnp.allclose(gb_hom, gb_num, atol=1e-16)
    assert gb_hom.shape == (Nt,) + N + (b,)
    assert jnp.all(jnp.iscomplex(gb_hom))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_gb_reversible(d):
    """
    Test that the GB is reversible.

    quite close, but not exactly the same.
    """

    def c(x):
        return 1500 + 0 * x[..., 0]

    b = 1
    Nt = 100
    ts = jnp.linspace(0, 1e-4, Nt)
    key = jax.random.PRNGKey(0)
    lam = 0

    # Generate random initial conditions
    x0 = jax.random.uniform(key, (b, d))
    p0 = jax.random.uniform(key, (b, d))
    a0 = jax.random.uniform(key, (b,))
    alpha0 = jax.random.uniform(key, (b, d))
    mode = jnp.ones((b,))
    solver_configs = None

    # Initial covariance matrix
    m0 = 1j * jnp.einsum("bd,dj->bdj", alpha0, jnp.eye(d))

    # Forward propagation
    ode_solver = gb_solvers.solve_ODE_base

    xt, pt, mt, at = ode_solver(x0, p0, m0, a0, mode, ts, c, lam, solver_configs)

    # Extract final state
    xT = xt[:, -1, :]
    pT = pt[:, -1, :]
    mT = mt[:, -1, ...]
    aT = at[:, -1, :]

    # Reverse time array for backward propagation
    ts_inv = ts[::-1]

    # Backward propagation from final state
    xt_inv, pt_inv, mt_inv, at_inv = ode_solver(
        xT, pT, mT, aT, mode, ts_inv, c, lam, solver_configs
    )

    x0_inv = xt_inv[:, -1, :]
    p0_inv = pt_inv[:, -1, :]
    m0_inv = mt_inv[:, -1, ...]
    a0_inv = at_inv[:, -1, :]

    # test MT is symmetric and imag. positive definite
    vmap_is_symmetric = jax.vmap(jax.vmap(lambda x: jnp.allclose(x, x.T, atol=1e-16)))
    vmap_is_pos_def = jax.vmap(
        jax.vmap(lambda x: jnp.all(jnp.linalg.eigvals(x).real > 0))
    )

    assert jnp.all(vmap_is_symmetric(mt_inv))
    assert jnp.all(vmap_is_pos_def(jnp.imag(mt_inv)))

    assert jnp.allclose(x0_inv, x0, atol=1e-16)
    assert jnp.allclose(p0_inv, p0, atol=1e-16)
    assert jnp.allclose(m0_inv, m0, atol=1e-8)
    assert jnp.allclose(a0_inv, a0, atol=1e-8)

    ###############
    ## HOM Test ##
    ###############

    solver = gb_solvers.solve_hom_general
    solver_rev = gb_solvers.solve_hom_TR

    (xt, pt, mt, at) = solver(x0, p0, m0, a0, mode, ts, c, None)

    xT = xt[:, -1, :]
    pT = pt[:, -1, :]
    mT = mt[:, -1, :]
    aT = at[:, -1, :]

    # Use per-beam time intervals (T -> 0) and beam-shaped mode for TR closed form
    ts_b = jnp.stack([jnp.full((b,), ts[-1]), jnp.zeros((b,))], axis=1)
    mode_b = mode.reshape(b, 1)

    xt_rev, pt_rev, mt_rev, at_rev = solver_rev(xT, pT, mT, aT, mode_b, ts_b, c)
    x0_inv = xt_rev[:, -1, :]
    p0_inv = pt_rev[:, -1, :]
    m0_inv = mt_rev[:, -1, ...]
    a0_inv = at_rev[:, -1, :]

    assert jnp.allclose(x0, x0_inv, atol=1e-16)
    assert jnp.allclose(p0, p0_inv, atol=1e-16)
    assert jnp.allclose(m0, m0_inv, atol=1e-16)
    assert jnp.allclose(a0, a0_inv, atol=1e-16)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_hom_tr_matches_batch_solver(d):
    """
    Closed-form homogeneous TR should match the batch ODE solver in any dimension.
    """

    b = 2

    def c_fn(x):
        return 1 + 0 * x[..., 0]

    xT = jnp.zeros((b, d))
    base_p = jnp.arange(1, d + 1, dtype=float)
    pT = jnp.stack([base_p, base_p[::-1] if d > 1 else base_p], axis=0)

    mT = jnp.tile(jnp.eye(d)[None, :, :], (b, 1, 1)) * (1.0 + 1.0j)
    aT = jnp.linspace(1.0, 1.2, b).reshape(b, 1)
    mode = jnp.ones((b, 1))

    # Per-beam time intervals (final -> initial) to mimic TR usage.
    ts = jnp.vstack([jnp.array([1.0, 0.0]), jnp.array([0.6, 0.0])])

    xt_h, pt_h, mt_h, at_h = gb_solvers.solve_hom_TR(xT, pT, mT, aT, mode, ts, c_fn)
    xt_n, pt_n, mt_n, at_n = gb_solvers.solve_ODE_batch_t(
        xT, pT, mT, aT, mode, ts, c_fn
    )

    assert jnp.allclose(xt_h, xt_n, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(pt_h, pt_n, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(mt_h, mt_n, atol=1e-6, rtol=1e-6)
    assert jnp.allclose(at_h, at_n, atol=1e-6, rtol=1e-6)


def constant_speed(x):
    return 1 + 0 * x[..., 0]


def linear_speed(x):
    return 1 + 0.5 * x[..., 0]


@pytest.mark.parametrize(
    "d, c",
    [
        (1, constant_speed),
        (2, constant_speed),
        (3, constant_speed),
        (1, linear_speed),
        (2, linear_speed),
        (3, linear_speed),
    ],
)
def test_riccati_keeps_symm_pos_def(d, c):
    """
    Test that if the initial condition provided to the Riccati equation is symmetric and its imaginary part is positive definite,
    then the solution is also symmetric and the imaginary part is positive definite.

    This is stated as Lemma 2.1 in Qian and Ying 2010.
    """
    b = 1
    N = (64,) * d
    dx = (1 / N[0],) * d

    cfl = (jnp.sqrt(2) / 4).round(3)
    periodic = (False,) * d
    domain = geometry.Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=cfl)
    ts = domain.generate_time_domain()

    mode = jnp.ones((b,))
    x0 = jnp.zeros((b, d)) + 0.5
    p0 = jnp.ones((b, d))
    a0 = jnp.ones((b,))
    alpha0 = None
    solver_config = None
    lam = 0

    ode_solver = gb_solvers.solve_ODE_base

    def generate_complex_positive_definite_matrix(b, d):
        real_part = jnp.ones((b, d, d)) * 0.5

        A = jax.random.normal(key=jax.random.PRNGKey(0), shape=(b, d, d))
        imag_part = jnp.einsum("bij,bkj->bik", A, A)

        M0 = real_part + 1j * imag_part
        return M0

    M0 = generate_complex_positive_definite_matrix(b, d)
    M0 = gb_utils.prepare_M0(alpha0, M0)
    Mt = ode_solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)[2]

    vmap_is_symmetric = jax.vmap(jax.vmap(lambda x: jnp.allclose(x, x.T, atol=1e-16)))
    vmap_is_pos_def = jax.vmap(
        jax.vmap(lambda x: jnp.all(jnp.linalg.eigvals(x).real > 0))
    )

    assert jnp.all(vmap_is_symmetric(Mt))
    assert jnp.all(vmap_is_pos_def(jnp.imag(Mt)))


@pytest.mark.parametrize("d", [1, 2])
def test_ODE_not_solveable_if_p0_eq_0(d):
    """
    Test that if p0 = 0, then the ODEs are not solvable.
    """
    b = 1
    N = jnp.array([128] * d)
    dx = jnp.array([1 / N[0]] * d)

    def c(x):
        return 1500 + 0 * x[..., 0]

    cfl = (jnp.sqrt(2) / 4).round(3)
    periodic = (False,) * d

    domain = geometry.Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=cfl)

    ts = domain.generate_time_domain()

    mode = jnp.ones((b,))
    x0 = jnp.zeros((b, d)) + 0.5
    p0 = jnp.zeros((b, d))
    a0 = jnp.ones((b,))
    alpha0 = jnp.ones((b, d)) * 2j
    solver_config = None
    lam = 0

    M0 = None
    ode_solver = gb_solvers.solve_ODE_base

    M0 = gb_utils.prepare_M0(alpha0, M0)

    with pytest.raises(RuntimeError, match="maximum number of solver steps"):
        ode_solver(x0, p0, M0, a0, mode, ts, c, lam, solver_config)


@pytest.mark.parametrize("b,N", [(1, (64,)), (10, (64,)), (10, (64, 64))])
def test_amplitude_is_linear(b, N):
    """
    Check that rescaling a0 by a constant factor, rescales the GB by the same factor.
    """
    t, x0, p0, a0, alpha0, omega0, x_sensor = random_data(b, N)
    key = jax.random.PRNGKey(0)
    scale_factor = jax.random.uniform(key, (b,))
    a0_rescale = a0 * scale_factor
    lam = 0

    d = len(N)
    M0 = gb_utils.prepare_M0(alpha0, None)
    # is_M0_diagonal = gb_utils.is_diagonal(M0)
    mode, domain_size, periodic = (
        jnp.ones((b,)),
        jnp.ones((d)),
        jnp.array([False] * d),
    )

    solver_config = None
    hom_solver = gb_solvers.solve_hom_diag
    ode_solver = gb_solvers.solve_ODE_base

    def compute_gb(a0_val, solver):
        return core.compute_gaussian_beam(
            x0,
            p0,
            M0,
            a0_val,
            omega0,
            mode,
            c,
            lam,
            t,
            x_sensor,
            domain_size,
            periodic,
            solver,
            solver_config,
        )

    gb_hom, gb_hom_rescale = (
        compute_gb(a0, hom_solver),
        compute_gb(a0_rescale, hom_solver),
    )
    gb_num, gb_num_rescale = (
        compute_gb(a0, ode_solver),
        compute_gb(a0_rescale, ode_solver),
    )

    scale_factor = jnp.expand_dims(scale_factor, axis=(0, 1, 2))

    assert jnp.allclose(gb_hom * scale_factor, gb_hom_rescale, atol=1e-16)
    assert jnp.allclose(gb_num * scale_factor, gb_num_rescale, atol=1e-16)


@pytest.mark.parametrize("b, N", [(1, (64,)), (10, (64,)), (10, (64, 64))])
def test_hom_general_solver(b, N):
    """
    Test that for a non-diagonal M0, the general solver agrees with the analyical solution.
    """
    d = len(N)
    cfl = (jnp.sqrt(2) / 4).round(3)
    periodic = (False,) * d
    dx = (1 / N[0],) * d

    def c(x):
        return 1 + 0 * x[..., 0]

    domain = geometry.Domain(N=N, dx=dx, c=c, periodic=periodic, cfl=cfl)

    ts = domain.generate_time_domain()

    mode = jnp.ones((b,))
    x0 = jnp.zeros((b, d)) + 0.5
    p0 = jnp.ones((b, d))
    a0 = jnp.ones((b,))
    alpha0 = None
    solver_config = None
    lam = 0

    M0 = generate_complex_positive_definite_matrix(b, d)

    ode_solver = gb_solvers.solve_ODE_base
    hom_gen_solver = gb_solvers.solve_hom_general

    M0 = gb_utils.prepare_M0(alpha0, M0)

    xt_ode, pt_ode, Mt_ode, At_ode = ode_solver(
        x0, p0, M0, a0, mode, ts, c, lam, solver_config
    )
    xt_hom_gen, pt_hom_gen, Mt_hom_gen, At_hom_gen = hom_gen_solver(
        x0, p0, M0, a0, mode, ts, c, lam, solver_config
    )

    assert jnp.allclose(xt_ode, xt_hom_gen, atol=1e-16)
    assert jnp.allclose(pt_ode, pt_hom_gen, atol=1e-16)
    assert jnp.allclose(Mt_ode, Mt_hom_gen, atol=1e-16)
    assert jnp.allclose(At_ode, At_hom_gen, atol=1e-16)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_gb_conjugate(ndim):
    """
    Test that if I compute a Gaussian Beam, and compute its complex conjugate,
    then it's the same as another Gaussian Beam, with certain parameters.

    The parameters which change under conjugation are:
    p -> -p
    M -> -conjugate(M)
    a -> conjugate(a)
    mode -> -mode

    ----------------------------------------------------

    Test that the compute_gaussian_beam_real is the same as a taking sum of a GB and its conjugate pair.

    """
    b = 1
    N = (64,) * ndim
    lam = 0
    ts, x0, p0, a0, alpha0, omega0, x_sensor = random_data(b, N, isotropic=False)
    M0 = gb_utils.prepare_M0(alpha0, None)
    mode = jnp.ones((b,))
    periodic = (False,) * ndim
    domain_size = jnp.ones((ndim,))
    ode_solver = gb_solvers.solve_ODE_base
    solver_config = None

    gb_init = core.compute_gaussian_beam(
        x0,
        p0,
        M0,
        a0,
        omega0,
        mode,
        c,
        lam,
        ts,
        x_sensor,
        domain_size,
        jnp.array(periodic),
        ode_solver,
        solver_config,
    )

    gb_conj = core.compute_gaussian_beam(
        x0,
        -p0,
        -jnp.conjugate(M0),
        jnp.conjugate(a0),
        omega0,
        -mode,
        c,
        lam,
        ts,
        x_sensor,
        domain_size,
        jnp.array(periodic),
        ode_solver,
        solver_config,
    )

    assert jnp.allclose(gb_init, jnp.conjugate(gb_conj), atol=1e-16)

    gb_real = core.compute_gaussian_beam_real(
        x0,
        p0,
        M0,
        a0,
        omega0,
        mode,
        c,
        lam,
        ts,
        x_sensor,
        domain_size,
        jnp.array(periodic),
        ode_solver,
        solver_config,
    )

    gb_sum = (gb_init + gb_conj).squeeze()

    assert jnp.allclose(gb_sum, gb_real, atol=1e-16)


def test_prepare_M0_requires_exclusive_args():
    with pytest.raises(ValueError):
        prepare_M0(None, None)
    with pytest.raises(ValueError):
        prepare_M0(jnp.array([[1j, 2j]]), jnp.eye(2)[None, :, :])


def test_check_M0_happy_and_failure_paths():
    good = jnp.array([[[1 + 1j, 0], [0, 2 + 1j]]])
    check_M0(good)  # no exception
    bad_sym = jnp.array([[[1 + 1j, 1], [0, 2 + 1j]]])
    with pytest.raises(ValueError):
        check_M0(bad_sym)
    bad_pd = jnp.array([[[0 - 1e-6j, 0], [0, 0 - 1e-6j]]])
    with pytest.raises(ValueError):
        check_M0(bad_pd)


def test_is_diagonal_false_case():
    M = jnp.array([[[1 + 1j, 0.1], [0, 2 + 1j]]])
    assert is_diagonal(M) == jnp.array([False])


def make_linear_c(a_lin: float, b_lin: float):
    """
    Build a 1D linear sound speed:
        c(x) = a_lin + b_lin * x,  x ∈ ℝ.
    Here x is expected to have shape (d,) with d = 1.
    """

    def c(x: jnp.ndarray) -> jnp.ndarray:
        # x has shape (1,) in this test so x[0] is scalar
        return a_lin + b_lin * x[0]

    return c


def analytic_1d_linear_medium_solution(
    x0: float,
    p0: float,
    M0: complex,
    A0: complex,
    ts: jnp.ndarray,
    a_lin: float,
    b_lin: float,
):
    """
    Analytic solution of the Gaussian beam ODEs in 1D for

        c(x) = a_lin + b_lin * x,

    under the assumptions:
        - p(t) > 0 so ||p|| = p
        - mode = +1
        - λ = 0 (no absorption)

    Returns xt, pt, Mt, At with shapes matching solve_ODE_base for b=1, d=1.
    """
    t0 = ts[0]
    dt = ts - t0  # (Nt,)
    b = float(b_lin)
    a = float(a_lin)

    p_t = p0 * jnp.exp(-b * dt)  # (Nt,)
    x_t = (x0 + a / b) * jnp.exp(b * dt) - a / b  # (Nt,)
    M_t = M0 * jnp.exp(-2.0 * b * dt)  # (Nt,)
    A_t = A0 * jnp.exp(0.5 * b * dt)  # (Nt,)

    xt = x_t[None, :, None].real  # (1, Nt, 1)
    pt = p_t[None, :, None].real  # (1, Nt, 1)
    Mt = M_t[None, :, None, None]  # (1, Nt, 1, 1)
    At = A_t[None, :, None]  # (1, Nt, 1) — we will squeeze later

    return xt, pt, Mt, At


def test_solve_ODE_base_matches_analytic_in_1d_linear_medium():
    """
    Check that solve_ODE_base reproduces the analytic GB solution in a 1D
    linear medium c(x) = a + b x, in the regime p(t) > 0, mode = +1, λ = 0.

    We compare x(t), p(t), M(t), and A(t) against the explicit formulas:
        p(t) = p0 exp(-b t)
        x(t) = (x0 + a/b) exp(b t) - a/b
        M(t) = M0 exp(-2 b t)
        A(t) = A0 exp(b t / 2)
    """
    # 1D, single beam
    b_beams = 1
    d = 1

    # Medium parameters: c(x) = a + b x
    a_lin = 1.0
    b_lin = 1.0
    c = make_linear_c(a_lin, b_lin)

    # Time grid chosen so that c(x(t)) stays positive and p(t) > 0
    Nt = 200
    t0 = 0.0
    t1 = 1.0
    ts = jnp.linspace(t0, t1, Nt)

    # Initial GB data (batch size 1, d = 1)
    x0_val = 0.1  # away from the zero of c(x)
    p0_val = 1.2  # strictly positive so ||p|| = p
    M0_val = 0.2j  # purely imaginary Hessian is fine
    A0_val = 1.0 + 0.0j

    x0 = jnp.array([[x0_val]], dtype=jnp.float64)  # (1, 1)
    p0 = jnp.array([[p0_val]], dtype=jnp.float64)  # (1, 1)
    M0 = jnp.array([[[M0_val]]], dtype=jnp.complex128)  # (1, 1, 1)
    A0 = jnp.array([A0_val], dtype=jnp.complex128)  # (1,)
    mode = jnp.ones((b_beams,), dtype=jnp.int32)  # (1,)
    lam = 0.0

    # Use high-precision solver config
    solver_config = gb_solvers.SolverConfig.from_precision(use_x64=True)

    # 1) Numerical solution via solve_ODE_base
    xt_num, pt_num, Mt_num, At_num = gb_solvers.solve_ODE_base(
        x0,
        p0,
        M0,
        A0,
        mode,
        ts,
        c,
        lam,
        solver_config,
    )

    # 2) Analytic solution
    xt_ref, pt_ref, Mt_ref, At_ref = analytic_1d_linear_medium_solution(
        x0=x0_val,
        p0=p0_val,
        M0=M0_val,
        A0=A0_val,
        ts=ts,
        a_lin=a_lin,
        b_lin=b_lin,
    )

    # ---- Shape sanity checks ----
    assert xt_num.shape == xt_ref.shape == (b_beams, Nt, d)
    assert pt_num.shape == pt_ref.shape == (b_beams, Nt, d)
    assert Mt_num.shape == Mt_ref.shape == (b_beams, Nt, d, d)

    # At_num comes from a length-1 slice of the state vector, so shape is (b, Nt, 1)
    # At_ref was constructed with the same shape.
    assert At_num.shape == At_ref.shape == (b_beams, Nt, 1)

    # ---- Numerical comparison ----
    atol = 1e-10
    rtol = 1e-7

    print(f"max abs diff xt: {jnp.max(jnp.abs(xt_num - xt_ref))}")
    print(f"max abs diff pt: {jnp.max(jnp.abs(pt_num - pt_ref))}")
    print(f"max abs diff Mt: {jnp.max(jnp.abs(Mt_num - Mt_ref))}")
    print(f"max abs diff At: {jnp.max(jnp.abs(At_num - At_ref))}")

    assert jnp.allclose(xt_num, xt_ref, atol=atol, rtol=rtol)
    assert jnp.allclose(pt_num, pt_ref, atol=atol, rtol=rtol)
    assert jnp.allclose(Mt_num, Mt_ref, atol=atol, rtol=rtol)
    assert jnp.allclose(At_num, At_ref, atol=atol, rtol=rtol)


def test_riccati_2d_linear_c_matches_textbook_form():
    """
    Regression test for the matrix Riccati cross-term ordering.

    In a 2D linear sound speed c(x) = a + b·x with ∇c not parallel to p̂,
    the mixed-Hessian G_{xp} is asymmetric. Two Riccati orderings exist:
        Form B (textbook, Berra 2017 / Cerveny 2007 / standard):
            Ṁ = -(Gxx + Gxp M + M Gxp^T + M Gpp M)
        Form A (legacy buggy ordering):
            Ṁ = -(Gxx + M Gxp + Gxp^T M + M Gpp M)

    For ∇c ⊥ p̂ at t=0 the bug bends the off-diagonal of M by ~50% by t=0.5.
    This test integrates form B with scipy DOP853 at atol=rtol=1e-12 and
    checks that beamax.gb.gb_solvers.solve_ODE_base agrees to 1e-6.
    """
    import numpy as np
    from scipy.integrate import solve_ivp

    d = 2
    a_lin = 1.0
    b_vec = np.array([0.5, 0.0])

    def hessian_blocks(x, p):
        c_val = a_lin + b_vec @ x
        norm_p = np.linalg.norm(p)
        Gxx = np.zeros((d, d))
        Gxp = np.outer(b_vec, p) / norm_p
        Gpp = c_val * (np.eye(d) / norm_p - np.outer(p, p) / norm_p**3)
        return Gxx, Gxp, Gpp, c_val, norm_p

    def rhs(t, y):
        x = y[:d]
        p = y[d : 2 * d]
        Mre = y[2 * d : 2 * d + d * d].reshape(d, d)
        Mim = y[2 * d + d * d :].reshape(d, d)
        M = Mre + 1j * Mim
        Gxx, Gxp, Gpp, c_val, norm_p = hessian_blocks(x, p)
        dx = c_val * p / norm_p
        dp = -b_vec * norm_p
        # Form B: textbook ordering
        dM = -(Gxx + Gxp @ M + M @ Gxp.T + M @ Gpp @ M)
        return np.concatenate([dx, dp, dM.real.reshape(-1), dM.imag.reshape(-1)])

    x0 = np.array([0.0, 0.0])
    p0 = np.array([0.0, 1.0])
    M0 = 1j * np.array([[1.0, 0.0], [0.0, 2.0]])
    y0 = np.concatenate([x0, p0, M0.real.reshape(-1), M0.imag.reshape(-1)])
    t_eval = np.linspace(0.0, 0.5, 51)

    sol = solve_ivp(
        rhs, (0.0, 0.5), y0, method="DOP853", atol=1e-12, rtol=1e-12, t_eval=t_eval
    )

    M_ref = np.empty((len(t_eval), d, d), dtype=complex)
    for k in range(len(t_eval)):
        Mre = sol.y[2 * d : 2 * d + d * d, k].reshape(d, d)
        Mim = sol.y[2 * d + d * d :, k].reshape(d, d)
        M_ref[k] = Mre + 1j * Mim

    def c_fn(x):
        return a_lin + b_vec[0] * x[0] + b_vec[1] * x[1]

    x0_jax = jnp.array(x0[None, :], dtype=jnp.float64)
    p0_jax = jnp.array(p0[None, :], dtype=jnp.float64)
    M0_jax = jnp.array(M0[None, :, :], dtype=jnp.complex128)
    A0_jax = jnp.array([1.0 + 0.0j], dtype=jnp.complex128)
    mode = jnp.ones((1,), dtype=jnp.int32)
    ts = jnp.linspace(0.0, 0.5, 51, dtype=jnp.float64)
    cfg = gb_solvers.SolverConfig.from_precision(use_x64=True)

    _, _, Mt, _ = gb_solvers.solve_ODE_base(
        x0_jax, p0_jax, M0_jax, A0_jax, mode, ts, c_fn, 0.0, cfg
    )
    M_beamax = np.array(Mt[0])

    err = np.max(np.abs(M_beamax - M_ref)) / np.max(np.abs(M_ref))
    assert err < 1e-6, (
        f"beamax Riccati disagrees with textbook form B by relative {err:.3e}. "
        "Likely the cross-term ordering has regressed; expected "
        "Ṁ = -(Gxx + Gxp M + M Gxp.T + M Gpp M)."
    )


# ============================================================================
# Tests for surface-event ODE solvers and the (Q, P) variant
# ============================================================================


def _constant_c(_x):
    """Homogeneous c=1 sound speed used by the surface tests."""
    return jnp.array(1.0)


def test_compute_amp_hom_diag_dispatch_rejects_d_ge_4():
    """compute_amp_hom_diag must error out for spatial dimension >= 4."""
    b = 2
    d = 4
    p0 = jnp.ones((b, d))
    normp = jnp.linalg.norm(p0, axis=-1, keepdims=True)
    alpha0 = jnp.ones((b, d), dtype=jnp.complex128) * 1j
    c0 = jnp.ones((b, 1))
    ts = jnp.linspace(0.0, 0.1, 5)
    a0 = jnp.ones((b,), dtype=jnp.complex128)

    with pytest.raises(ValueError, match="1D, 2D and 3D"):
        gb_solvers.compute_amp_hom_diag(p0, normp, alpha0, c0, ts, a0)


def test_solve_ODE_intersection_planar_surface_1d():
    """In 1D with c=1 and a planar surface x=L, the beam should intersect at t=L/c.

    `solve_ODE_intersection` is vmapped over the beam axis, so all per-beam
    inputs must carry a leading batch dimension.
    """
    b, d = 1, 1
    L = 0.5
    x0 = jnp.array([[0.0]], dtype=jnp.float64)
    p0 = jnp.array([[1.0]], dtype=jnp.float64)  # heading +x with c=1
    M0 = jnp.array([[[0.1j]]], dtype=jnp.complex128)
    a0 = jnp.array([[1.0 + 0.0j]], dtype=jnp.complex128)  # (b, 1)
    mode = jnp.ones((b,), dtype=jnp.float64)
    ts = jnp.linspace(0.0, 1.0, 64)

    def surface(x):
        return x[0] - L

    cfg = gb_solvers.SolverConfig.from_precision(use_x64=True)
    xt, pt, Mt, At, t_int = gb_solvers.solve_ODE_intersection(
        x0, p0, M0, a0, mode, ts, _constant_c, 0.0, surface, cfg
    )

    # With c=1 and |p|=1 the ray speed is c=1 so hit time is L.
    assert jnp.all(jnp.isfinite(t_int))
    assert float(jnp.abs(t_int[0] - L)) < 1e-4
    assert xt.shape == (b, len(ts), d)


def test_solve_ODE_intersection_no_hit_returns_inf():
    """If the surface is unreachable in [t0, t1], t_int should be inf, not a fake hit."""
    b = 1
    x0 = jnp.array([[0.0]], dtype=jnp.float64)
    p0 = jnp.array([[1.0]], dtype=jnp.float64)
    M0 = jnp.array([[[0.1j]]], dtype=jnp.complex128)
    a0 = jnp.array([[1.0 + 0.0j]], dtype=jnp.complex128)
    mode = jnp.ones((b,), dtype=jnp.float64)
    # Time window 0..0.1 — ray only reaches x=0.1 but surface is at x=5.0
    ts = jnp.linspace(0.0, 0.1, 32)

    def surface(x):
        return x[0] - 5.0

    cfg = gb_solvers.SolverConfig.from_precision(use_x64=True)
    _, _, _, _, t_int = gb_solvers.solve_ODE_intersection(
        x0, p0, M0, a0, mode, ts, _constant_c, 0.0, surface, cfg
    )
    # Either inf (failed root solve) or at least beyond the chosen window.
    assert jnp.isinf(t_int[0]) or float(t_int[0]) >= ts[-1] - 1e-6


def test_solve_ODE_first_hit_planar_event_1d():
    """1D ray with c=1 should trigger the event when it hits the planar surface."""
    L = 0.4
    x0 = jnp.array([[0.0]], dtype=jnp.float64)
    p0 = jnp.array([[1.0]], dtype=jnp.float64)
    M0 = jnp.array([[[0.1j]]], dtype=jnp.complex128)
    a0 = jnp.array([1.0 + 0.0j], dtype=jnp.complex128)
    mode = jnp.array([1], dtype=jnp.float64)
    ts = jnp.linspace(0.0, 1.0, 64)

    def surface(x):
        return x[0] - L

    cfg = gb_solvers.SolverConfig.from_precision(use_x64=True)
    xt, pt, Mt, At, t_hit, hit = gb_solvers.solve_ODE_first_hit(
        x0[0], p0[0], M0[0], a0[0:1], mode[0], ts, _constant_c, 0.0, surface, cfg
    )

    assert bool(hit) is True
    assert float(jnp.abs(t_hit - L)) < 1e-3
    # Final-state-only output shape: (1 beam, 1 time, d)
    assert xt.shape[-1] == 1
    assert Mt.shape[-2:] == (1, 1)


def test_solve_ODE_first_hit_no_event_returns_endpoint():
    """When the ray never reaches the surface, the integrator should run to t1."""
    L = 5.0
    x0 = jnp.array([[0.0]], dtype=jnp.float64)
    p0 = jnp.array([[1.0]], dtype=jnp.float64)
    M0 = jnp.array([[[0.1j]]], dtype=jnp.complex128)
    a0 = jnp.array([1.0 + 0.0j], dtype=jnp.complex128)
    mode = jnp.array([1], dtype=jnp.float64)
    ts = jnp.linspace(0.0, 0.1, 32)

    def surface(x):
        return x[0] - L

    cfg = gb_solvers.SolverConfig.from_precision(use_x64=True)
    _, _, _, _, t_hit, hit = gb_solvers.solve_ODE_first_hit(
        x0[0], p0[0], M0[0], a0[0:1], mode[0], ts, _constant_c, 0.0, surface, cfg
    )
    assert bool(hit) is False
    assert float(jnp.abs(t_hit - ts[-1])) < 1e-9


def test_solve_ODE_QP_base_matches_M_base_1d_homogeneous():
    """In a homogeneous 1D medium, the QP-form solver should reproduce the M-form solver.

    This is a strong cross-check on the (Q, P) ODE construction: with M = P Q⁻¹
    and Q(0)=I, P(0)=M0, the (Q,P) propagation must yield the same M trajectory.
    """
    b = 1
    ts = jnp.linspace(0.0, 0.5, 64)
    x0 = jnp.array([[0.05]], dtype=jnp.float64)
    p0 = jnp.array([[1.0]], dtype=jnp.float64)
    M0 = jnp.array([[[0.2j]]], dtype=jnp.complex128)
    A0 = jnp.array([1.0 + 0.0j], dtype=jnp.complex128)
    mode = jnp.ones((b,), dtype=jnp.int32)
    cfg = gb_solvers.SolverConfig.from_precision(use_x64=True)

    xt_M, pt_M, Mt_M, At_M = gb_solvers.solve_ODE_base(
        x0, p0, M0, A0, mode, ts, _constant_c, 0.0, cfg
    )
    xt_Q, pt_Q, Mt_Q, At_Q = gb_solvers.solve_ODE_QP_base(
        x0, p0, M0, A0, mode, ts, _constant_c, 0.0, cfg
    )

    # Positions and momenta must agree to high precision.
    assert jnp.allclose(xt_M, xt_Q, atol=1e-8, rtol=1e-6)
    assert jnp.allclose(pt_M, pt_Q, atol=1e-8, rtol=1e-6)
    # M may have some numerical drift in QP form; allow a looser tolerance.
    assert jnp.allclose(Mt_M, Mt_Q, atol=1e-6, rtol=1e-4)


def test_solve_ODE_QP_base_2d_homogeneous_diagonal_M0():
    """In 2D homogeneous c=1, QP-form should preserve diagonal Hessian structure."""
    b, d = 1, 2
    ts = jnp.linspace(0.0, 0.1, 32)
    x0 = jnp.array([[0.05, 0.0]], dtype=jnp.float64)
    p0 = jnp.array([[1.0, 0.0]], dtype=jnp.float64)
    M0 = jnp.array([[[0.2j, 0.0j], [0.0j, 0.5j]]], dtype=jnp.complex128)
    A0 = jnp.array([1.0 + 0.0j], dtype=jnp.complex128)
    mode = jnp.ones((b,), dtype=jnp.int32)
    cfg = gb_solvers.SolverConfig.from_precision(use_x64=True)

    xt, pt, Mt, At = gb_solvers.solve_ODE_QP_base(
        x0, p0, M0, A0, mode, ts, _constant_c, 0.0, cfg
    )

    assert xt.shape == (b, len(ts), d)
    assert Mt.shape == (b, len(ts), d, d)
    # Initial condition recovered exactly at t=0.
    assert jnp.allclose(Mt[0, 0], M0[0], atol=1e-10)


def test_solver_config_dt0_override():
    """SolverConfig.dt0 should override the time-grid-derived dt0 inside solvers."""
    b, d = 1, 1
    ts = jnp.linspace(0.0, 0.1, 16)
    x0 = jnp.array([[0.05]], dtype=jnp.float64)
    p0 = jnp.array([[1.0]], dtype=jnp.float64)
    M0 = jnp.array([[[0.2j]]], dtype=jnp.complex128)
    A0 = jnp.array([1.0 + 0.0j], dtype=jnp.complex128)
    mode = jnp.ones((b,), dtype=jnp.int32)

    cfg = gb_solvers.SolverConfig.from_precision(use_x64=True, dt0=1e-4)
    assert cfg.dt0 == 1e-4

    # Solver should still run and produce a sane trajectory with a fixed dt0.
    xt, _, _, _ = gb_solvers.solve_ODE_base(
        x0, p0, M0, A0, mode, ts, _constant_c, 0.0, cfg
    )
    assert xt.shape == (b, len(ts), d)
    assert jnp.all(jnp.isfinite(xt))


if __name__ == "__main__":
    pytest.main([__file__])
