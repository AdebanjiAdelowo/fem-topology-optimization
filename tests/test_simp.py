import numpy as np

from src.elasticity import build_mesh
from src.simp import (
    SimpParams, build_filter, apply_filter, elemental_area, compliance_only,
    compliance_gradient, run_simp, grayness,
)


def test_simp_interpolation_endpoints_and_penalization():
    """E(rho=0) = Emin, E(rho=1) = E0, and for p>1 the interpolation lies
    BELOW the linear (p=1) rule-of-mixtures for intermediate rho (the
    defining property of the SIMP penalty)."""
    E0, Emin, p = 1.0, 1e-9, 3.0
    rho = np.array([0.0, 1.0, 0.5])
    E = Emin + rho**p * (E0 - Emin)
    assert np.isclose(E[0], Emin)
    assert np.isclose(E[1], E0)
    E_linear_half = Emin + 0.5 * (E0 - Emin)
    assert E[2] < E_linear_half


def test_filter_preserves_uniform_density():
    em = build_mesh(8, 4, L=2.0, H=1.0)
    H, Hs = build_filter(em, r_min=0.15)
    n = em.V0.dofmap.index_map.size_local
    rho = np.full(n, 0.37)
    rho_phys = apply_filter(rho, H, Hs)
    assert np.allclose(rho_phys, 0.37, atol=1e-10)


def test_filter_smooths_checkerboard():
    """A checkerboard density pattern should have strictly lower total
    variation after filtering (the filter's purpose: suppress
    checkerboarding, Sigmund & Petersson 1998)."""
    em = build_mesh(16, 8, L=2.0, H=1.0)
    H, Hs = build_filter(em, r_min=0.2)
    centers = em.mesh.geometry.x
    n = em.V0.dofmap.index_map.size_local
    rng = np.random.default_rng(0)
    rho = rng.choice([0.1, 0.9], size=n)
    rho_phys = apply_filter(rho, H, Hs)
    assert np.std(rho_phys) < np.std(rho)


def test_compliance_sensitivity_taylor_remainder():
    """Finite-difference (Taylor-remainder) check of the analytic compliance
    gradient dc/drho against directional finite differences: the remainder
    |c(rho+t*d) - c(rho) - t*grad.d| must shrink as O(t^2) as t -> 0,
    confirming the analytic sensitivity is correct before it is trusted for
    optimization."""
    em = build_mesh(8, 4, L=2.0, H=1.0)
    params = SimpParams(E0=1.0, Emin=1e-9, nu=0.3, p=3.0, Vf=0.4, r_min=0.1, move=0.2)
    point_loads = [((2.0, 0.5), (0.0, -1.0))]
    Hf, Hs = build_filter(em, params.r_min)
    elem_area = elemental_area(em)

    rng = np.random.default_rng(0)
    n = em.V0.dofmap.index_map.size_local
    rho0 = rng.uniform(0.2, 0.8, n)
    d = rng.uniform(-1.0, 1.0, n)

    c0, dc = compliance_gradient(em, rho0, params, point_loads, Hf, Hs, elem_area)
    grad_dir = dc @ d

    ts = [1e-2, 1e-3, 1e-4]
    remainders = []
    for t in ts:
        c_pert = compliance_only(em, rho0 + t * d, params, point_loads, Hf, Hs, elem_area)
        remainders.append(abs(c_pert - c0 - t * grad_dir))

    # observed order between successive (t, t/10) pairs should be close to 2
    for i in range(len(ts) - 1):
        order = np.log(remainders[i] / remainders[i + 1]) / np.log(ts[i] / ts[i + 1])
        assert 1.8 < order < 2.2


def test_small_mesh_optimization_runs_and_satisfies_volume_constraint():
    em = build_mesh(12, 6, L=2.0, H=1.0)
    params = SimpParams(E0=1.0, Emin=1e-9, nu=0.3, p=3.0, Vf=0.35, r_min=0.12, move=0.2)
    point_loads = [((2.0, 0.5), (0.0, -1.0))]
    res = run_simp(em, params, point_loads, max_iter=20, change_tol=1e-3)
    assert abs(res.history["volume_fraction"][-1] - params.Vf) < 1e-3
    # compliance should have decreased substantially from the uniform-density start
    assert res.history["compliance"][-1] < 0.5 * res.history["compliance"][0]
    g = grayness(res.rho_phys)
    assert 0.0 <= g <= 1.0


def test_optimization_reproducibility():
    em1 = build_mesh(10, 5, L=2.0, H=1.0)
    params = SimpParams(E0=1.0, Emin=1e-9, nu=0.3, p=3.0, Vf=0.4, r_min=0.1, move=0.2)
    point_loads = [((2.0, 0.5), (0.0, -1.0))]
    res1 = run_simp(em1, params, point_loads, max_iter=10, change_tol=1e-3)

    em2 = build_mesh(10, 5, L=2.0, H=1.0)
    res2 = run_simp(em2, params, point_loads, max_iter=10, change_tol=1e-3)

    assert np.array_equal(res1.rho, res2.rho)
    assert res1.history["compliance"] == res2.history["compliance"]
