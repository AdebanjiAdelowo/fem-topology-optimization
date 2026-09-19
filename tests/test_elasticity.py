import numpy as np
import ufl
from dolfinx import fem

from src.elasticity import (
    build_mesh, lame_parameters, strain, stress, solve_elasticity, nearest_node_dofs,
)


def test_lame_parameters_plane_stress_vs_strain_differ():
    E, nu = 1.0, 0.3
    lam_stress, mu_stress = lame_parameters(E, nu, "stress")
    lam_strain, mu_strain = lame_parameters(E, nu, "strain")
    assert mu_stress == mu_strain  # mu is identical in both cases
    assert lam_stress != lam_strain


def test_lame_parameters_invalid_plane_raises():
    try:
        lame_parameters(1.0, 0.3, "nonsense")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_strain_tensor_is_symmetric_part_of_gradient():
    em = build_mesh(4, 4, L=1.0, H=1.0)
    x = ufl.SpatialCoordinate(em.mesh)
    u = ufl.as_vector((x[0] ** 2, x[0] * x[1]))
    eps = strain(u)
    # grad(u) = [[du1/dx0, du1/dx1], [du2/dx0, du2/dx1]] = [[2*x0, 0], [x1, x0]]
    # eps = sym(grad(u)) = [[2*x0, x1/2], [x1/2, x0]]
    expected = ufl.as_matrix([[2 * x[0], x[1] / 2], [x[1] / 2, x[0]]])
    diff = fem.assemble_scalar(fem.form(ufl.inner(eps - expected, eps - expected) * ufl.dx))
    assert diff < 1e-20


def test_stress_reduces_to_hydrostatic_for_pure_dilation():
    """For u = c*x (pure isotropic scaling), epsilon = c*I, so
    sigma = 2*mu*c*I + lambda*tr(c*I)*I = (2*mu*c + 2*lambda*c)*I -- purely
    diagonal, no shear."""
    em = build_mesh(4, 4, L=1.0, H=1.0)
    x = ufl.SpatialCoordinate(em.mesh)
    c = 0.1
    u = c * x
    lam, mu = lame_parameters(1.0, 0.3, "stress")
    sigma = stress(u, lam, mu)
    off_diag = fem.assemble_scalar(fem.form(sigma[0, 1] ** 2 * ufl.dx))
    assert off_diag < 1e-20


def test_homogeneous_material_matches_known_cantilever_scaling():
    """Displacement must scale exactly linearly with 1/E and with the load
    magnitude for linear elasticity."""
    em = build_mesh(10, 5, L=2.0, H=1.0)
    point, load = (2.0, 0.5), (0.0, -1.0)

    u1 = solve_elasticity(em, 1.0, 0.3, point_loads=[(point, load)])
    u2 = solve_elasticity(em, 2.0, 0.3, point_loads=[(point, load)])
    u3 = solve_elasticity(em, 1.0, 0.3, point_loads=[(point, (0.0, -2.0))])

    _, dofs = nearest_node_dofs(em.V, point)
    v1, v2, v3 = u1.x.array[dofs[1]], u2.x.array[dofs[1]], u3.x.array[dofs[1]]

    assert np.isclose(v2, v1 / 2, rtol=1e-10)
    assert np.isclose(v3, v1 * 2, rtol=1e-10)


def test_clamped_boundary_has_zero_displacement():
    em = build_mesh(10, 5, L=2.0, H=1.0)
    uh = solve_elasticity(em, 1.0, 0.3, point_loads=[((2.0, 0.5), (0.0, -1.0))])
    coords = em.V.tabulate_dof_coordinates()
    left_blocks = np.where(np.isclose(coords[:, 0], 0.0))[0]
    bs = em.V.dofmap.index_map_bs
    for b in left_blocks:
        assert abs(uh.x.array[b * bs]) < 1e-12
        assert abs(uh.x.array[b * bs + 1]) < 1e-12


def test_deterministic_reproducibility():
    em1 = build_mesh(8, 4, L=2.0, H=1.0)
    u1 = solve_elasticity(em1, 1.0, 0.3, point_loads=[((2.0, 0.5), (0.0, -1.0))])
    em2 = build_mesh(8, 4, L=2.0, H=1.0)
    u2 = solve_elasticity(em2, 1.0, 0.3, point_loads=[((2.0, 0.5), (0.0, -1.0))])
    assert np.array_equal(u1.x.array, u2.x.array)


def test_nearest_node_dofs_finds_correct_point():
    em = build_mesh(4, 4, L=1.0, H=1.0)
    block, dofs = nearest_node_dofs(em.V, (0.0, 0.0))
    coords = em.V.tabulate_dof_coordinates()
    assert np.allclose(coords[block, :2], [0.0, 0.0])
