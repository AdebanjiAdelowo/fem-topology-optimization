import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import Function

from src.elasticity import build_mesh, lame_parameters
from src.postprocess import von_mises, strain_energy_density


def test_von_mises_zero_for_zero_stress():
    em = build_mesh(4, 4, L=1.0, H=1.0)
    u = Function(em.V)
    u.x.array[:] = 0.0
    lam, mu = lame_parameters(1.0, 0.3, "stress")
    val = fem.assemble_scalar(fem.form(von_mises(u, lam, mu) * ufl.dx))
    assert abs(val) < 1e-20


def test_von_mises_uniaxial_matches_axial_stress():
    """For a pure uniaxial stress state (sigma_xx = s, sigma_yy = sigma_xy = 0),
    von Mises reduces to |s|."""
    em = build_mesh(4, 4, L=1.0, H=1.0)
    x = ufl.SpatialCoordinate(em.mesh)
    lam, mu = lame_parameters(1.0, 0.0, "stress")  # nu=0 decouples axes
    s_target = 2.0
    # with nu=0: sigma_xx = E*epsilon_xx => epsilon_xx = s_target/E = s_target
    u = ufl.as_vector((s_target * x[0], 0.0 * x[0]))
    vm_val = fem.assemble_scalar(fem.form(von_mises(u, lam, mu) * ufl.dx))
    assert abs(vm_val - s_target) < 1e-10


def test_strain_energy_density_is_nonnegative():
    em = build_mesh(4, 4, L=1.0, H=1.0)
    x = ufl.SpatialCoordinate(em.mesh)
    u = ufl.as_vector((ufl.sin(x[0]), ufl.cos(x[1])))
    lam, mu = lame_parameters(1.0, 0.3, "stress")
    val = fem.assemble_scalar(fem.form(strain_energy_density(u, lam, mu) * ufl.dx))
    assert val >= 0.0


def test_strain_energy_zero_for_rigid_body_translation():
    em = build_mesh(4, 4, L=1.0, H=1.0)
    u = Function(em.V)
    u.interpolate(lambda x: np.vstack([np.full(x.shape[1], 1.0), np.full(x.shape[1], 2.0)]))
    lam, mu = lame_parameters(1.0, 0.3, "stress")
    val = fem.assemble_scalar(fem.form(strain_energy_density(u, lam, mu) * ufl.dx))
    assert abs(val) < 1e-20
