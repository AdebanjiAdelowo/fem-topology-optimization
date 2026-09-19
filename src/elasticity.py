"""2D linear-elasticity FEM solver: -div(sigma(u)) = f, small-strain,
isotropic, plane stress (default) or plane strain.

## Plane stress vs. plane strain

Both are 2D idealisations of 3D linear elasticity, differing in which
out-of-plane assumption is made:

- **Plane stress** assumes the out-of-plane stress sigma_zz = 0 (valid for
  a thin structure, loaded in its own plane, free to contract/expand in the
  thickness direction).
- **Plane strain** assumes the out-of-plane strain epsilon_zz = 0 (valid
  for a structure that is very long/constrained in the out-of-plane
  direction, e.g. a long dam cross-section).

**This project uses plane stress**, the standard choice for topology
optimisation of thin structural components (the classical SIMP cantilever/
MBB-beam benchmarks -- Sigmund's 99-line and 88-line MATLAB codes, and the
broader topology-optimisation literature following Bendsøe & Sigmund, 2003,
*Topology Optimization: Theory, Methods and Applications*, Springer --
model a thin plate loaded in-plane, which is a plane-stress idealisation).

Both are implemented via the effective Lame parameters:

    plane stress:  lambda* = E*nu / (1 - nu^2),      mu = E / (2*(1+nu))
    plane strain:  lambda  = E*nu / ((1+nu)*(1-2nu)), mu = E / (2*(1+nu))

(mu is identical in both cases; only the effective lambda differs.) The
constitutive law is then the standard isotropic form

    sigma(u) = 2*mu*epsilon(u) + lambda*tr(epsilon(u))*I,
    epsilon(u) = (1/2)*(grad(u) + grad(u)^T).

## Weak formulation

Starting from equilibrium -div(sigma(u)) = f in Omega with u = 0 on the
clamped boundary Gamma_D and sigma(u).n = t on the traction (Neumann)
boundary Gamma_N (t = 0 elsewhere on the free boundary): multiply by a
vector test function v in the space V_0 = {v in [H^1(Omega)]^2 : v = 0 on
Gamma_D}, integrate by parts:

    integral_Omega sigma(u):epsilon(v) dx = integral_Omega f.v dx
                                              + integral_{Gamma_N} t.v ds

(sigma(u):epsilon(v) = sigma(u):grad(v) since sigma is symmetric, so the
skew part of grad(v) drops out automatically). The trial space is
u in [H^1_{u_D}(Omega)]^2 (u = u_D, here 0, on Gamma_D); the bilinear form
a(u,v) = integral sigma(u):epsilon(v) dx is exactly TWICE the strain
energy density integrated over the domain when v = u -- a(u,u) = 2*U_strain,
U_strain = (1/2) integral sigma(u):epsilon(u) dx being the physical elastic
strain energy stored in the deformed body. This is discretised with
continuous piecewise-linear (P1) VECTOR Lagrange elements (2 components per
node), the standard, computationally efficient choice for SIMP topology
optimisation, where the elasticity problem is solved many times per
optimisation run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh as dmesh
from dolfinx.fem import Function, dirichletbc, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, set_bc


@dataclass
class ElasticityMesh:
    mesh: object
    V: object  # vector displacement space (P1)
    V0: object  # scalar DG0 space (for density / elementwise fields)
    L: float
    H: float


def build_mesh(nx: int, ny: int, L: float = 2.0, H: float = 1.0) -> ElasticityMesh:
    msh = dmesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([L, H])], (nx, ny),
        cell_type=dmesh.CellType.quadrilateral,
    )
    V = functionspace(msh, ("Lagrange", 1, (2,)))
    V0 = functionspace(msh, ("DG", 0))
    return ElasticityMesh(mesh=msh, V=V, V0=V0, L=L, H=H)


def lame_parameters(E, nu, plane: str = "stress"):
    mu = E / (2.0 * (1.0 + nu))
    if plane == "stress":
        lam = E * nu / (1.0 - nu**2)
    elif plane == "strain":
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    else:
        raise ValueError(f"plane must be 'stress' or 'strain', got {plane!r}")
    return lam, mu


def strain(u):
    return ufl.sym(ufl.grad(u))


def stress(u, lam, mu):
    return 2.0 * mu * strain(u) + lam * ufl.tr(strain(u)) * ufl.Identity(2)


def left_boundary(L: float = 2.0):
    def predicate(x):
        return np.isclose(x[0], 0.0)
    return predicate


def nearest_node_dofs(V, point: tuple[float, float]) -> tuple[int, np.ndarray]:
    """Return (block_index, [scalar_dof_x, scalar_dof_y]) for the mesh node
    geometrically nearest to `point`, for a blocked (vector) space V with
    block_size 2. Used to apply a concentrated point load, which is not
    representable as a UFL surface/volume integral (a true Dirac load is a
    distribution, not an L2 function) -- the standard practical alternative,
    exactly as used in the classical SIMP reference codes (Sigmund, 2001,
    "A 99 line topology optimization code written in Matlab", Struct.
    Multidiscip. Optim. 21, 120-127), is to add the force directly to the
    assembled right-hand-side vector at the nearest node's degrees of
    freedom.
    """
    coords = V.tabulate_dof_coordinates()
    dists = np.sum((coords[:, :2] - np.array(point)) ** 2, axis=1)
    block = int(np.argmin(dists))
    bs = V.dofmap.index_map_bs
    return block, np.array([block * bs, block * bs + 1], dtype=np.int32)


def solve_elasticity(em: ElasticityMesh, E, nu, f=None, point_loads=None, plane: str = "stress",
                      clamped_boundary=None):
    """Solve the linear elasticity problem.

    E, nu: scalars, or UFL expressions (e.g. SIMP-interpolated E(rho)) for
        spatially varying material.
    f: optional UFL body-force expression (default zero).
    point_loads: optional list of ((x, y), (Fx, Fy)) pairs: concentrated
        forces applied at the mesh node nearest each (x, y), added directly
        to the assembled RHS vector (see `nearest_node_dofs`).
    clamped_boundary: geometric predicate for the clamped (Dirichlet, u=0)
        boundary; defaults to the left edge x=0.
    """
    msh = em.mesh
    lam, mu = lame_parameters(E, nu, plane)

    u = ufl.TrialFunction(em.V)
    v = ufl.TestFunction(em.V)
    a_form = fem.form(ufl.inner(stress(u, lam, mu), strain(v)) * ufl.dx)

    if f is None:
        f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    L_form = fem.form(ufl.inner(f, v) * ufl.dx)

    if clamped_boundary is None:
        clamped_boundary = left_boundary(em.L)

    zero = Function(em.V)
    zero.x.array[:] = 0.0
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    facets = dmesh.locate_entities_boundary(msh, fdim, clamped_boundary)
    dofs = locate_dofs_topological(em.V, fdim, facets)
    bc = dirichletbc(zero, dofs)

    A = assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = assemble_vector(L_form)

    if point_loads:
        for point, force in point_loads:
            _, scalar_dofs = nearest_node_dofs(em.V, point)
            for comp, dof in enumerate(scalar_dofs):
                b.array[dof] += force[comp]

    set_bc(b, [bc])

    uh = Function(em.V)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    return uh
