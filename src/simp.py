"""SIMP (Solid Isotropic Material with Penalization) topology optimization
for minimum-compliance design under a volume constraint.

Given a fixed rectangular design domain, boundary conditions and applied
load, this module minimizes structural compliance

    c(rho) = f^T u(rho),   subject to   K(rho) u(rho) = f,
                                         mean(rho) <= Vf,   0 < rho_min <= rho <= 1

using the classical SIMP material interpolation (Bendsoe, 1989; Bendsoe &
Sigmund, 1999, "Material interpolation schemes in topology optimization",
Arch. Appl. Mech. 69):

    E(rho) = Emin + rho^p * (E0 - Emin),   0 <= rho <= 1

Emin > 0 (rather than 0) avoids singularity of the stiffness matrix for
void elements while remaining physically negligible; Emin = 1e-9 * E0 is
the standard default (Sigmund, 2001, "A 99 line topology optimization code
written in Matlab", Struct. Multidiscip. Optim. 21; Andreassen et al.,
2011, "Efficient topology optimization in MATLAB using 88 lines of code",
Struct. Multidiscip. Optim. 43). p > 1 penalizes intermediate ("grey")
densities relative to a linear (p=1) rule-of-mixtures interpolation,
pushing the optimizer toward a near-0/1 design; p=3 is the standard
default.

## Reusing the verified elasticity solver

Since lambda(E) and mu(E) (src/elasticity.py:lame_parameters) are both
linear in E at fixed Poisson ratio nu, sigma(u; E) = E * sigma_unit(u)
where sigma_unit uses the E=1 Lame parameters. The elastic bilinear form
therefore factors as

    a(u, v; rho) = integral E(rho) * sigma_unit(u):epsilon(v) dx,

so `solve_elasticity` (already independently verified via manufactured
solution and physical scaling checks, see scripts/verify_elasticity.py and
examples/cantilever_baseline.py) can be reused UNCHANGED for the
topology-optimization state solve: E is simply passed as a UFL expression
built from the elementwise (DG0) density field instead of a constant.

## Compliance sensitivity (self-adjoint)

c(rho) = f^T u(rho), K(rho) u(rho) = f, f independent of rho. Implicit
differentiation of the equilibrium constraint gives K du/drho_e =
-(dK/drho_e) u, so

    dc/drho_e = f^T du/drho_e = -u^T K du/drho_e ... = -u^T (dK/drho_e) u

i.e. the "adjoint" state for compliance is simply lambda = -u -- no extra
linear solve is needed (Bendsoe & Sigmund, 2003, *Topology Optimization:
Theory, Methods and Applications*, Sec. 1.3). Since K_e(rho_e) = E(rho_e) *
k0_e, with k0_e the elemental stiffness built with the UNIT-E constitutive
law,

    dK_e/drho_e = p * rho_e^(p-1) * (E0 - Emin) * k0_e
    dc/drho_e   = -p * rho_e^(p-1) * (E0 - Emin) * w_e,

where w_e = u_e^T k0_e u_e = integral_{Omega_e} sigma_unit(u):epsilon(u) dx
is the elemental bilinear-form value computed with the UNIT-E material
(NOT the physical elemental strain energy -- see `elemental_unit_energy`).
This sensitivity is verified independently against a finite-difference
(Taylor-remainder) check in tests/test_simp.py and
scripts/verify_sensitivity.py before being trusted for optimization.

## Density filtering

Unfiltered SIMP optimization is well known to produce mesh-dependent,
checkerboard-patterned designs (an artefact of the low-order FE
discretization, not a physical feature -- Sigmund & Petersson, 1998,
"Numerical instabilities in topology optimization", Struct. Optim. 16).
This module implements the density (Bourdin, 2001, "Filters in topology
optimization", Int. J. Numer. Methods Eng. 50) rather than sensitivity
(Sigmund, 1997) filter: the OPTIMIZATION VARIABLE rho is mapped to a
"physical" density rho_phys via a linear cone-weighted local average before
it is used in the state equation or the objective,

    rho_phys_e = sum_i H_ei rho_i / sum_i H_ei,   H_ei = max(0, r_min - |x_e - x_i|),

and the objective/constraint sensitivities with respect to rho are obtained
by the exact chain rule through this linear map,
d(.)/drho = H^T (d(.)/drho_phys / rowsum(H)) -- this is what is meant by
"the filter is used in the optimization loop", as opposed to a cosmetic
post-hoc smoothing of the final image.

## Optimizer

A standard optimality-criteria (OC) update with bisection on the Lagrange
multiplier for the volume constraint (Bendsoe, 1995; Sigmund, 2001;
Andreassen et al., 2011):

    rho_new_e = clip( rho_e * sqrt(-dc/drho_e / (dv/drho_e * lambda)),
                       rho_e - move, rho_e + move ), further clipped to
                       [rho_min, 1],

with lambda found by bisection so the volume constraint is (near-)active.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import Function
from dolfinx.fem.petsc import assemble_vector
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

from src.elasticity import ElasticityMesh, lame_parameters, solve_elasticity, stress, strain


@dataclass(frozen=True)
class SimpParams:
    E0: float = 1.0
    Emin: float = 1e-9
    nu: float = 0.3
    p: float = 3.0
    Vf: float = 0.4
    r_min: float = 0.075
    move: float = 0.2
    rho_min: float = 1e-3
    plane: str = "stress"


def cell_centers(em: ElasticityMesh) -> np.ndarray:
    x = em.mesh.geometry.x
    cells = em.mesh.geometry.dofmap.reshape(-1, 4)
    return np.array([x[c, :2].mean(axis=0) for c in cells])


def elemental_area(em: ElasticityMesh) -> np.ndarray:
    """integral_{Omega_e} 1 dx for each element, via a DG0 test-function
    assembly (the DG0 basis function is the indicator of its cell, so this
    picks out exactly the cell area/volume)."""
    v0 = ufl.TestFunction(em.V0)
    vec = assemble_vector(fem.form(v0 * ufl.dx))
    return vec.array.copy()


def elemental_unit_energy(em: ElasticityMesh, u, nu: float, plane: str = "stress") -> np.ndarray:
    """w_e = integral_{Omega_e} sigma_unit(u):epsilon(u) dx (E=1 material),
    the quantity the SIMP compliance sensitivity is built from (see module
    docstring)."""
    lam1, mu1 = lame_parameters(1.0, nu, plane)
    v0 = ufl.TestFunction(em.V0)
    vec = assemble_vector(fem.form(v0 * ufl.inner(stress(u, lam1, mu1), strain(u)) * ufl.dx))
    return vec.array.copy()


def build_filter(em: ElasticityMesh, r_min: float) -> tuple[csr_matrix, np.ndarray]:
    """Sigmund/Bourdin linear cone density filter. Returns (H, Hs) with Hs
    the row sums of H (so rho_phys = (H @ rho) / Hs)."""
    centers = cell_centers(em)
    n = centers.shape[0]
    tree = cKDTree(centers)
    rows, cols, vals = [], [], []
    for e in range(n):
        for i in tree.query_ball_point(centers[e], r_min):
            w = r_min - np.linalg.norm(centers[e] - centers[i])
            if w > 0:
                rows.append(e)
                cols.append(i)
                vals.append(w)
    H = csr_matrix((vals, (rows, cols)), shape=(n, n))
    Hs = np.asarray(H.sum(axis=1)).flatten()
    return H, Hs


def apply_filter(rho: np.ndarray, H: csr_matrix, Hs: np.ndarray) -> np.ndarray:
    return (H @ rho) / Hs


def filter_sensitivity(d_phys: np.ndarray, H: csr_matrix, Hs: np.ndarray) -> np.ndarray:
    return H.T @ (d_phys / Hs)


def solve_state(em: ElasticityMesh, rho_phys: np.ndarray, params: SimpParams, point_loads,
                 clamped_boundary=None):
    """Solve the elasticity state equation with SIMP-interpolated material.
    Reuses `solve_elasticity` unchanged, passing E(rho) as a UFL expression
    built from a DG0 density Function."""
    rho_fun = Function(em.V0)
    rho_fun.x.array[:] = rho_phys
    E_expr = params.Emin + rho_fun**params.p * (params.E0 - params.Emin)
    u = solve_elasticity(em, E_expr, params.nu, point_loads=point_loads, plane=params.plane,
                          clamped_boundary=clamped_boundary)
    return u


def compliance_and_sensitivity(rho_phys: np.ndarray, w_e: np.ndarray, elem_area: np.ndarray,
                                params: SimpParams):
    """Compliance c(rho_phys) = sum_e E(rho_phys_e) * w_e (= u^T K u = f^T u,
    see module docstring) and its raw (unfiltered, w.r.t. rho_phys)
    sensitivity dc/drho_phys_e, plus the volume-fraction-constraint
    sensitivity dv/drho_phys_e = area_e / sum(area)."""
    E0, Emin, p = params.E0, params.Emin, params.p
    E_e = Emin + rho_phys**p * (E0 - Emin)
    c = float(np.sum(E_e * w_e))
    dc_phys = -p * rho_phys ** (p - 1) * (E0 - Emin) * w_e
    dv_phys = elem_area / elem_area.sum()
    return c, dc_phys, dv_phys


def oc_update(rho: np.ndarray, dc: np.ndarray, dv: np.ndarray, Vf: float, elem_area: np.ndarray,
              move: float = 0.2, rho_min: float = 1e-3, rho_max: float = 1.0,
              bisection_tol: float = 1e-6) -> np.ndarray:
    """Optimality-criteria update with bisection on the volume-constraint
    Lagrange multiplier lambda (Sigmund, 2001; Andreassen et al., 2011)."""
    l1, l2 = 1e-9, 1e9
    total_area = elem_area.sum()
    target = Vf * total_area
    rho_new = rho.copy()
    while (l2 - l1) / (l1 + l2) > bisection_tol:
        lmid = 0.5 * (l1 + l2)
        B = np.maximum(-dc / (dv * lmid), 0.0)
        cand = rho * np.sqrt(B)
        lower = np.maximum(rho_min, rho - move)
        upper = np.minimum(rho_max, rho + move)
        rho_new = np.clip(cand, lower, upper)
        if np.sum(rho_new * elem_area) > target:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new


@dataclass
class SimpResult:
    rho: np.ndarray
    rho_phys: np.ndarray
    history: dict
    n_iter: int
    converged: bool


def compliance_only(em: ElasticityMesh, rho: np.ndarray, params: SimpParams, point_loads,
                     H: csr_matrix, Hs: np.ndarray, elem_area: np.ndarray,
                     clamped_boundary=None) -> float:
    """Compliance for a raw (pre-filter) design vector rho, used by the
    Taylor-remainder sensitivity check."""
    rho_phys = apply_filter(rho, H, Hs)
    u = solve_state(em, rho_phys, params, point_loads, clamped_boundary)
    w_e = elemental_unit_energy(em, u, params.nu, params.plane)
    c, _, _ = compliance_and_sensitivity(rho_phys, w_e, elem_area, params)
    return c


def compliance_gradient(em: ElasticityMesh, rho: np.ndarray, params: SimpParams, point_loads,
                         H: csr_matrix, Hs: np.ndarray, elem_area: np.ndarray,
                         clamped_boundary=None):
    """Compliance value and full gradient dc/drho (w.r.t. the raw design
    variable, filtered via the exact chain rule) at a given design rho."""
    rho_phys = apply_filter(rho, H, Hs)
    u = solve_state(em, rho_phys, params, point_loads, clamped_boundary)
    w_e = elemental_unit_energy(em, u, params.nu, params.plane)
    c, dc_phys, _ = compliance_and_sensitivity(rho_phys, w_e, elem_area, params)
    dc = filter_sensitivity(dc_phys, H, Hs)
    return c, dc


def run_simp(em: ElasticityMesh, params: SimpParams, point_loads, clamped_boundary=None,
             max_iter: int = 100, change_tol: float = 1e-3, rho_init: np.ndarray | None = None,
             callback=None) -> SimpResult:
    """Run the SIMP optimization loop: filter -> solve state -> compliance +
    sensitivity -> filter sensitivity -> OC update, until the maximum
    elementwise design change drops below `change_tol` or `max_iter` is
    reached."""
    n = em.V0.dofmap.index_map.size_local
    rho = np.full(n, params.Vf) if rho_init is None else rho_init.copy()
    H, Hs = build_filter(em, params.r_min)
    elem_area = elemental_area(em)
    total_area = elem_area.sum()

    history = {"compliance": [], "volume_fraction": [], "max_change": []}
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        rho_phys = apply_filter(rho, H, Hs)
        u = solve_state(em, rho_phys, params, point_loads, clamped_boundary)
        w_e = elemental_unit_energy(em, u, params.nu, params.plane)
        c, dc_phys, dv_phys = compliance_and_sensitivity(rho_phys, w_e, elem_area, params)
        dc = filter_sensitivity(dc_phys, H, Hs)
        dv = filter_sensitivity(dv_phys, H, Hs)

        rho_new = oc_update(rho, dc, dv, params.Vf, elem_area, move=params.move,
                             rho_min=params.rho_min)
        change = float(np.max(np.abs(rho_new - rho)))
        vf_actual = float(np.sum(rho_phys * elem_area) / total_area)

        history["compliance"].append(c)
        history["volume_fraction"].append(vf_actual)
        history["max_change"].append(change)
        if callback is not None:
            callback(it, rho, rho_phys, c, vf_actual, change)

        rho = rho_new
        if change < change_tol:
            converged = True
            break

    rho_phys_final = apply_filter(rho, H, Hs)
    return SimpResult(rho=rho, rho_phys=rho_phys_final, history=history, n_iter=it,
                       converged=converged)


def grayness(rho_phys: np.ndarray) -> float:
    """Fraction-like grayness metric M_nd = sum(4*rho*(1-rho))/n in [0, 1]:
    0 for a fully 0/1 ("black and white") design, 1 if every element sits
    at rho=0.5 (Sigmund & Petersson-style discreteness metric, as used e.g.
    in Andreassen et al. 2011's 88-line code, "grey level" Mnd)."""
    return float(np.mean(4.0 * rho_phys * (1.0 - rho_phys)))
