"""Manufactured solution for verifying the linear-elasticity solver.

A smooth, non-polynomial exact displacement field is chosen, and the body
force required to make it an exact solution of -div(sigma(u)) = f is
obtained by UFL's own symbolic differentiation (no external CAS, no
hand-derived algebra -- the same approach used throughout the companion
`navier-stokes-2d`, `fem-cylinder-flow`, and `darcy-inverse-problem`
projects).
"""
from __future__ import annotations

import ufl

from src.elasticity import stress, strain


def exact_fields(x, lam, mu):
    """x: ufl.SpatialCoordinate. Returns (u_exact, f) as UFL expressions,
    for the unit square [0,1]^2 with u_exact = 0 on the whole boundary
    (so it can be used directly as a homogeneous-Dirichlet manufactured
    problem, independent of the cantilever's specific BCs)."""
    u1 = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u2 = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    u_exact = ufl.as_vector((u1, u2))
    sigma_exact = stress(u_exact, lam, mu)
    f = -ufl.div(sigma_exact)
    return u_exact, f
