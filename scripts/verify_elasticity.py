"""Verify the linear-elasticity solver against a manufactured solution and
perform a mesh-refinement convergence study.

P1 vector elements on a smooth elasticity problem are expected to give
||u-u_h||_{L2} = O(h^2) and ||u-u_h||_{H1} = O(h) (standard elliptic-system
FEM a priori estimates for P1 elements, e.g. Braess, "Finite Elements",
Ch. II; the L2 rate is one order higher than H1 via the Aubin-Nitsche
duality argument, exactly as in the companion `darcy-inverse-problem`
project's forward-solver verification).

Usage: python scripts/verify_elasticity.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import Function, dirichletbc, locate_dofs_topological
from dolfinx import mesh as dmesh
from mpi4py import MPI

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.elasticity import build_mesh, lame_parameters, stress, strain, solve_elasticity
from src.manufactured import exact_fields

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

E, NU = 1.0, 0.3


def run_case(n: int):
    em = build_mesh(n, n, L=1.0, H=1.0)
    lam, mu = lame_parameters(E, NU, "stress")
    x = ufl.SpatialCoordinate(em.mesh)
    u_exact, f = exact_fields(x, lam, mu)

    def whole_boundary(x):
        return np.full(x.shape[1], True)

    u_h = solve_elasticity(em, E, NU, f=f, clamped_boundary=whole_boundary)

    e_l2 = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u_h - u_exact, u_h - u_exact) * ufl.dx)))
    e_h1 = np.sqrt(
        fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_h - u_exact), ufl.grad(u_h - u_exact)) * ufl.dx))
    )
    return 1.0 / n, e_l2, e_h1


def main() -> None:
    Ns = [4, 8, 16, 32, 64]
    hs, e_l2s, e_h1s = [], [], []
    lines = [f"{'N':>4} {'h':>8} {'L2 error':>12} {'order':>7} {'H1 error':>12} {'order':>7}"]
    for N in Ns:
        h, e_l2, e_h1 = run_case(N)
        hs.append(h); e_l2s.append(e_l2); e_h1s.append(e_h1)

    for i, N in enumerate(Ns):
        if i == 0:
            o_l2 = o_h1 = float("nan")
        else:
            o_l2 = np.log(e_l2s[i - 1] / e_l2s[i]) / np.log(hs[i - 1] / hs[i])
            o_h1 = np.log(e_h1s[i - 1] / e_h1s[i]) / np.log(hs[i - 1] / hs[i])
        line = f"{N:4d} {hs[i]:8.4f} {e_l2s[i]:12.4e} {o_l2:7.3f} {e_h1s[i]:12.4e} {o_h1:7.3f}"
        print(line)
        lines.append(line)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "elasticity_verification.txt").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(hs, e_l2s, "o-", label=r"$\|u-u_h\|_{L^2}$ (expect $O(h^2)$)")
    ax.loglog(hs, e_h1s, "s-", label=r"$\|u-u_h\|_{H^1}$ (expect $O(h^1)$)")
    ax.loglog(hs, e_l2s[0] * (np.array(hs) / hs[0]) ** 2, "k--", alpha=0.5, label=r"$O(h^2)$ ref.")
    ax.loglog(hs, e_h1s[0] * (np.array(hs) / hs[0]) ** 1, "k:", alpha=0.5, label=r"$O(h^1)$ ref.")
    ax.set_xlabel("mesh size $h$")
    ax.set_ylabel("error")
    ax.set_title("Linear elasticity: manufactured-solution convergence")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "elasticity_convergence.png", dpi=150)
    print(f"\nWrote results/elasticity_verification.txt and figures/elasticity_convergence.png")


if __name__ == "__main__":
    main()
