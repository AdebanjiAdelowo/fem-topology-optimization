"""Physical cantilever baseline: a rectangular domain clamped on the left
edge, a downward point load at the mid-height of the free (right) edge.
Displacement, von Mises stress, and strain-energy density fields, plus a
physical-scaling sanity check (displacement should be exactly linear in
1/E and in the load magnitude, for linear elasticity).

Usage: python examples/cantilever_baseline.py
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
from scipy.interpolate import griddata
from dolfinx import fem
from dolfinx.fem import Function, functionspace, Expression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.elasticity import build_mesh, lame_parameters, solve_elasticity, nearest_node_dofs
from src.postprocess import von_mises, strain_energy_density

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

L, H = 2.0, 1.0
E0, NU = 1.0, 0.3
LOAD = (0.0, -1.0)
LOAD_POINT = (L, H / 2.0)


def to_regular_grid(x, y, values, L, H, nx=300, ny=150):
    """Interpolate scattered (x, y, values) onto a regular grid for clean
    pcolormesh rendering (avoids triangulation-rendering artefacts seen with
    tricontourf/tripcolor on a structured quad-derived triangulation)."""
    xi = np.linspace(0, L, nx)
    yi = np.linspace(0, H, ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), values, (Xi, Yi), method="linear")
    return Xi, Yi, Zi


def main() -> None:
    em = build_mesh(nx=80, ny=40, L=L, H=H)
    lam, mu = lame_parameters(E0, NU, "stress")

    uh = solve_elasticity(em, E0, NU, point_loads=[(LOAD_POINT, LOAD)])

    V1 = functionspace(em.mesh, ("Lagrange", 1))
    speed = Function(V1)
    speed_expr = (uh[0] ** 2 + uh[1] ** 2) ** 0.5
    speed.interpolate(Expression(speed_expr, V1.element.interpolation_points))

    vm = Function(V1)
    vm.interpolate(Expression(von_mises(uh, lam, mu), V1.element.interpolation_points))

    V0 = em.V0
    energy = Function(V0)
    energy.interpolate(Expression(strain_energy_density(uh, lam, mu), V0.element.interpolation_points))

    node_x, node_y = V1.tabulate_dof_coordinates()[:, 0], V1.tabulate_dof_coordinates()[:, 1]
    x_cells = em.mesh.geometry.x
    cell_centers = np.array([x_cells[c].mean(axis=0) for c in em.mesh.geometry.dofmap.reshape(-1, 4)])

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    Xi, Yi, Zi = to_regular_grid(node_x, node_y, speed.x.array, L, H)
    im0 = axes[0].pcolormesh(Xi, Yi, Zi, shading="auto", cmap="viridis")
    axes[0].set_title("displacement magnitude $|u|$")
    fig.colorbar(im0, ax=axes[0])

    Xi, Yi, Zi = to_regular_grid(node_x, node_y, vm.x.array, L, H)
    im1 = axes[1].pcolormesh(Xi, Yi, Zi, shading="auto", cmap="inferno")
    axes[1].set_title(r"von Mises stress $\sigma_{vm}$")
    fig.colorbar(im1, ax=axes[1])

    Xi, Yi, Zi = to_regular_grid(cell_centers[:, 0], cell_centers[:, 1], energy.x.array, L, H)
    im2 = axes[2].pcolormesh(Xi, Yi, Zi, shading="auto", cmap="magma")
    axes[2].set_title("strain-energy density")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_ylabel("y")
    axes[-1].set_xlabel("x")
    fig.suptitle(f"Cantilever baseline (E={E0}, nu={NU}, point load {LOAD} at {LOAD_POINT})")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "cantilever_baseline.png", dpi=150)

    # physical scaling checks
    _, dofs = nearest_node_dofs(em.V, LOAD_POINT)
    tip_defl_baseline = uh.x.array[dofs[1]]

    uh_2E = solve_elasticity(em, 2 * E0, NU, point_loads=[(LOAD_POINT, LOAD)])
    tip_defl_2E = uh_2E.x.array[dofs[1]]

    uh_2F = solve_elasticity(em, E0, NU, point_loads=[(LOAD_POINT, (0.0, -2.0))])
    tip_defl_2F = uh_2F.x.array[dofs[1]]

    total_strain_energy = fem.assemble_scalar(fem.form(strain_energy_density(uh, lam, mu) * ufl.dx))
    lines = [
        f"Cantilever baseline: L={L}, H={H}, E={E0}, nu={NU}, load={LOAD} at {LOAD_POINT}",
        f"max |u| = {speed.x.array.max():.6f}",
        f"max von Mises stress = {vm.x.array.max():.6f}",
        f"total strain energy (integral over domain) = {total_strain_energy:.6f}",
        "",
        "Physical scaling checks (exact linearity expected for linear elasticity):",
        f"  tip deflection (v-component) at baseline (E={E0}, F=1): {tip_defl_baseline:.6f}",
        f"  tip deflection at E={2*E0} (F=1): {tip_defl_2E:.6f}  "
        f"(expect exactly {tip_defl_baseline/2:.6f}; ratio={tip_defl_2E/tip_defl_baseline:.6f})",
        f"  tip deflection at E={E0} (F=2): {tip_defl_2F:.6f}  "
        f"(expect exactly {tip_defl_baseline*2:.6f}; ratio={tip_defl_2F/tip_defl_baseline:.6f})",
    ]
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "cantilever_baseline.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    print(f"\nWrote figures/cantilever_baseline.png and results/cantilever_baseline.txt")


if __name__ == "__main__":
    main()
