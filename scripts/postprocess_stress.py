"""Evaluate von Mises stress on the FINAL optimized (compliance-minimizing)
structure from scripts/run_topopt.py.

IMPORTANT SCOPE NOTE: this is a POST-HOC evaluation only. The structure was
optimized to minimize compliance under a volume constraint -- it was NOT
optimized to minimize or bound stress in any way. Stress-constrained
topology optimization is a substantially different (and harder) problem: it
requires a stress measure in the objective/constraints (typically an
aggregated p-norm approximation of the pointwise von Mises field, since the
true max-stress constraint is non-differentiable and the local stress
constraints are famously "singular" at low density -- see Duysinx &
Bendsoe, 1998, "Topology optimization of continuum structures with local
stress constraints", Int. J. Numer. Methods Eng. 43), with its own
sensitivity analysis and a different optimizer behaviour entirely. THAT
problem is explicitly OUT OF SCOPE for this project (see README
Limitations). Nothing in this script's output should be described as
"stress-optimized" -- it is a compliance-optimized structure whose stress
field is inspected afterward, nothing more.

Usage: python scripts/postprocess_stress.py [--config configs/local.yaml]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
import yaml
from dolfinx.fem import Function, functionspace, Expression

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.elasticity import build_mesh, lame_parameters
from src.postprocess import von_mises
from src.simp import SimpParams, solve_state
from src.viz import to_regular_grid, plot_density

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "local.yaml"))
    args = ap.parse_args()
    config_tag = pathlib.Path(args.config).stem
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())["canonical"]

    rho_path = RESULTS_DIR / f"topopt_canonical_rho_phys_{config_tag}.npy"
    if not rho_path.exists():
        raise SystemExit(f"Run scripts/run_topopt.py --config {args.config} first to produce "
                          f"{rho_path.relative_to(ROOT)}.")
    rho_phys = np.load(rho_path)

    L, H = cfg["L"], cfg["H"]
    em = build_mesh(cfg["nx"], cfg["ny"], L=L, H=H)
    params = SimpParams(E0=cfg["E0"], Emin=cfg["Emin"], nu=cfg["nu"], p=cfg["p"], Vf=cfg["Vf"],
                         r_min=cfg["r_min"])
    point_loads = [((L, H / 2.0), (0.0, -1.0))]

    u = solve_state(em, rho_phys, params, point_loads)

    lam, mu = lame_parameters(params.E0, params.nu, params.plane)
    V1 = functionspace(em.mesh, ("Lagrange", 1))
    vm = Function(V1)
    vm.interpolate(Expression(von_mises(u, lam, mu), V1.element.interpolation_points))

    # von Mises evaluated with the SOLID material law (E0), but only
    # meaningful where rho_phys is close to 1 -- in low-density ("void")
    # regions the physical stress is not well-defined by this SIMP
    # post-processing (a well-known limitation of naive SIMP stress
    # evaluation, see e.g. Le et al. 2010, "Stress-based topology
    # optimization for continua", Struct. Multidiscip. Optim. 41). We mask
    # out elements with rho_phys < 0.5 in the reported max/figure to avoid
    # reporting a physically meaningless stress spike in near-void material.
    node_x, node_y = V1.tabulate_dof_coordinates()[:, 0], V1.tabulate_dof_coordinates()[:, 1]

    # nearest-neighbour density at each displacement-space node, to build the mask
    from scipy.interpolate import griddata
    from src.simp import cell_centers
    centers = cell_centers(em)
    rho_at_nodes = griddata(centers, rho_phys, np.column_stack([node_x, node_y]), method="nearest")
    solid_mask = rho_at_nodes >= 0.5

    vm_solid = vm.x.array[solid_mask]
    max_vm_solid = float(vm_solid.max())
    max_vm_everywhere = float(vm.x.array.max())

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    im0 = plot_density(axes[0], em, rho_phys, L, H)
    axes[0].set_title("compliance-optimized density (from scripts/run_topopt.py)")
    fig.colorbar(im0, ax=axes[0])

    vm_masked = np.where(solid_mask, vm.x.array, np.nan)
    Xi, Yi, Zi = to_regular_grid(node_x, node_y, vm_masked, L, H, method="linear")
    im1 = axes[1].pcolormesh(Xi, Yi, Zi, shading="auto", cmap="inferno")
    axes[1].set_aspect("equal")
    axes[1].set_title("von Mises stress, post-hoc, solid (rho>=0.5) material only", fontsize=10)
    fig.colorbar(im1, ax=axes[1])

    for ax in axes:
        ax.set_ylabel("y")
    axes[-1].set_xlabel("x")
    fig.suptitle("Post-hoc stress evaluation of a COMPLIANCE-minimizing SIMP design")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"topopt_stress_postprocess_{config_tag}.png", dpi=150)

    lines = [
        "Post-hoc von Mises stress evaluation of the compliance-optimized cantilever "
        "(scripts/run_topopt.py output).",
        "This structure minimizes COMPLIANCE under a volume constraint; it was NOT "
        "stress-constrained or stress-optimized in any way. Stress-constrained topology "
        "optimization (Duysinx & Bendsoe 1998) is out of scope for this project (see README).",
        "",
        f"max von Mises stress (solid material, rho_phys >= 0.5): {max_vm_solid:.6f}",
        f"max von Mises stress (everywhere, including near-void elements -- "
        f"NOT physically meaningful, reported only for completeness): {max_vm_everywhere:.6f}",
    ]
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"topopt_stress_postprocess_{config_tag}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    print(f"\nWrote figures/topopt_stress_postprocess_{config_tag}.png and "
          f"results/topopt_stress_postprocess_{config_tag}.txt")


if __name__ == "__main__":
    main()
