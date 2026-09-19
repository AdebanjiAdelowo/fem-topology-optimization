"""Canonical SIMP topology-optimization benchmark: a cantilever clamped on
its left edge with a downward point load at the mid-height of the free
(right) edge -- the same domain, boundary conditions and load case as the
Part I physical baseline (examples/cantilever_baseline.py), so the
optimized structure can be compared against the solid baseline directly.

Saves: the initial (uniform) and final (optimized) density fields, the
compliance/volume-fraction/max-design-change iteration histories, and a
results text file with the final numbers and the grayness (discreteness)
metric.

Usage: python scripts/run_topopt.py [--config configs/local.yaml]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.elasticity import build_mesh
from src.simp import SimpParams, run_simp, grayness
from src.viz import plot_density

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "local.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())["canonical"]

    L, H = cfg["L"], cfg["H"]
    em = build_mesh(cfg["nx"], cfg["ny"], L=L, H=H)
    params = SimpParams(E0=cfg["E0"], Emin=cfg["Emin"], nu=cfg["nu"], p=cfg["p"], Vf=cfg["Vf"],
                         r_min=cfg["r_min"], move=cfg["move"])
    point_loads = [((L, H / 2.0), (0.0, -1.0))]

    res = run_simp(em, params, point_loads, max_iter=cfg["max_iter"], change_tol=cfg["change_tol"])

    # --- density fields: initial (uniform Vf) vs. final ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    rho_init = np.full_like(res.rho_phys, params.Vf)
    im0 = plot_density(axes[0], em, rho_init, L, H)
    axes[0].set_title(f"initial density (uniform, Vf={params.Vf})")
    fig.colorbar(im0, ax=axes[0])
    im1 = plot_density(axes[1], em, res.rho_phys, L, H)
    axes[1].set_title(f"optimized density (iteration {res.n_iter}, converged={res.converged})")
    fig.colorbar(im1, ax=axes[1])
    for ax in axes:
        ax.set_ylabel("y")
    axes[-1].set_xlabel("x")
    fig.suptitle(f"SIMP cantilever benchmark: {cfg['nx']}x{cfg['ny']} mesh, p={params.p}, "
                 f"r_min={params.r_min}, Vf={params.Vf}")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "topopt_canonical_density.png", dpi=150)

    # --- convergence histories ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    it = np.arange(1, res.n_iter + 1)
    axes2[0].plot(it, res.history["compliance"])
    axes2[0].set_ylabel("compliance")
    axes2[1].plot(it, res.history["volume_fraction"])
    axes2[1].axhline(params.Vf, color="k", linestyle="--", alpha=0.5, label="target Vf")
    axes2[1].set_ylabel("volume fraction")
    axes2[1].legend(fontsize=8)
    axes2[2].semilogy(it, res.history["max_change"])
    axes2[2].axhline(cfg["change_tol"], color="k", linestyle="--", alpha=0.5, label="change_tol")
    axes2[2].set_ylabel("max design change")
    axes2[2].set_xlabel("iteration")
    axes2[2].legend(fontsize=8)
    fig2.suptitle("SIMP optimization convergence history")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "topopt_canonical_history.png", dpi=150)

    lines = [
        f"Canonical SIMP cantilever benchmark: mesh {cfg['nx']}x{cfg['ny']} "
        f"({em.V0.dofmap.index_map.size_local} elements), L={L}, H={H}",
        f"E0={params.E0}, Emin={params.Emin}, nu={params.nu}, p={params.p}, Vf={params.Vf}, "
        f"r_min={params.r_min}, move={params.move}",
        f"point load (0,-1) at ({L}, {H/2})",
        "",
        f"iterations run: {res.n_iter} (max_iter={cfg['max_iter']})",
        f"change_tol reached (converged by design-change criterion): {res.converged}",
        f"final compliance: {res.history['compliance'][-1]:.6f}",
        f"initial (uniform-density) compliance: {res.history['compliance'][0]:.6f}",
        f"final volume fraction: {res.history['volume_fraction'][-1]:.6f} (target {params.Vf})",
        f"final max design change: {res.history['max_change'][-1]:.6f}",
        f"grayness (discreteness) metric: {grayness(res.rho_phys):.6f}  "
        "(0 = fully 0/1 design, 1 = every element at rho=0.5)",
    ]
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "topopt_canonical.txt").write_text("\n".join(lines) + "\n")
    np.save(RESULTS_DIR / "topopt_canonical_rho_phys.npy", res.rho_phys)
    for line in lines:
        print(line)
    print("\nWrote figures/topopt_canonical_density.png, "
          "figures/topopt_canonical_history.png, results/topopt_canonical.txt, "
          "results/topopt_canonical_rho_phys.npy")


if __name__ == "__main__":
    main()
