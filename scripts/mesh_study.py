"""Mesh-resolution study: run the canonical cantilever SIMP benchmark at
several mesh resolutions, holding the PHYSICAL filter radius r_min fixed in
absolute length units (r_min = r_min_per_h * h, h the element size) so mesh
independence of the optimized topology can be honestly assessed --
otherwise refining the mesh while holding r_min in element-count units
would trivially change the physical filter length and confound the
resolution effect with a filter effect (a common pitfall, see Sigmund &
Petersson 1998; Bourdin 2001).

Reports compliance, iteration count, wall-clock runtime, and grayness at
each resolution, and saves a density-field figure per resolution so mesh
dependence (or its absence) can be inspected visually.

Usage: python scripts/mesh_study.py [--config configs/local.yaml]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

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
    config_tag = pathlib.Path(args.config).stem
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())["mesh_study"]

    L, H = cfg["L"], cfg["H"]
    resolutions = cfg["resolutions"]

    fig, axes = plt.subplots(len(resolutions), 1, figsize=(9, 2.4 * len(resolutions)), sharex=True)
    if len(resolutions) == 1:
        axes = [axes]

    lines = [f"Mesh-resolution study: Vf={cfg['Vf']}, p={cfg['p']}, "
             f"r_min = {cfg['r_min_per_h']} * h (h = L/nx)", "",
             f"{'nx':>4} {'ny':>4} {'h':>8} {'r_min':>8} {'n_elem':>7} {'iters':>6} "
             f"{'time(s)':>8} {'compliance':>12} {'grayness':>9}"]

    for i, (nx, ny) in enumerate(resolutions):
        h = L / nx
        r_min = cfg["r_min_per_h"] * h
        em = build_mesh(nx, ny, L=L, H=H)
        params = SimpParams(E0=1.0, Emin=1e-9, nu=0.3, p=cfg["p"], Vf=cfg["Vf"], r_min=r_min)
        point_loads = [((L, H / 2.0), (0.0, -1.0))]

        t0 = time.time()
        res = run_simp(em, params, point_loads, max_iter=cfg["max_iter"], change_tol=cfg["change_tol"])
        dt = time.time() - t0

        n_elem = em.V0.dofmap.index_map.size_local
        c_final = res.history["compliance"][-1]
        g = grayness(res.rho_phys)
        lines.append(f"{nx:4d} {ny:4d} {h:8.4f} {r_min:8.4f} {n_elem:7d} {res.n_iter:6d} "
                      f"{dt:8.2f} {c_final:12.4f} {g:9.4f}")

        im = plot_density(axes[i], em, res.rho_phys, L, H)
        axes[i].set_title(f"{nx}x{ny} ({n_elem} elem), r_min={r_min:.4f}, compliance={c_final:.2f}")
        fig.colorbar(im, ax=axes[i])
        axes[i].set_ylabel("y")

    axes[-1].set_xlabel("x")
    fig.suptitle("Mesh-resolution study (r_min held fixed in absolute length units)")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"mesh_study_{config_tag}.png", dpi=150)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"mesh_study_{config_tag}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    print(f"\nWrote figures/mesh_study_{config_tag}.png and results/mesh_study_{config_tag}.txt")


if __name__ == "__main__":
    main()
