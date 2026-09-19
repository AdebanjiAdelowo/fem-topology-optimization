"""Volume-fraction study: run the canonical cantilever SIMP benchmark at
several target volume fractions Vf, holding mesh, penalization and filter
radius fixed. A lower Vf forces a sparser, more strongly braced structure
(higher compliance, since less material is available); this study reports
that trade-off quantitatively and shows the resulting topologies.

Usage: python scripts/volume_fraction_study.py [--config configs/local.yaml]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())["volume_fraction_study"]

    L, H = cfg["L"], cfg["H"]
    vfs = cfg["volume_fractions"]
    point_loads = [((L, H / 2.0), (0.0, -1.0))]

    fig, axes = plt.subplots(len(vfs), 1, figsize=(9, 2.4 * len(vfs)), sharex=True)
    if len(vfs) == 1:
        axes = [axes]

    lines = [f"Volume-fraction study: mesh {cfg['nx']}x{cfg['ny']}, p={cfg['p']}, "
             f"r_min={cfg['r_min']}", "",
             f"{'Vf':>6} {'iters':>6} {'compliance':>12} {'grayness':>9}"]

    for i, vf in enumerate(vfs):
        em = build_mesh(cfg["nx"], cfg["ny"], L=L, H=H)
        params = SimpParams(E0=1.0, Emin=1e-9, nu=0.3, p=cfg["p"], Vf=vf, r_min=cfg["r_min"])
        res = run_simp(em, params, point_loads, max_iter=cfg["max_iter"], change_tol=cfg["change_tol"])
        c_final = res.history["compliance"][-1]
        g = grayness(res.rho_phys)
        lines.append(f"{vf:6.2f} {res.n_iter:6d} {c_final:12.4f} {g:9.4f}")

        im = plot_density(axes[i], em, res.rho_phys, L, H)
        axes[i].set_title(f"Vf={vf}, compliance={c_final:.2f}")
        fig.colorbar(im, ax=axes[i])
        axes[i].set_ylabel("y")

    axes[-1].set_xlabel("x")
    fig.suptitle("Volume-fraction study")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "volume_fraction_study.png", dpi=150)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "volume_fraction_study.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    print("\nWrote figures/volume_fraction_study.png and results/volume_fraction_study.txt")


if __name__ == "__main__":
    main()
