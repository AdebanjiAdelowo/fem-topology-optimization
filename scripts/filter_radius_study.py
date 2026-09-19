"""Filter-radius study: run the canonical cantilever SIMP benchmark at
several density-filter radii r_min, holding mesh, Vf and p fixed. A larger
r_min enforces a larger minimum member width (more diffuse, more
manufacturable-looking members); a smaller r_min allows thinner members and
more geometric detail but with more risk of mesh-scale artefacts.

Usage: python scripts/filter_radius_study.py [--config configs/local.yaml]
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
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())["filter_radius_study"]

    L, H = cfg["L"], cfg["H"]
    r_mins = cfg["filter_radii"]
    point_loads = [((L, H / 2.0), (0.0, -1.0))]

    fig, axes = plt.subplots(len(r_mins), 1, figsize=(9, 2.4 * len(r_mins)), sharex=True)
    if len(r_mins) == 1:
        axes = [axes]

    lines = [f"Filter-radius study: mesh {cfg['nx']}x{cfg['ny']}, Vf={cfg['Vf']}, p={cfg['p']}", "",
             f"{'r_min':>7} {'iters':>6} {'compliance':>12} {'grayness':>9}"]

    for i, r_min in enumerate(r_mins):
        em = build_mesh(cfg["nx"], cfg["ny"], L=L, H=H)
        params = SimpParams(E0=1.0, Emin=1e-9, nu=0.3, p=cfg["p"], Vf=cfg["Vf"], r_min=r_min)
        res = run_simp(em, params, point_loads, max_iter=cfg["max_iter"], change_tol=cfg["change_tol"])
        c_final = res.history["compliance"][-1]
        g = grayness(res.rho_phys)
        lines.append(f"{r_min:7.3f} {res.n_iter:6d} {c_final:12.4f} {g:9.4f}")

        im = plot_density(axes[i], em, res.rho_phys, L, H)
        axes[i].set_title(f"r_min={r_min}, compliance={c_final:.2f}, grayness={g:.4f}")
        fig.colorbar(im, ax=axes[i])
        axes[i].set_ylabel("y")

    axes[-1].set_xlabel("x")
    fig.suptitle("Filter-radius study")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "filter_radius_study.png", dpi=150)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "filter_radius_study.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    print("\nWrote figures/filter_radius_study.png and results/filter_radius_study.txt")


if __name__ == "__main__":
    main()
