"""Penalization-exponent study: run the canonical cantilever SIMP benchmark
at several SIMP penalization exponents p. p=1 (no penalization, a linear
rule-of-mixtures interpolation) is expected to produce a "grey", diffuse,
often checkerboard-tainted design since intermediate densities are not
penalized relative to solid material; p=3 (the standard default) should
produce a near-binary (low-grayness) design. Reports the grayness metric
quantitatively as a function of p, rather than asserting the qualitative
effect without evidence.

Usage: python scripts/penalization_study.py [--config configs/local.yaml]
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
    config_tag = pathlib.Path(args.config).stem
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())["penalization_study"]

    L, H = cfg["L"], cfg["H"]
    ps = cfg["penalizations"]
    point_loads = [((L, H / 2.0), (0.0, -1.0))]

    fig, axes = plt.subplots(len(ps), 1, figsize=(9, 2.4 * len(ps)), sharex=True)
    if len(ps) == 1:
        axes = [axes]

    lines = [f"Penalization study: mesh {cfg['nx']}x{cfg['ny']}, Vf={cfg['Vf']}, "
             f"r_min={cfg['r_min']}", "",
             f"{'p':>6} {'iters':>6} {'compliance':>12} {'grayness':>9}"]

    for i, p in enumerate(ps):
        em = build_mesh(cfg["nx"], cfg["ny"], L=L, H=H)
        params = SimpParams(E0=1.0, Emin=1e-9, nu=0.3, p=p, Vf=cfg["Vf"], r_min=cfg["r_min"])
        res = run_simp(em, params, point_loads, max_iter=cfg["max_iter"], change_tol=cfg["change_tol"])
        c_final = res.history["compliance"][-1]
        g = grayness(res.rho_phys)
        lines.append(f"{p:6.1f} {res.n_iter:6d} {c_final:12.4f} {g:9.4f}")

        im = plot_density(axes[i], em, res.rho_phys, L, H)
        axes[i].set_title(f"p={p}, compliance={c_final:.2f}, grayness={g:.4f}")
        fig.colorbar(im, ax=axes[i])
        axes[i].set_ylabel("y")

    axes[-1].set_xlabel("x")
    fig.suptitle("Penalization-exponent study")
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / f"penalization_study_{config_tag}.png", dpi=150)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"penalization_study_{config_tag}.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    print(f"\nWrote figures/penalization_study_{config_tag}.png and results/penalization_study_{config_tag}.txt")


if __name__ == "__main__":
    main()
