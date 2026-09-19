"""Verify the SIMP compliance sensitivity dc/drho by a Taylor-remainder
(finite-difference) check, BEFORE trusting it for optimization, exactly as
the forward/adjoint solvers in the companion projects
(navier-stokes-2d, darcy-inverse-problem) are verified before use.

For a random density field rho0 and random direction d, the analytic
gradient predicts

    c(rho0 + t*d) = c(rho0) + t*(grad c . d) + O(t^2)

so the remainder r(t) = |c(rho0+t*d) - c(rho0) - t*(grad c . d)| should
shrink as O(t^2): halving t should quarter r(t), i.e. the observed order
log(r(t)/r(t/10)) / log(10) should be close to 2.

Usage: python scripts/verify_sensitivity.py [--config configs/local.yaml]
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
from src.simp import SimpParams, build_filter, elemental_area, compliance_only, compliance_gradient

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "local.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())["sensitivity_taylor"]

    em = build_mesh(cfg["nx"], cfg["ny"], L=cfg["L"], H=cfg["H"])
    params = SimpParams(E0=1.0, Emin=1e-9, nu=0.3, p=cfg["p"], Vf=cfg["Vf"], r_min=cfg["r_min"])
    point_loads = [((cfg["L"], cfg["H"] / 2.0), (0.0, -1.0))]
    Hf, Hs = build_filter(em, params.r_min)
    elem_area = elemental_area(em)

    rng = np.random.default_rng(0)
    n = em.V0.dofmap.index_map.size_local
    rho0 = rng.uniform(0.2, 0.8, n)
    d = rng.uniform(-1.0, 1.0, n)

    c0, dc = compliance_gradient(em, rho0, params, point_loads, Hf, Hs, elem_area)
    grad_dir = dc @ d

    ts = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
    remainders = []
    for t in ts:
        c_pert = compliance_only(em, rho0 + t * d, params, point_loads, Hf, Hs, elem_area)
        remainders.append(abs(c_pert - c0 - t * grad_dir))
    remainders = np.array(remainders)

    lines = [f"Sensitivity Taylor-remainder test: mesh {cfg['nx']}x{cfg['ny']}, "
             f"p={cfg['p']}, Vf={cfg['Vf']}, r_min={cfg['r_min']}, n_elements={n}",
             f"c(rho0) = {c0:.8e}, grad.d = {grad_dir:.8e}",
             "",
             f"{'t':>10} {'remainder':>16} {'observed order':>16}"]
    for i, t in enumerate(ts):
        if i == 0:
            order_str = "      --"
        else:
            order = np.log(remainders[i - 1] / remainders[i]) / np.log(ts[i - 1] / ts[i])
            order_str = f"{order:8.3f}"
        lines.append(f"{t:10.1e} {remainders[i]:16.6e} {order_str:>16}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "sensitivity_taylor_test.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.loglog(ts, remainders, "o-", label="Taylor remainder $|c(\\rho_0+td)-c(\\rho_0)-t\\,\\nabla c \\cdot d|$")
    ax.loglog(ts, remainders[0] * (ts / ts[0]) ** 2, "k--", alpha=0.5, label="$O(t^2)$ reference")
    ax.set_xlabel("perturbation size $t$")
    ax.set_ylabel("remainder")
    ax.set_title("SIMP compliance-sensitivity Taylor-remainder test")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "sensitivity_taylor_test.png", dpi=150)
    print("\nWrote results/sensitivity_taylor_test.txt and figures/sensitivity_taylor_test.png")


if __name__ == "__main__":
    main()
