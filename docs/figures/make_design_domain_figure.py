"""Draw the canonical SIMP cantilever design domain, supports, load, and filter radius.

Dimensions, mesh and r_min are read from configs/local.yaml (section `canonical`); the load
position (L, H/2) and direction (0, -1) follow scripts/run_topopt.py.

    python docs/figures/make_design_domain_figure.py   ->   docs/figures/design_domain.svg
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
cfg = yaml.safe_load((ROOT / "configs" / "local.yaml").read_text())["canonical"]
L, H, nx, ny, r_min = cfg["L"], cfg["H"], cfg["nx"], cfg["ny"], cfg["r_min"]
h = L / nx

plt.rcParams.update({"font.size": 9, "svg.fonttype": "path", "svg.hashsalt": "fig"})
fig, ax = plt.subplots(figsize=(7.2, 3.9))
fig.patch.set_facecolor("white")

ax.add_patch(Rectangle((0, 0), L, H, fc="#eef3f8", ec="#1f3b57", lw=1.4))
# element grid (every element drawn, very light)
for x in np.linspace(0, L, nx + 1):
    ax.plot([x, x], [0, H], color="#c9d6e3", lw=0.25, zorder=1)
for y in np.linspace(0, H, ny + 1):
    ax.plot([0, L], [y, y], color="#c9d6e3", lw=0.25, zorder=1)

# clamped edge (hatched support)
ax.add_patch(Rectangle((-0.08, 0), 0.08, H, fc="none", ec="#1f3b57", hatch="////", lw=0))
ax.plot([0, 0], [0, H], color="#1f3b57", lw=2.4)
ax.text(-0.11, H / 2, "clamped\n$\\mathbf{u} = 0$", ha="right", va="center", fontsize=8.5)

# point load at (L, H/2), direction (0, -1)
ax.add_patch(FancyArrowPatch((L, H / 2 + 0.32), (L, H / 2), arrowstyle="-|>",
                             mutation_scale=14, color="#c0392b", lw=1.8, zorder=5))
ax.plot(L, H / 2, "o", color="#c0392b", ms=4, zorder=6)
ax.text(L + 0.05, H / 2 + 0.2, "point load\n$F = (0, -1)$\nat $(L, H/2)$", ha="left",
        va="center", color="#c0392b", fontsize=8.5)

# filter radius, to scale, with its element size
cx, cy = 0.62 * L, 0.3 * H
ax.add_patch(Circle((cx, cy), r_min, fc="#f5d7a8", ec="#a04e00", lw=1.0, zorder=3))
ax.add_patch(Rectangle((cx - h / 2, cy - h / 2), h, h, fc="#a04e00", ec="none", zorder=4))
ax.annotate(f"filter radius $r_{{min}} = {r_min:g}$\n(element size $h = {h:.4f}$)",
            xy=(cx + r_min * 0.7, cy - r_min * 0.7), xytext=(cx + 0.25, 0.1 * H), fontsize=8,
            color="#a04e00", arrowprops=dict(arrowstyle="-", lw=0.6, color="#a04e00"))

ax.text(0.33 * L, 0.72 * H, "design domain $\\Omega$\n"
        f"{nx} x {ny} quadrilateral elements\n"
        "one density $\\rho_e \\in [\\rho_{min}, 1]$ per element (DG0)\n"
        f"volume constraint: mean($\\rho$) $\\leq V_f = {cfg['Vf']:g}$",
        ha="center", va="center", fontsize=8.5,
        bbox=dict(fc="white", ec="#9db7cf", lw=0.6, boxstyle="round,pad=0.35"))

ax.annotate("", xy=(0, -0.09), xytext=(L, -0.09), arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.35"))
ax.text(L / 2, -0.11, f"L = {L:g}", ha="center", va="top", fontsize=8.5, color="0.35")
ax.annotate("", xy=(L + 0.55, 0), xytext=(L + 0.55, H), arrowprops=dict(arrowstyle="<->", lw=0.7, color="0.35"))
ax.text(L + 0.58, H / 2, f"H = {H:g}", ha="left", va="center", fontsize=8.5, color="0.35")

ax.set_xlim(-0.5, L + 0.8)
ax.set_ylim(-0.25, H + 0.08)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
out = Path(__file__).with_name("design_domain.svg")
fig.savefig(out, facecolor="white", metadata={"Date": None})  # deterministic output
if len(sys.argv) > 1:  # optional raster preview path
    fig.savefig(sys.argv[1], dpi=150, facecolor="white")
print(f"wrote {out.relative_to(ROOT)}")
