"""Shared plotting helpers for fields on the structured quadrilateral mesh.

matplotlib's triangulation-based routines (tricontourf, tripcolor) produce
rendering artefacts on a quad-split triangulation of this mesh (see
examples/cantilever_baseline.py); the reliable approach used throughout this
project is to interpolate the scattered field values onto a regular grid and
render with pcolormesh.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata

from src.simp import cell_centers


def to_regular_grid(x, y, values, L, H, nx=300, ny=150, method="linear"):
    xi = np.linspace(0, L, nx)
    yi = np.linspace(0, H, ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), values, (Xi, Yi), method=method)
    return Xi, Yi, Zi


def plot_density(ax, em, rho_phys, L, H, cmap="Greys", nx=400, ny=200):
    """Piecewise-constant (DG0) density field -- nearest-neighbour
    interpolation avoids smoothing the element-wise field across the
    0/1 boundary."""
    centers = cell_centers(em)
    Xi, Yi, Zi = to_regular_grid(centers[:, 0], centers[:, 1], rho_phys, L, H,
                                  nx=nx, ny=ny, method="nearest")
    im = ax.pcolormesh(Xi, Yi, Zi, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_aspect("equal")
    return im
