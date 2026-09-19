"""Post-processing quantities derived from a displacement field: von Mises
stress and strain-energy density.

Von Mises stress (2D plane-stress form, with sigma_zz = 0 by the plane-
stress assumption itself):

    sigma_vm = sqrt(sigma_xx^2 - sigma_xx*sigma_yy + sigma_yy^2 + 3*sigma_xy^2)

the standard scalar equivalent-stress measure used to compare a general
multiaxial stress state against a uniaxial yield criterion (von Mises,
1913). Strain-energy density is w(u) = (1/2)*sigma(u):epsilon(u); its
integral over the domain is exactly compliance/2 for the load case solved
(see src/simp.py for the compliance functional used in topology
optimisation).
"""
from __future__ import annotations

import ufl

from src.elasticity import stress, strain


def von_mises(u, lam, mu):
    s = stress(u, lam, mu)
    sxx, syy, sxy = s[0, 0], s[1, 1], s[0, 1]
    return ufl.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)


def strain_energy_density(u, lam, mu):
    return 0.5 * ufl.inner(stress(u, lam, mu), strain(u))
