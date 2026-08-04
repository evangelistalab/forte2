"""
Convention bridges between forte2 and block2.

forte2 stores the two-electron integrals in physicist's notation,
``V[p, q, r, s] = <pq|rs>``, and its spin-free RDMs contract with those
integrals as

    E = E_core + sum_pq H_pq gamma1_pq + 0.5 * sum_pqrs V_pqrs gamma2_pqrs.

block2 works in chemist's notation, ``g2e[p, q, r, s] = (pq|rs)``, and returns
its two-particle density matrix in that same chemist layout. The transforms
below map between the two. They were validated numerically against the exact CI
solver (energies to ~1e-13, RDMs to ~1e-6).
"""

import numpy as np


def physicist_to_chemist_g2e(V):
    r"""
    Convert two-electron integrals from forte2's physicist convention
    ``V[p, q, r, s] = <pq|rs>`` to block2's chemist convention
    ``g2e[p, q, r, s] = (pq|rs) = <pr|qs> = V[p, r, q, s]``.
    """
    return np.ascontiguousarray(np.transpose(V, (0, 2, 1, 3)))


def block2_2pdm_to_sf_2rdm(pdm2):
    r"""
    Convert a block2 (chemist-notation) 2-particle density matrix into forte2's
    spin-free 2-RDM convention: ``gamma2[p, q, r, s] = pdm2[p, q, s, r]``.
    """
    return np.ascontiguousarray(np.transpose(pdm2, (0, 1, 3, 2)))
