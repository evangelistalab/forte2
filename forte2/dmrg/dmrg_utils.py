"""
Convention bridges between forte2 and block2.

forte2 stores the two-electron integrals in physicist's notation,
``V[p, q, r, s] = <pq|rs>``, and its spin-free RDMs contract with those
integrals as

    E = E_core + sum_pq H_pq gamma1_pq + 0.5 * sum_pqrs V_pqrs gamma2_pqrs.

block2 works in chemist's notation, ``g2e[p, q, r, s] = (pq|rs)``, and returns
its two-particle density matrix in that same chemist layout. The transforms
below map between the two. They were validated numerically against the exact CI
solver (energies to ~1e-13, RDMs to ~1e-6). The 3-RDM follows the same pattern:
block2 returns the NPDM with the creation indices in order and the annihilation
indices reversed, so the forte2 RDM is recovered by reversing the trailing
(annihilation) axes.
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


def block2_3pdm_to_sf_3rdm(pdm3):
    r"""
    Convert a block2 3-particle density matrix into forte2's spin-free 3-RDM
    convention.

    block2 returns the (spin-traced) NPDM with the creation indices in order and
    the annihilation indices reversed,

        pdm3[i, j, k, c, b, a] = <a^+_i a^+_j a^+_k a_c a_b a_a>,

    whereas forte2 stores

        gamma3[p, q, r, s, t, u] = <a^+_p a^+_q a^+_r a_u a_t a_s>.

    The two are related by reversing the last three (annihilation) axes,
    ``gamma3[p, q, r, s, t, u] = pdm3[p, q, r, u, t, s]``. This is the
    three-particle analogue of the ``(0, 1, 3, 2)`` swap used for the 2-RDM.
    """
    return np.ascontiguousarray(np.transpose(pdm3, (0, 1, 2, 5, 4, 3)))
