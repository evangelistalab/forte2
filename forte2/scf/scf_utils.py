import numpy as np
import scipy as sp

from forte2 import integrals
from forte2.system.build_basis import build_sap_potential_basis
from forte2.helpers.matrix_functions import givens_rotation


def sap_guess_hamiltonian(system, H):
    """
    Return the superposition of atomic potentials (SAP) guess Hamiltonian.

    S. Lehtola, J. Chem. Theory Comput. 15, 1593-1604 (2019), arXiv:1810.11659.
    For details, see https://doi.org/10.1063/5.0004046

    Parameters
    ----------
    system : forte2.System
        The system object containing the atoms and basis set.
    H : NDArray
        The core Hamiltonian matrix.

    Returns
    -------
    NDArray
        The core Hamiltonian plus the SAP potential, whose orbitals are the initial guess.
    """

    # Convert the SAP potential coefficients to a normalized charge-density basis.
    scaled_sap_basis = build_sap_potential_basis(
        "sap_helfem_large",
        system.geom_helper,
    )

    # generate the SAP integrals (P|mn)
    SAP_ints = integrals.coulomb_3c(system, scaled_sap_basis)

    # generate the SAP potential V_mn = sum_P (P|mn)
    SAP_V = np.einsum("Pmn->mn", SAP_ints)

    if system.two_component:
        _SAP_V = sp.linalg.block_diag(SAP_V, SAP_V).astype(complex)
    else:
        _SAP_V = SAP_V

    return H + _SAP_V


def smallest_gap_same_irrep(eps, irreps, nocc):
    """
    Find two orbitals that belong to the same irrep with the smallest gap.

    Parameters
    ----------
    eps : NDArray
        Orbital energies, in ascending order within the occupied orbitals and within
        the virtual ones.
    irreps : NDArray
        The irrep index of each orbital.
    nocc : int
        The number of occupied orbitals, which come first.

    Returns
    -------
    tuple[int, int] | None
        The HOMO and LUMO if they share an irrep. Otherwise, of the occupied and
        virtual orbitals that share an irrep, the pair with the smallest energy gap, or
        None if no such pair exists.
    """
    homo, lumo = nocc - 1, nocc
    if irreps[homo] == irreps[lumo]:
        return homo, lumo
    best = None
    for irrep in np.unique(irreps[:nocc]):
        virtuals = nocc + np.flatnonzero(irreps[nocc:] == irrep)
        if len(virtuals) == 0:
            continue
        pair = (int(np.flatnonzero(irreps[:nocc] == irrep)[-1]), int(virtuals[0]))
        if best is None or eps[pair[1]] - eps[pair[0]] < eps[best[1]] - eps[best[0]]:
            best = pair
    return best


def guess_mix(C, occ_idx, vir_idx, mixing_parameter=np.pi / 4):
    """
    Induce the breaking of S^2 symmetry for UHF ms=0.0 calculations.
    This is helpful for obtaining proxies for open-shell singlets, for example.

    Parameters
    ----------
    C : NDArray
        The MO coefficients.
    occ_idx : int
        The index of the occupied orbital to mix, usually the HOMO.
    vir_idx : int
        The index of the virtual orbital to mix, usually the LUMO.
    mixing_parameter : float, optional
        The mixing parameter for the Givens rotation.

    Returns
    -------
    NDArray
        The modified MO coefficients.

    Notes
    -----
    See Szabo and Ostlund Ch. 3.8.7.

    """
    cosq = np.cos(mixing_parameter)
    sinq = np.sin(mixing_parameter)
    Ca = givens_rotation(C, cosq, sinq, occ_idx, vir_idx)
    Cb = givens_rotation(C, cosq, -sinq, occ_idx, vir_idx)
    return [Ca, Cb]


def guess_mix_ghf(C, ha, hb, la, lb, mixing_parameter=np.pi / 4):
    """
    Induce the breaking of S^2 symmetry for UHF ms=0.0 calculations.
    This is helpful for obtaining proxies for open-shell singlets, for example.

    Parameters
    ----------
    C : NDArray
        The MO coefficients.
    ha : int
        The index of the highest occupied alpha(-like) orbital.
    hb : int
        The index of the highest occupied beta(-like) orbital.
    la : int
        The index of the lowest unoccupied alpha(-like) orbital.
    lb : int
        The index of the lowest unoccupied beta(-like) orbital.
    mixing_parameter : float, optional
        The mixing parameter for the Givens rotation.

    Returns
    -------
    NDArray
        The modified MO coefficients.

    Notes
    -----
    See Szabo and Ostlund Ch. 3.8.7.

    """
    cosq = np.cos(mixing_parameter)
    sinq = np.sin(mixing_parameter)
    # alpha channel
    C = givens_rotation(C, cosq, sinq, ha, la)
    # beta channel
    C = givens_rotation(C, cosq, -sinq, hb, lb)
    return C


def sz_adapt_degenerate_orbitals(eps, C, S, degeneracy_tol=1.0e-8):
    """
    Rotate each degenerate set of two-component orbitals onto eigenfunctions of S_z.

    An eigensolver returns an arbitrary basis within a degenerate eigenspace, and this
    choice differs between LAPACK implementations. Diagonalizing S_z within each
    eigenspace fixes the basis, up to phases, independently of the eigensolver.

    Parameters
    ----------
    eps : NDArray
        The orbital energies, in ascending order.
    C : NDArray
        The two-component MO coefficients, with the alpha AO block above the beta block.
    S : NDArray
        The one-component AO overlap matrix.
    degeneracy_tol : float, optional
        The largest orbital energy gap within a degenerate set.

    Returns
    -------
    NDArray
        The rotated MO coefficients. Each degenerate set is ordered as pairs of
        opposite S_z, starting with the largest S_z.
    """
    nbf = S.shape[0]
    C = C.copy()
    bounds = np.flatnonzero(np.diff(eps) > degeneracy_tol) + 1
    for block in np.split(np.arange(len(eps)), bounds):
        if len(block) < 2:
            continue
        Ca = C[:nbf, block]
        Cb = C[nbf:, block]
        # 2 < i | S_z | j > for i,j in a degenerate set
        _, U = np.linalg.eigh(Ca.conj().T @ S @ Ca - Cb.conj().T @ S @ Cb)
        # Interleave the largest and smallest S_z so each +S_z orbital is followed by a -S_z one
        n = len(block)
        order = np.stack((np.arange(n)[::-1], np.arange(n)), axis=1).ravel()[:n]
        C[:, block] = C[:, block] @ U[:, order]
    return C


def alpha_beta_mix(C, mixing_parameter=0.1):
    """
    Induce the breaking of S_z symmetry for GHF calculations.
    This function explicitly mixes the degenerate alpha and beta MOs,
    which results in non-vanishing D_ab/D_ba density matrix elements.

    Parameters
    ----------
    C : NDArray
        The MO coefficients.
    mixing_parameter : float, optional
        The mixing parameter for the Givens rotation.

    Returns
    -------
    NDArray
        The modified MO coefficients.
    """
    cosq = np.cos(mixing_parameter)
    sinq = np.sin(mixing_parameter)
    nmo = C.shape[1]
    for i in range(0, nmo, 2):
        C = givens_rotation(C, cosq, sinq, i, i + 1)
    return C


def break_complex_conjugation_symmetry(C, pert_strength=0.1):
    """
    Break the time-reversal/complex conjugation symmetry of the MO coefficients.
    A random phase is applied to each AO (cannot be MO as it would not change the density matrix).

    Parameters
    ----------
    C : NDArray
        The MO coefficients.
    pert_strength : float, optional
        The strength of the perturbation (in radians).

    Returns
    -------
    NDArray
        The modified MO coefficients.
    """
    # make sure alpha and beta are not complex conjugates
    phi = np.random.uniform(low=-pert_strength, high=pert_strength, size=C.shape[1])
    U = np.diag(np.exp(1.0j * phi))
    C = U @ C
    return C
