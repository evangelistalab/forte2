import numpy as np

from forte2.lib import ints
from forte2.system import System
from forte2.helpers import invsqrt_matrix
from forte2.symmetry import real_sph_to_j_adapted


class IAO:
    """
    Class to represent the intrinsic atomic orbital (IAO).

    Parameters
    ----------
    system : System
        The system for which the IAO is to be calculated.
    C_occ : NDArray
        The occupied molecular orbital coefficients.
    j_adapt : bool, optional, default=False
        If True, the IAOs are built in the j-adapted spinor basis.
    minao_basis : Basis, optional
        The minimal basis the IAOs are built from. Defaults to ``system.minao_basis``.

    Attributes
    ----------
    C_iao : NDArray
        The orthonormalized IAO coefficients, shape (nbf, nminao).

    Notes
    -----
    JCTC 2013, 9, 4834-4843
    """

    def __init__(
        self, system: System, C_occ: np.ndarray, j_adapt=False, minao_basis=None
    ):
        self.system = system
        self.C_occ = C_occ.copy()
        self.j_adapt = j_adapt
        self.minao_basis = system.minao_basis if minao_basis is None else minao_basis
        self.C_iao = self._make_iao(self.C_occ)
        self.nocc = C_occ.shape[1]

    @classmethod
    def from_mos(cls, system, mos, mo_space=None, minao_basis=None):
        """
        Builds the IAOs of the occupied orbitals in a set of MOs.

        If ``mo_space`` is given, its frozen core, core, and active orbitals count as
        occupied. Otherwise, the orbitals occupied in the reference determinant of
        ``mos`` do.

        Parameters
        ----------
        system : System
            The system the MOs belong to.
        mos : MO
            The restricted MOs.
        mo_space : MOSpace, optional
            The MO space that defines the occupied orbitals.
        minao_basis : Basis, optional
            The minimal basis the IAOs are built from. Defaults to ``system.minao_basis``.

        Returns
        -------
        IAO
            The IAOs of the occupied orbitals.

        Raises
        ------
        ValueError
            If the MOs aren't restricted, or if neither ``mo_space`` nor the reference
            occupations of ``mos`` define the occupied orbitals.
        """
        if not mos.restricted:
            raise ValueError("IAO.from_mos needs restricted MOs.")
        if mo_space is not None:
            occupied = mo_space.docc_indices + mo_space.active_indices
        elif mos.occupations is not None:
            occupied = np.flatnonzero(sum(mos.occupations)).tolist()
        else:
            raise ValueError(
                "The MOs have no reference occupations and no MO space is given, so "
                "the occupied orbitals are unknown."
            )
        return cls(system, mos.C[0][:, occupied], minao_basis=minao_basis)

    def _make_iao(self, C):
        basis = self.system.basis
        nbf = C.shape[0]
        minao_basis = self.minao_basis
        if minao_basis is None:
            raise ValueError("No minao_basis given, and the System has none.")

        if self.j_adapt:
            U1a, U1b = real_sph_to_j_adapted(basis)
            U2a, U2b = real_sph_to_j_adapted(minao_basis)
            U1 = np.vstack((U1a, U1b))
            C = U1.T.conj() @ C

        # various overlap matrices, see appendix C of JCTC 2013, 9, 4834-4843
        self.S1 = ints.overlap(basis, basis)
        self.S12 = ints.overlap(basis, minao_basis)
        self.S2 = ints.overlap(minao_basis, minao_basis)

        if self.j_adapt:
            _S1 = U1a.T.conj() @ self.S1 @ U1a
            _S1 += U1b.T.conj() @ self.S1 @ U1b
            self.S1 = _S1
            _S2 = U2a.T.conj() @ self.S2 @ U2a
            _S2 += U2b.T.conj() @ self.S2 @ U2b
            self.S2 = _S2
            _S12 = U1a.T.conj() @ self.S12 @ U2a
            _S12 += U1b.T.conj() @ self.S12 @ U2b
            self.S12 = _S12

        if C.shape[1] > self.S2.shape[0]:
            raise ValueError(
                f"IAOs from {self.S2.shape[0]} minimal basis functions can't span "
                f"{C.shape[1]} orbitals. Use a larger minimal basis."
            )

        S1_inv = np.linalg.pinv(self.S1, hermitian=True)
        S2_inv = np.linalg.pinv(self.S2, hermitian=True)

        # projector onto the large basis
        P12 = S1_inv @ self.S12
        # projector onto the minao basis
        P21 = S2_inv @ self.S12.T.conj()
        # downproject and upproject the occupied MOs to get a set of depolarized MOs
        # cf. eq 1
        C_depolarized = P12 @ P21 @ C
        # orthonormal set of depolarized MOs
        Ct = _orthogonalize(C_depolarized, self.S1)

        C_polarized_occ = C @ C.T.conj() @ self.S1 @ Ct @ Ct.T.conj() @ self.S1 @ P12
        C_polarized_vir = (
            (np.eye(nbf) - C @ C.T.conj() @ self.S1)
            @ (np.eye(nbf) - Ct @ Ct.T.conj() @ self.S1)
            @ P12
        )

        C_iao = _orthogonalize(C_polarized_occ + C_polarized_vir, self.S1)
        return C_iao

    def make_sf_1rdm(self, sf_1rdm_ao):
        r"""
        Generate the spin-free 1-particle density matrix in the IAO basis, given by

        .. math::
            \gamma_{\rho\sigma} = \langle\rho|\hat{\gamma}|\sigma\rangle,

        where :math:`\hat{\gamma}=2\sum_{i \in \text{occ}} |i\rangle\langle i|` is the
        closed-shell RHF 1e density matrix (see eq 3 in the JCTC paper).

        Parameters
        ----------
        sf_1rdm_ao : NDArray
            The spin-free 1-particle density matrix in the large AO basis.

        Returns
        -------
        NDArray
            The spin-free 1-particle density matrix in the IAO basis.
        """
        if self.j_adapt:
            raise NotImplementedError
        S = self.system.ints_overlap()
        # contracting two AO indices requires an intervening overlap matrix
        # due to the non-orthogonality of the AOs
        return self.C_iao.T @ (S @ sf_1rdm_ao @ S) @ self.C_iao

    def charge_matrices(self, C):
        """
        Compute the IAO charge matrix of each atom for a set of orbitals.

        Parameters
        ----------
        C : NDArray
            The orbital coefficients, shape (nbf, norb). The orbitals must lie in the
            span of the IAOs.

        Returns
        -------
        NDArray
            The charge matrices, shape (natom, norb, norb). Element (A, i, j) is the
            overlap of orbitals i and j restricted to the IAOs of atom A.

        Raises
        ------
        ValueError
            If the IAOs don't represent the orbitals.
        """
        if self.j_adapt:
            raise NotImplementedError
        X = self.C_iao.T @ self.S1 @ C
        if not np.allclose(self.C_iao @ X, C, rtol=0, atol=1e-8):
            raise ValueError(
                "The IAOs don't represent the orbitals. The IAOs span the orbitals they "
                "were built from, and a minimal set of valence-like orbitals beyond them."
            )
        atom_slices = [slice(*x) for x in self.minao_basis.center_first_and_last]
        return np.array([X[sl].T @ X[sl] for sl in atom_slices])


def _orthogonalize(C, S):
    """
    Given a subset of molecular orbitals C, shape (nbf, n), n<= nbf,
    orthonormalize among the n orbitals under the metric S,
    such that C_ortho.T @ S @ C_ortho = I.
    See appendix C of JCTC 2013, 9, 4834-4843
    """

    X = C.T.conj() @ S @ C
    X_invsqrt, *_ = invsqrt_matrix(X)
    return C @ X_invsqrt
