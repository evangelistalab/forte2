import numpy as np
from numpy.typing import ArrayLike, NDArray

from forte2.system import System
from forte2.state import MOSpace, EmbeddingMOSpace
from .orbital_blocks import OrbitalBlockBuilder


class Semicanonicalizer:
    r"""
    Class to perform semicanonicalization of a set of molecular orbitals.
    The semi-canonical basis is defined as a basis where the generalized Fock matrix
    is diagonal in a set of subspaces.

    Parameters
    ----------
    mo_space : MOSpace or EmbeddingMOSpace
        The molecular orbital space defining the subspaces.
    system : System
        The system object containing the basis set and other properties.
    irrep_indices : np.ndarray or list[int], optional
        Orbital irrep labels in the same contiguous order as ``C_contig``. If provided,
        semicanonicalization is performed separately within each irrep.
    mix_inactive : bool, optional, default=False
        If True, frozen_core and core orbitals will be diagonalized together,
        virtual and frozen_virt also will be diagonalized together.
    mix_active : bool, optional, default=False
        If True, all GAS active orbitals will be mixed, breaking the GAS subspace structure.
    do_frozen : bool, optional, default=True
        If True, the frozen core and frozen virtual orbitals will be semi-canonicalized.
        If False, they will be left in the original basis.
    do_active : bool, optional, default=True
        If True, the active orbitals will be semi-canonicalized.
        If False, they will be left in the original basis.

    Attributes
    ----------
    fock : np.ndarray
        The generalized Fock matrix in the original basis.
    fock_semican : np.ndarray
        The generalized Fock matrix in the semi-canonical basis.
    eps_semican : np.ndarray
        The diagonal entries of the Fock matrix in the semi-canonical basis.
    C_semican : np.ndarray
        The molecular orbital coefficients in the semi-canonical basis.
    U : np.ndarray
        The unitary transformation matrix from the original to the semi-canonical basis.
    Uactv : np.ndarray
        The unitary transformation matrix within the active space.

    Notes
    -----
    The generalized Fock matrix is defined as

    .. math::
        f_p^q = h_p^q + \sum_{ij}^{\mathbf{H}}v_{pi}^{qj}\gamma_j^i,

    where :math:`\mathbf{H}` is the set of hole orbitals (i.e., all orbitals that are not unoccupied).
    The task of the `Semicanonicalizer` class is then to form the generalized Fock matrix
    and accumulate unitary transformations that diagonalizes the Fock matrix in the specified subspaces.
    If a subspace is to be untouched, the corresponding subblock of unitary transformation is set to the identity.
    """

    def __init__(
        self,
        system: System,
        mo_space: MOSpace | EmbeddingMOSpace,
        irrep_indices: ArrayLike | None = None,
        mix_inactive: bool = False,
        mix_active: bool = False,
        do_frozen: bool = True,
        do_active: bool = True,
    ) -> None:
        if mix_inactive and not do_frozen:
            raise ValueError(
                "Semicanonicalizer: mix_inactive=True is incompatible with do_frozen=False."
            )
        if not isinstance(mo_space, (MOSpace, EmbeddingMOSpace)):
            raise ValueError(
                "Semicanonicalizer: mo_space must be a MOSpace or EmbeddingMOSpace."
            )

        self.mo_space = mo_space
        self.two_component = system.two_component
        self.system = system
        self.fock_builder = system.fock_builder
        # These options define the semicanonicalization subspaces.
        self.mix_inactive = mix_inactive
        self.mix_active = mix_active
        self.do_frozen = do_frozen
        self.do_active = do_active
        self.orbital_blocks = OrbitalBlockBuilder(
            system, mo_space, irrep_indices, spaces=self._semicanonical_spaces()
        )

    def semi_canonicalize(self, g1: ArrayLike, C_contig: ArrayLike) -> None:
        """
        Perform the semi-canonicalization.

        Parameters
        ----------
        g1 : np.ndarray
            The active space 1-electron density matrix in the molecular orbital basis.
            Spin-summed if non-relativistic, spin-orbital if relativistic.
        C_contig : np.ndarray
            The molecular orbital coefficients, in the "contiguous" order of the orbitals.
            Note that all other quantities are also defined in this order.
        """
        g1, C_contig = self._validate_inputs(g1, C_contig)
        self.fock = self._build_fock(g1, C_contig)
        eps = np.zeros(self.mo_space.nmo)
        # U_init = I so that skipped blocks are not modified
        U = np.eye(self.mo_space.nmo, dtype=self.fock.dtype)

        def _eigh(idx):
            return np.linalg.eigh(self.fock[np.ix_(idx, idx)])

        # This loop diagonalizes Fock blocks in the requested orbital subspaces.
        for orb_idx in self.orbital_blocks.blocks_for_spaces():
            # avoid calling eigh on empty arrays
            if orb_idx.size == 0:
                continue
            e, c = _eigh(orb_idx)
            eps[orb_idx] = e
            U[np.ix_(orb_idx, orb_idx)] = c

        self.U = U
        self.Uactv = U[self.mo_space.actv, self.mo_space.actv]
        self.C_semican = C_contig @ U
        self.eps_semican = eps
        self.fock_semican = U.T.conj() @ self.fock @ U

    def _semicanonical_spaces(self) -> list[str]:
        spaces = []
        if isinstance(self.mo_space, MOSpace):
            if self.mix_inactive:
                spaces.append("docc")
            else:
                if self.do_frozen:
                    spaces.append("frozen_core")
                spaces.append("core")
            if self.do_active:
                if self.mix_active:
                    spaces.append("actv")
                else:
                    spaces.append("gas")
            if self.mix_inactive:
                spaces.append("uocc")
            else:
                spaces.append("virt")
                if self.do_frozen:
                    spaces.append("frozen_virt")
        elif isinstance(self.mo_space, EmbeddingMOSpace):
            if self.do_frozen:
                spaces.append("frozen_core")
            spaces.append("B_core")
            spaces.append("A_core")
            if self.do_active:
                spaces.append("actv")
            spaces.append("A_virt")
            spaces.append("B_virt")
            if self.do_frozen:
                spaces.append("frozen_virt")

        return spaces

    def _validate_inputs(
        self, g1: ArrayLike, C_contig: ArrayLike
    ) -> tuple[NDArray, NDArray]:
        C_contig = np.asarray(C_contig)
        g1 = np.asarray(g1)
        if C_contig.ndim != 2:
            raise ValueError("C_contig must be a two-dimensional array.")
        if C_contig.shape[1] != self.mo_space.nmo:
            raise ValueError("C_contig must have one column per MO.")

        nactv = self.mo_space.actv.stop - self.mo_space.actv.start
        if g1.shape != (nactv, nactv):
            raise ValueError("g1 must have shape (nactv, nactv).")

        return g1, C_contig

    def _build_fock(self, g1, C_contig):
        # core contribution to the generalized Fock matrix
        hcore = self.system.ints_hcore()
        # 'docc' slice includes frozen core in Fock build
        docc = self.mo_space.docc
        C_docc = C_contig[:, docc]
        J, K = self.fock_builder.build_JK([C_docc])
        Jfactor = 1 if self.two_component else 2
        gfactor = 1 if self.two_component else 0.5
        fock = hcore + Jfactor * J[0] - K[0]

        # active contribution to the generalized Fock matrix
        C_act = C_contig[:, self.mo_space.actv]
        J, K = self.fock_builder.build_JK_generalized(C_act, g1 * gfactor)
        fock += Jfactor * J - K
        fock = C_contig.conj().T @ fock @ C_contig
        return fock
