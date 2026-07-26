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
        do_active: bool = True,
    ) -> None:
        if not isinstance(mo_space, (MOSpace, EmbeddingMOSpace)):
            raise ValueError(
                "Semicanonicalizer: mo_space must be a MOSpace or EmbeddingMOSpace."
            )

        self.mo_space = mo_space
        self.system = system
        # These options define the semicanonicalization subspaces.
        self.mix_inactive = mix_inactive
        self.mix_active = mix_active
        self.do_active = do_active
        self.orbital_blocks = OrbitalBlockBuilder(mo_space, irrep_indices)

    def semi_canonicalize(self, g1: NDArray, C_contig: NDArray) -> None:
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
        for orb_idx in self.orbital_blocks.blocks_for_spaces(
            self._semicanonical_spaces()
        ):
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
            spaces.extend(["docc"] if self.mix_inactive else ["frozen_core", "core"])
            if self.do_active:
                spaces.append("actv" if self.mix_active else "gas")
            spaces.extend(["uocc"] if self.mix_inactive else ["virt", "frozen_virt"])
        elif isinstance(self.mo_space, EmbeddingMOSpace):
            spaces.extend(["frozen_core", "B_core", "A_core"])
            if self.do_active:
                spaces.append("actv" if self.mix_active else "gas")
            spaces.extend(["A_virt", "B_virt", "frozen_virt"])

        return spaces

    def _validate_inputs(
        self, g1: NDArray, C_contig: NDArray
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
        # 'docc' slice includes frozen core in Fock build
        C_docc = C_contig[:, self.mo_space.docc]
        C_act = C_contig[:, self.mo_space.actv]
        fock_ao = self.system.fock_builder.build_generalized_fock(
            C_core=C_docc,
            C_act=C_act,
            g1=g1,
        )
        return C_contig.conj().T @ fock_ao @ C_contig
