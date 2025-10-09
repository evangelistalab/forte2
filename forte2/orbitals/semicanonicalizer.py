import numpy as np

from forte2.system import System
from forte2.state import MOSpace, EmbeddingMOSpace
from forte2.jkbuilder import FockBuilder


class Semicanonicalizer:
    r"""
    Class to perform semicanonicalization of a set of molecular orbitals.
    The semi-canonical basis is defined as a basis where the generalized Fock matrix
    is diagonal in a set of subspaces.

    Parameters
    ----------
    mo_space : MOSpace
        The molecular orbital space defining the subspaces.
    system : System
        The system object containing the basis set and other properties.
    fock_builder : FockBuilder, optional
        An instance of FockBuilder to compute the Fock matrix.
        If None, a new FockBuilder will be created.
    mix_inactive : bool, optional, default=False
        If True, frozen_core and core orbitals will be diagonalized together,
        virtual and frozen_virt also will be diagonalized together.
    mix_active : bool, optional, default=False
        If True, all GAS active orbitals will be diagonalized together.

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
        mo_space: MOSpace | EmbeddingMOSpace = None,
        fock_builder: FockBuilder = None,
        mix_inactive: bool = False,
        mix_active: bool = False,
        do_frozen: bool = True,
        do_active: bool = True,
    ):
        self.mo_space = mo_space
        self.two_component = system.two_component
        self.system = system
        self.fock_builder = fock_builder
        # these are only used for MOSpace
        self.mix_inactive = mix_inactive
        self.mix_active = mix_active
        # these are only used for EmbeddingMOSpace
        self.do_frozen = do_frozen
        self.do_active = do_active

        if self.fock_builder is None:
            self.fock_builder = FockBuilder(self.system, use_aux_corr=True)

    def semi_canonicalize(self, g1, C_contig):
        """
        Perform the semi-canonicalization.

        Parameters
        ----------
        g1 : np.ndarray
            The active space 1-electron density matrix in the molecular orbital basis.
        C_contig : np.ndarray
            The molecular orbital coefficients, in the "contiguous" order of the orbitals.
            Note that all other quantities are also defined in this order.
        """
        self.fock = self._build_fock(g1, C_contig)
        eps = np.zeros(self.mo_space.nmo)
        # U_init = I so that skipped blocks are not modified
        U = np.eye(self.mo_space.nmo, dtype=self.fock.dtype)

        def _eigh(sl):
            return np.linalg.eigh(self.fock[sl, sl])

        slice_list = self._generate_elementary_spaces()

        for sl in slice_list:
            # avoid calling eigh on empty arrays
            if sl.stop - sl.start > 0:
                e, c = _eigh(sl)
                eps[sl] = e
                U[sl, sl] = c

        self.U = U
        self.Uactv = U[self.mo_space.actv, self.mo_space.actv]
        self.C_semican = C_contig @ U
        self.eps_semican = eps
        self.fock_semican = U.T.conj() @ self.fock @ U

    def _generate_elementary_spaces(self):
        slice_list = []
        if isinstance(self.mo_space, MOSpace):
            if self.mix_inactive:
                slice_list.append(self.mo_space.docc)
            else:
                slice_list.append(self.mo_space.frozen_core)
                slice_list.append(self.mo_space.core)
            if self.mix_active:
                slice_list.append(self.mo_space.actv)
            else:
                slice_list.extend(self.mo_space.gas)
            if self.mix_inactive:
                slice_list.append(self.mo_space.uocc)
            else:
                slice_list.append(self.mo_space.virt)
                slice_list.append(self.mo_space.frozen_virt)
        elif isinstance(self.mo_space, EmbeddingMOSpace):
            if self.do_frozen:
                slice_list.append(self.mo_space.frozen_core)
            slice_list.append(self.mo_space.B_core)
            slice_list.append(self.mo_space.A_core)
            if self.do_active:
                slice_list.append(self.mo_space.actv)
            slice_list.append(self.mo_space.A_virt)
            slice_list.append(self.mo_space.B_virt)
            if self.do_frozen:
                slice_list.append(self.mo_space.frozen_virt)

        return slice_list

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
