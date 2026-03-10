import itertools
from functools import cached_property
import numpy as np
import scipy as sp
import math

import forte2
from forte2 import integrals
from forte2.helpers import logger
from forte2.helpers.matrix_functions import cholesky_wrapper


class FockBuilder:
    """
    Class to build the Fock matrix using the Cholesky decomposition of the auxiliary basis integrals.
    This class computes the atomic Coulomb (J) and exchange (K) matrices
    using the auxiliary basis functions.

    Parameters
    ----------
    system : System or ModelSystem
        The system for which to build the Fock matrix.
        If a ModelSystem is provided, it will decompose the 4D ERI tensor using Cholesky decomposition with complete pivoting.
    use_aux_corr : bool, optional, default=False
        If True, uses ``system.auxiliary_basis_corr`` instead of ``system.auxiliary_basis``.

    Attributes
    ----------
    B_Pmn : NDArray
        The B tensor with shape (Naux, Nao, Nao). Lazily evaluated.
    naux : int
        The number of auxiliary basis functions.
    nbf : int
        The number of basis functions in the system.
    """

    def __init__(self, system, use_aux_corr=False):
        self.system = system
        self.use_aux_corr = use_aux_corr
        self.nbf = system.nbf

    @cached_property
    def B_Pmn(self):
        if isinstance(self.system, forte2.ModelSystem):
            res, naux = self._build_B_model_system()
        else:
            if not self.system.cholesky_tei:
                basis = self.system.basis
                if self.use_aux_corr:
                    assert hasattr(
                        self.system, "auxiliary_basis_corr"
                    ), "The system does not have an auxiliary_basis_corr defined."
                    aux_basis = self.system.auxiliary_basis_corr
                else:
                    assert hasattr(
                        self.system, "auxiliary_basis"
                    ), "The system does not have an auxiliary_basis defined."
                    aux_basis = self.system.auxiliary_basis
                res, naux = self._build_B_density_fitting(basis, aux_basis)
            else:
                res, naux = self._build_B_cholesky(
                    self.system.basis, self.system.cholesky_tol
                )
                # set the number of auxiliary basis, unknown before this point
                self.system.naux = naux
        self.naux = naux
        return res

    def _build_B_model_system(self):
        nbf = self.system.nbf
        eri = self.system.eri.reshape((nbf**2,) * 2)
        B = cholesky_wrapper(eri, tol=-1)
        B = B.reshape((B.shape[0], nbf, nbf))
        naux = B.shape[0]
        return B, naux

    def _build_B_cholesky(self, basis, cholesky_tol):
        # Compute the memory requirements
        nbf = basis.size
        memory_gb = 8 * (nbf**4) / (1024**3)
        logger.log_info1("Building B tensor using Cholesky decomposition")
        logger.log_info1(
            f"Temporary memory requirement for 4-index integrals: {memory_gb:.2f} GB"
        )
        eri_full = integrals.coulomb_4c(self.system)
        eri = eri_full.reshape((nbf**2,) * 2)

        B = cholesky_wrapper(eri, tol=cholesky_tol)
        B = B.reshape((B.shape[0], nbf, nbf))

        naux = B.shape[0]

        memory_gb = 8 * (naux * nbf**2) / (1024**3)
        logger.log_info1(f"Memory requirements: {memory_gb:.2f} GB")
        logger.log_info1(f"Number of system basis functions: {nbf}")
        logger.log_info1(f"Number of auxiliary basis functions: {naux}")

        return B, naux

    def _build_B_density_fitting(self, basis, auxiliary_basis):
        # Compute the memory requirements
        nb = basis.size
        naux = auxiliary_basis.size
        memory_gb = 8 * (naux**2 + naux * nb**2) / (1024**3)
        logger.log_info1(f"Memory requirements: {memory_gb:.2f} GB")
        logger.log_info1(f"Number of system basis functions: {nb}")
        logger.log_info1(f"Number of auxiliary basis functions: {naux}")

        # Compute the integrals (P|Q) with P, Q in the auxiliary basis
        M = integrals.coulomb_2c(self.system, auxiliary_basis)

        # Decompose M = L L.T
        L = sp.linalg.cholesky(M, lower=True)

        # Solve L X = I, or X = L^{-1} = M^{-1/2}
        I = np.eye(M.shape[0])
        M_inv_sqrt = sp.linalg.solve_triangular(L, I, lower=True)

        # Compute the integrals (P|mn) with P in the auxiliary basis and m, n in the system basis
        Pmn = integrals.coulomb_3c(self.system, auxiliary_basis)

        # Compute B[P|mn] = M^{-1/2}[P|Q] (Q|mn)
        B = np.einsum("PQ,Qmn->Pmn", M_inv_sqrt, Pmn, optimize=True)
        del Pmn

        return B, naux

    def build_J(self, D):
        J = [
            np.einsum("Pmn,Prs,sr->mn", self.B_Pmn, self.B_Pmn, Di, optimize=True)
            for Di in D
        ]
        return J

    def build_K(self, C):
        if self.system.two_component:
            assert (
                len(C) == 1
            ), "C must be a list with one element for two-component systems."
            C = [C[0][: self.nbf, :], C[0][self.nbf :, :]]
        Y = [np.einsum("Pms,si->Pmi", self.B_Pmn, Ci, optimize=True) for Ci in C]
        if self.system.two_component:
            K = []
            for Yi in Y:
                for Yj in Y:
                    K.append(np.einsum("Pmi,Pni->mn", Yi, Yj.conj(), optimize=True))
        else:
            K = [np.einsum("Pmi,Pni->mn", Yi, Yi.conj(), optimize=True) for Yi in Y]
        return K

    def build_JK(self, C):
        r"""
        Compute the Coulomb and exchange matrices for a given set of orbitals.

        .. math::

            J_{\mu\nu} = \sum_{i}\sum_{\rho\sigma} (\mu\nu|\rho\sigma) C_{\sigma i} C^*_{\rho i}\\
            K_{\mu\nu} = \sum_{i}\sum_{\rho\sigma} (\mu\sigma|\rho\nu) C_{\sigma i} C^*_{\rho i}

        Parameters
        ----------
        C : list of NDArray
            List of coefficient matrices for the orbitals.
        
        Returns
        -------
        tuple(list[NDArray], list[NDArray])
            A tuple containing the lists of Coulomb (J) and exchange (K) matrices.
        """
        nbf = self.system.nbf
        D = [np.einsum("mi,ni->mn", Ci, Ci.conj(), optimize=True) for Ci in C]
        if self.system.two_component:
            assert (
                len(C) == 1
            ), "C must be a list with one element for two-component systems."
            # build_J only needs the aa and bb parts of the density matrix
            D = [D[0][:nbf, :nbf], D[0][nbf:, nbf:]]
        J = self.build_J(D)
        K = self.build_K(C)
        if self.system.two_component:
            # assemble the two-component JK matrices
            Jaa, Jbb = J
            Kaa, Kab, Kba, Kbb = K
            J_spinor = np.zeros((nbf * 2,) * 2, dtype=np.complex128)
            K_spinor = np.zeros((nbf * 2,) * 2, dtype=np.complex128)
            J_spinor[:nbf, :nbf] += Jaa + Jbb
            J_spinor[nbf:, nbf:] += Jaa + Jbb
            K_spinor[:nbf, :nbf] += Kaa
            K_spinor[nbf:, nbf:] += Kbb
            K_spinor[:nbf, nbf:] += Kab
            K_spinor[nbf:, :nbf] += Kba
            J, K = [J_spinor], [K_spinor]
        return J, K

    def build_JK_generalized(self, C, g1):
        r"""
        Compute the generalized Coulomb and exchange matrices for a given set of orbitals.
        These are used in building the generalized Fock matrix in multi-reference methods.
        The generalized J and K matrices are defined as

        .. math::
            J_{\mu\nu} = \sum_{uv}\sum_{\rho\sigma} (\mu\nu|\rho\sigma) C^*_{\rho u} C_{\sigma v} \gamma_{uv}\\
            K_{\mu\nu} = \sum_{uv}\sum_{\rho\sigma} (\mu\sigma|\rho\nu) C^*_{\rho u} C_{\sigma v} \gamma_{uv}

        Parameters
        ----------
        C : NDArray
            Coefficient matrix for the orbitals.
        g1 : NDArray    
            One-electron density matrix (1-RDM) in the MO basis.
        
        Returns
        -------
        tuple(NDArray, NDArray)
            A tuple containing the generalized Coulomb (J) and exchange (K) matrices.
        """
        assert C.shape[1] == g1.shape[0], "C and g1 must have compatible dimensions"

        try:
            # C^*_{\rho u} C_{\sigma v} \gamma_{uv} = C^*_{\rho u} C_{\sigma v} L_ua L_va^*
            # = (C_{\sigma v} L_va^*) (C^*_{\rho u} L_ua)
            # = C_{\sigma a} C^*_{\rho a}
            L = np.linalg.cholesky(g1, upper=False)
            Cp = C @ L.conj()
        except np.linalg.LinAlgError:
            n, L = np.linalg.eigh(g1)
            assert np.all(n > -1.0e-11), "g1 must be positive semi-definite"
            n = np.maximum(n, 0)
            Cp = C @ L.conj() @ np.diag(np.sqrt(n))

        J, K = self.build_JK([Cp])
        return J[0], K[0]

    def two_electron_integrals_gen_block(self, C1, C2, C3, C4):
        r"""
        Compute the two-electron integrals for a given set of orbitals. This method is
        general and can handle different sets of orbitals for each index (p, q, r, s).

        The resulting integrals are stored in a 4D array with the following convention:
        V[p,q,r,s] = :math:`\langle pq | rs \rangle`, where

        .. math::

            \langle pq | rs \rangle = \iint \phi^*_p(r_1) \phi^*_q(r_2) \frac{1}{r_{12}} \phi_r(r_1) \phi_s(r_2) dr_1 dr_2

        Parameters
        ----------
        C1 : NDArray
            Coefficient matrix for the first set of orbitals (index p).
        C2 : NDArray
            Coefficient matrix for the second set of orbitals (index q).
        C3 : NDArray
            Coefficient matrix for the third set of orbitals (index r).
        C4 : NDArray
            Coefficient matrix for the fourth set of orbitals (index s).

        Returns
        -------
        V : NDArray
            The two-electron integrals in the form of a 4D array.
        """
        V = np.einsum(
            "Pmn,Prs,mi,rj,nk,sl->ijkl",
            self.B_Pmn,
            self.B_Pmn,
            C1.conj(),
            C2.conj(),
            C3,
            C4,
            optimize=True,
        )
        return V

    def two_electron_integrals_block(self, C):
        r"""
        Compute the two-electron integrals for a given set of orbitals.

        The resulting integrals are stored in a 4D array with the following convention:
        V[p,q,r,s] = :math:`\langle pq | rs \rangle`, where

        .. math::

            \langle pq | rs \rangle = \iint \phi^*_p(r_1) \phi^*_q(r_2) \frac{1}{r_{12}} \phi_r(r_1) \phi_s(r_2) dr_1 dr_2

        Parameters
        ----------
        C : NDArray
            Coefficient matrix for the set of orbitals.

        Returns
        -------
        V : NDArray
            The two-electron integrals in the form of a 4D array.
        """
        return self.two_electron_integrals_gen_block(C, C, C, C)

    def two_electron_integrals_gen_block_spinor(self, C1, C2, C3, C4):
        r"""
        Compute the two-electron integrals for a given set of orbitals. This method is
        general and can handle different sets of orbitals for each index (p, q, r, s).

        The resulting integrals are stored in a 4D array with the following convention:
        V[p,q,r,s] = :math:`\langle pq | rs \rangle`, where

        .. math::

            \langle pq | rs \rangle = \iint \phi^*_p(r_1) \phi^*_q(r_2) \frac{1}{r_{12}} \phi_r(r_1) \phi_s(r_2) dr_1 dr_2


        Parameters
        ----------
        C1 : NDArray
            Coefficient matrix for the first set of orbitals (index p).
        C2 : NDArray
            Coefficient matrix for the second set of orbitals (index q).
        C3 : NDArray
            Coefficient matrix for the third set of orbitals (index r).
        C4 : NDArray
            Coefficient matrix for the fourth set of orbitals (index s).

        Returns
        -------
        V : NDArray
            The two-electron integrals in the form of a 4D array.
        """
        nbf = self.nbf
        V = np.zeros(
            (C1.shape[1], C2.shape[1], C3.shape[1], C4.shape[1]), dtype=complex
        )
        _a = slice(0, nbf)
        _b = slice(nbf, nbf * 2)
        # equivalent to 4 nested for loops over a,b parts of of C1/2/3/4
        for s1, s2, s3, s4 in itertools.product([_a, _b], repeat=4):
            # this essentially enforces the spin orthogonality of the AOs
            if (s1 != s3) or (s2 != s4):
                continue
            V += np.einsum(
                "Pmn,Prs,mi,rj,nk,sl->ijkl",
                self.B_Pmn,
                self.B_Pmn,
                C1[s1, :].conj(),
                C2[s2, :].conj(),
                C3[s3, :],
                C4[s4, :],
                optimize=True,
            )

        return V

    def two_electron_integrals_block_spinor(self, C):
        r"""
        Compute the two-electron integrals for a given set of spin-orbitals.

        The resulting integrals are stored in a 4D array with the following convention:
        V[p,q,r,s] = :math:`\langle pq | rs \rangle`, where

        .. math::

            \langle pq | rs \rangle = \iint \phi^*_p(r_1) \phi^*_q(r_2) \frac{1}{r_{12}} \phi_r(r_1) \phi_s(r_2) dr_1 dr_2

        Parameters
        ----------
        C : NDArray
            Coefficient matrix for the set of spin-orbitals.
        antisymmetrize : bool, optional, default=False
            Whether to antisymmetrize the integrals. If True, the integrals are antisymmetrized as:
            V[p,q,r,s] = :math:`\langle pq || rs \rangle = \langle pq | rs \rangle - \langle pq | sr \rangle`

        Returns
        -------
        V : NDArray
            The two-electron integrals in the form of a 4D array.
        """
        return self.two_electron_integrals_gen_block_spinor(C, C, C, C)

    def B_tensor_gen_block(self, C1, C2):
        r"""
        Compute the MO basis B tensor for a given set of orbitals. This method is
        general and can handle different sets of orbitals for each index (p, q).

        The resulting B tensor is stored in a 3D array with the following convention:
        B[P,m,n] = :math:`(P | mn)`, where

        .. math::

            (P | mn) = \iint \phi_P(r_1)  \frac{1}{r_{12}} \phi_m(r_2) \phi_n(r_2)dr_1 dr_2

        Parameters
        ----------
        C1 : NDArray
            Coefficient matrix for the first set of orbitals (index p).
        C2 : NDArray
            Coefficient matrix for the second set of orbitals (index q).

        Returns
        -------
        B : NDArray
            The B tensor in the form of a 3D array.
        """
        B = np.einsum(
            "Pmn,mi,nj->Pij",
            self.B_Pmn,
            C1.conj(),
            C2,
            optimize=True,
        )
        return B

    def B_tensor_gen_block_spinor(self, C1, C2):
        r"""
        Compute the spinorbital basis B tensor for a given set of spin-orbitals. This method is
        general and can handle different sets of orbitals for each index (p, q).

        The resulting B tensor is stored in a 3D array with the following convention:
        B[P,m,n] = :math:`(P | mn)`, where

        .. math::

            (P | mn) = \iint \phi_P(r_1)  \frac{1}{r_{12}} \phi_m(r_2) \phi_n(r_2)dr_1 dr_2

        Parameters
        ----------
        C1 : NDArray
            Coefficient matrix for the first set of spin-orbitals (index p).
        C2 : NDArray
            Coefficient matrix for the second set of spin-orbitals (index q).

        Returns
        -------
        B : NDArray
            The B tensor in the form of a 3D array.
        """
        nbf = self.nbf
        _a = slice(0, nbf)
        _b = slice(nbf, nbf * 2)
        B = np.einsum(
            "Pmn,mi,nj->Pij",
            self.B_Pmn,
            C1[_a, :].conj(),
            C2[_a, :],
            optimize=True,
        )
        B += np.einsum(
            "Pmn,mi,nj->Pij",
            self.B_Pmn,
            C1[_b, :].conj(),
            C2[_b, :],
            optimize=True,
        )
        return B


class FockBuilderOTF:
    """
    Class to build the Fock matrix on-the-fly without storing the B tensor. This is useful for large systems where storing the B tensor is not feasible.

    Parameters
    ----------
    system : System
        The system for which to build the Fock matrix.
    use_aux_corr : bool, optional, default=False
        If True, uses ``system.auxiliary_basis_corr`` instead of ``system.auxiliary_basis``.
    memory_threshold_mb : float, optional, default=4000
        The memory threshold in MB for deciding how to compute the J and K matrices. If the estimated memory requirement for storing the B tensor exceeds this threshold, the J and K matrices will be computed in a more memory-efficient way that does not require storing the B tensor.
    """

    def __init__(self, system, use_aux_corr=False, memory_threshold_mb=4000):
        self.system = system
        self.use_aux_corr = use_aux_corr
        self.nbf = system.nbf
        self.memory_threshold_mb = memory_threshold_mb
        self.auxbasis = (
            self.system.auxiliary_basis_corr
            if self.use_aux_corr
            else self.system.auxiliary_basis
        )
        self.basis = self.system.basis
        self.nshb = self.basis.nshells
        self.naux = len(self.auxbasis)
        self.nshaux = self.auxbasis.nshells
        self._build_metric()
        self._allocate_buffers()

    build_JK_generalized = FockBuilder.build_JK_generalized

    def _allocate_buffers(self):
        """
        We need three buffers:
        1. The Pmn buffer for storing the (P|mn) integrals for a block of auxiliary shells.
        2. The Qmi buffer for storing the largest intermediate in the K build for a block of occupied indices.
        3. The Pmi buffer to hold (P|Q)^{-1/2} (Q|mi).
        Therefore, for a given memory budget, we need to balance the three buffers. The Pmn buffer scales as pblksize * nbf^2, while the Qmi buffer scales as naux * nbf * iblksize. So heuristically we can set pblksize = (naux/nbf) * iblksize.
        """
        _cmplx = self.system.two_component
        nbytes = 16 if _cmplx else 8
        # number of buffers of variable types (Pmn is always real, but Pmi/Qmi are complex for two-component systems)
        nbuf_vt = 3 if _cmplx else 2
        # total size = 8 * nb^2 p + nbytes * nbuf_vt * nb * na * i ~= (nbuf_vt * nbytes + 8) * nb * na * i
        total_bytes_per_iblk = (nbuf_vt * nbytes + 8) * self.nbf * self.naux
        self.iblksize = min(
            self.nbf,
            math.ceil(self.memory_threshold_mb * 1024**2 / total_bytes_per_iblk),
        )
        self.pblksize = min(
            self.naux, math.ceil((self.naux / self.nbf) * self.iblksize)
        )

        self.aux_first_and_size = self.auxbasis.shell_first_and_size
        max_nbasis_in_shell = max(size for _, size in self.aux_first_and_size)
        if self.pblksize < max_nbasis_in_shell:
            suggested_mem_mb = math.ceil(
                (nbuf_vt * nbytes + 8) * max_nbasis_in_shell * self.naux**2 / 1024**2
            )
            raise ValueError(
                f"[FockBuilderOTF]: Memory threshold {self.memory_threshold_mb} is too low to even hold the largest shell of the auxiliary basis. Please increase the memory threshold to {suggested_mem_mb} MB."
            )
        # this buffer always holds a block of real (P|mn) integrals, even for two-component systems
        self._Pmn_buf = np.zeros((self.pblksize, self.nbf, self.nbf))
        alloc_size_mb_P = self.pblksize * self.nbf**2 * 8 / 1024**2
        logger.log_info1(
            f"[FockBuilderOTF]: Allocated buffer for ([P]|mn) with shape {self._Pmn_buf.shape} and size {alloc_size_mb_P:.2f} MB"
        )
        self._Qmi_buf = np.zeros(
            (self.naux, self.nbf, self.iblksize), dtype=complex if _cmplx else float
        )
        if _cmplx:
            self._Qmi_buf2 = np.zeros(
                (self.naux, self.nbf, self.iblksize), dtype=complex if _cmplx else float
            )
        self._Pmi_buf = np.zeros(
            (self.naux, self.nbf, self.iblksize), dtype=complex if _cmplx else float
        )
        alloc_size_mb_Q = self.naux * self.nbf * self.iblksize * nbytes / 1024**2
        logger.log_info1(
            f"[FockBuilderOTF]: Allocated buffers for X_Qm[i] and X_Pm[i] with shape {self._Qmi_buf.shape} and size {alloc_size_mb_Q*nbuf_vt:.2f} MB"
        )
        logger.log_info1(
            f"[FockBuilderOTF]: Memory budget: {self.memory_threshold_mb:.2f} MB, total allocated buffer size: {alloc_size_mb_P + nbuf_vt*alloc_size_mb_Q:.2f} MB"
        )
        self.cmplx = _cmplx

    def _require_complex(self):
        if not self.system.two_component:
            return
        if not self.cmplx:
            # deallocate all buffers first
            del self._Pmn_buf
            del self._Qmi_buf
            del self._Pmi_buf
            self._allocate_buffers()

    def _build_metric(self):
        # compute the metric (P|Q)^{-1}
        # M = (P|Q)
        M = integrals.coulomb_2c(self.system, self.auxbasis)
        # M = L L.T
        L = sp.linalg.cholesky(M, lower=True)
        # M^{-1} = L^{-T} L^{-1}
        # two triangular solves to get M^{-1}:
        # 1. solve L Y = I for Y = L^{-1}
        # 2. solve L.T X = Y for X = M^{-1}
        I = np.eye(M.shape[0])
        Y = sp.linalg.solve_triangular(L, I, lower=True)
        self.Mm1 = sp.linalg.solve_triangular(L.T, Y, lower=False)

        # M^{-1/2} = L^{-T}
        self.Mm12 = sp.linalg.solve_triangular(L, I, lower=True)

    def _find_aux_shell_block(self, pshell0):
        # find the block of auxiliary shells that fit in the buffer, starting from pshell0
        pshell1 = pshell0
        fs = self.aux_first_and_size
        nb = 0
        while pshell1 < self.nshaux and nb + fs[pshell1][1] <= self.pblksize:
            nb += fs[pshell1][1]
            pshell1 += 1
        pb0 = self.aux_first_and_size[pshell0][0]
        pb1 = (
            self.aux_first_and_size[pshell1 - 1][0]
            + self.aux_first_and_size[pshell1 - 1][1]
        )
        return pshell0, pshell1, pb0, pb1

    def _fill_Pmn_buffer(self, pshell0, pshell1):
        forte2.ints.coulomb_3c_by_shell(
            self.auxbasis,
            self.basis,
            self.basis,
            [(pshell0, pshell1), (0, self.nshb), (0, self.nshb)],
            self._Pmn_buf,
        )

    def _J_kernel(self, D):
        # 1. Batch over the (raw) P index of (P|mn) integrals
        pshell0, pshell1 = 0, 0
        bP = np.zeros(self.naux, dtype=D.dtype)
        while pshell0 < self.nshaux:
            pshell0, pshell1, pb0, pb1 = self._find_aux_shell_block(pshell0)
            # 2. Compute the B_Pmn blocks for the current batch of auxiliary shells
            self._fill_Pmn_buffer(pshell0, pshell1)
            np.einsum(
                "Pmn,nm->P",
                self._Pmn_buf[: pb1 - pb0, ...],
                D,
                optimize=True,
                out=bP[pb0:pb1],
            )
            pshell0 = pshell1
        bP = self.Mm1 @ bP

        J = np.zeros_like(D)
        # 3. Batch over the (raw) P index again to build the J matrix
        pshell0, pshell1 = 0, 0
        while pshell0 < self.nshaux:
            pshell0, pshell1, pb0, pb1 = self._find_aux_shell_block(pshell0)
            self._fill_Pmn_buffer(pshell0, pshell1)
            J += np.einsum(
                "Pmn,P->mn", self._Pmn_buf[: pb1 - pb0, ...], bP[pb0:pb1], optimize=True
            )
            pshell0 = pshell1

        return J

    def _K_kernel(self, C):
        i0, i1 = 0, 0
        K = np.zeros((self.nbf, self.nbf), dtype=C.dtype)
        nocc = C.shape[1]
        # batch over occupied indices
        while i0 < nocc:
            i1 = min(i1 + self.iblksize, nocc)
            ctemp = np.ascontiguousarray(C[:, i0:i1])
            # batch over the auxiliary shells
            pshell0, pshell1 = 0, 0
            while pshell0 < self.nshaux:
                pshell0, pshell1, pb0, pb1 = self._find_aux_shell_block(pshell0)
                self._fill_Pmn_buffer(pshell0, pshell1)
                np.einsum(
                    "Qms,si->Qmi",
                    self._Pmn_buf[: pb1 - pb0, ...],
                    ctemp,
                    optimize=True,
                    out=self._Qmi_buf[pb0:pb1, :, : i1 - i0],
                )
                pshell0 = pshell1
            np.einsum(
                "PQ,Qmi->Pmi",
                self.Mm12,
                self._Qmi_buf[:, :, : i1 - i0],
                optimize=True,
                out=self._Pmi_buf[:, :, : i1 - i0],
            )
            K += np.einsum(
                "Pmi,Pni->mn",
                self._Pmi_buf[:, :, : i1 - i0],
                self._Pmi_buf[:, :, : i1 - i0].conj(),
                optimize=True,
            )
            i0 = i1
        return K

    def _K_kernel_2c(self, Ca, Cb):
        if Ca.shape[1] != Cb.shape[1]:
            raise ValueError(
                "Ca and Cb must have the same number of columns (occupied orbitals)"
            )
        # K = [Kaa, Kab, Kbb], the Kba block is Kab^+
        K = [np.zeros((self.nbf, self.nbf), dtype=Ca.dtype) for _ in range(3)]
        nocc = Ca.shape[1]
        # batch over occupied indices
        i0, i1 = 0, 0
        while i0 < nocc:
            i1 = min(i1 + self.iblksize, nocc)
            catemp = np.ascontiguousarray(Ca[:, i0:i1])
            cbtemp = np.ascontiguousarray(Cb[:, i0:i1])
            # batch over the auxiliary shells
            pshell0, pshell1 = 0, 0
            while pshell0 < self.nshaux:
                pshell0, pshell1, pb0, pb1 = self._find_aux_shell_block(pshell0)
                self._fill_Pmn_buffer(pshell0, pshell1)
                np.einsum(
                    "Qms,si->Qmi",
                    self._Pmn_buf[: pb1 - pb0, ...],
                    catemp,
                    optimize=True,
                    out=self._Qmi_buf[pb0:pb1, :, : i1 - i0],
                )
                np.einsum(
                    "Qms,si->Qmi",
                    self._Pmn_buf[: pb1 - pb0, ...],
                    cbtemp,
                    optimize=True,
                    out=self._Qmi_buf2[pb0:pb1, :, : i1 - i0],
                )
                pshell0 = pshell1
            # _Pmi_buf holds the alpha block of Pmi
            np.einsum(
                "PQ,Qmi->Pmi",
                self.Mm12,
                self._Qmi_buf[:, :, : i1 - i0],
                optimize=True,
                out=self._Pmi_buf[:, :, : i1 - i0],
            )
            # _Qmi_buf holds the beta block of Pmi
            np.einsum(
                "PQ,Qmi->Pmi",
                self.Mm12,
                self._Qmi_buf2[:, :, : i1 - i0],
                optimize=True,
                out=self._Qmi_buf[:, :, : i1 - i0],
            )
            # aa contribution
            K[0] += np.einsum(
                "Pmi,Pni->mn",
                self._Pmi_buf[:, :, : i1 - i0],
                self._Pmi_buf[:, :, : i1 - i0].conj(),
                optimize=True,
            )
            # ab contribution
            K[1] += np.einsum(
                "Pmi,Pni->mn",
                self._Pmi_buf[:, :, : i1 - i0],
                self._Qmi_buf[:, :, : i1 - i0].conj(),
                optimize=True,
            )
            # bb contribution
            K[2] += np.einsum(
                "Pmi,Pni->mn",
                self._Qmi_buf[:, :, : i1 - i0],
                self._Qmi_buf[:, :, : i1 - i0].conj(),
                optimize=True,
            )
            i0 = i1
        return K

    def _JK_kernel(self, C):
        D = np.einsum("mi,ni->mn", C, C.conj(), optimize=True)
        # For the J build
        bP = np.zeros(self.naux, dtype=D.dtype)

        # The J build is 2-pass, and the number of passes for the K build
        # is ceil(nocc / iblksize)
        J_pass = 0
        J = np.zeros_like(D)
        K = np.zeros_like(D)
        nocc = C.shape[1]
        # batch over occupied indices
        i0, i1 = 0, 0
        while i0 < nocc:
            i1 = min(i1 + self.iblksize, nocc)
            ctemp = np.ascontiguousarray(C[:, i0:i1])
            # batch over the auxiliary shells
            pshell0, pshell1 = 0, 0
            while pshell0 < self.nshaux:
                pshell0, pshell1, pb0, pb1 = self._find_aux_shell_block(pshell0)
                self._fill_Pmn_buffer(pshell0, pshell1)
                np.einsum(
                    "Qms,si->Qmi",
                    self._Pmn_buf[: pb1 - pb0, ...],
                    ctemp,
                    optimize=True,
                    out=self._Qmi_buf[pb0:pb1, :, : i1 - i0],
                )

                if J_pass == 0:
                    np.einsum(
                        "Pmn,nm->P",
                        self._Pmn_buf[: pb1 - pb0, ...],
                        D,
                        optimize=True,
                        out=bP[pb0:pb1],
                    )
                elif J_pass == 1:
                    J += np.einsum(
                        "Pmn,P->mn",
                        self._Pmn_buf[: pb1 - pb0, ...],
                        bP[pb0:pb1],
                        optimize=True,
                    )
                pshell0 = pshell1
            np.einsum(
                "PQ,Qmi->Pmi",
                self.Mm12,
                self._Qmi_buf[:, :, : i1 - i0],
                optimize=True,
                out=self._Pmi_buf[:, :, : i1 - i0],
            )
            K += np.einsum(
                "Pmi,Pni->mn",
                self._Pmi_buf[:, :, : i1 - i0],
                self._Pmi_buf[:, :, : i1 - i0].conj(),
                optimize=True,
            )
            if J_pass == 0:
                bP = self.Mm1 @ bP
            J_pass += 1
            i0 = i1

        if J_pass == 1:
            # only one pass done for J, need to do the second pass
            pshell0, pshell1 = 0, 0
            while pshell0 < self.nshaux:
                pshell0, pshell1, pb0, pb1 = self._find_aux_shell_block(pshell0)
                self._fill_Pmn_buffer(pshell0, pshell1)
                J += np.einsum(
                    "Pmn,P->mn",
                    self._Pmn_buf[: pb1 - pb0, ...],
                    bP[pb0:pb1],
                    optimize=True,
                )
                pshell0 = pshell1
        return J, K

    def _JK_kernel_2c(self, Ca, Cb):
        if Ca.shape[1] != Cb.shape[1]:
            raise ValueError(
                "Ca and Cb must have the same number of columns (occupied orbitals)"
            )
        # J only needs the spin-diagonal blocks of D
        Da = np.einsum("mi,ni->mn", Ca, Ca.conj(), optimize=True)
        Db = np.einsum("mi,ni->mn", Cb, Cb.conj(), optimize=True)
        # For the J build
        bPa = np.zeros(self.naux, dtype=complex)
        bPb = np.zeros(self.naux, dtype=complex)

        # The J build is 2-pass, and the number of passes for the K build
        # is ceil(nocc / iblksize)
        J_pass = 0
        # [Jaa, Jbb]
        J = [np.zeros((self.nbf, self.nbf), dtype=complex) for _ in range(2)]
        # [Kaa, Kab, Kbb], Kba is Kab^+
        K = [np.zeros((self.nbf, self.nbf), dtype=complex) for _ in range(3)]
        nocc = Ca.shape[1]
        # batch over occupied indices
        i0, i1 = 0, 0
        while i0 < nocc:
            i1 = min(i1 + self.iblksize, nocc)
            catemp = np.ascontiguousarray(Ca[:, i0:i1])
            cbtemp = np.ascontiguousarray(Cb[:, i0:i1])
            # batch over the auxiliary shells
            pshell0, pshell1 = 0, 0
            while pshell0 < self.nshaux:
                pshell0, pshell1, pb0, pb1 = self._find_aux_shell_block(pshell0)
                self._fill_Pmn_buffer(pshell0, pshell1)
                np.einsum(
                    "Qms,si->Qmi",
                    self._Pmn_buf[: pb1 - pb0, ...],
                    catemp,
                    optimize=True,
                    out=self._Qmi_buf[pb0:pb1, :, : i1 - i0],
                )
                np.einsum(
                    "Qms,si->Qmi",
                    self._Pmn_buf[: pb1 - pb0, ...],
                    cbtemp,
                    optimize=True,
                    out=self._Qmi_buf2[pb0:pb1, :, : i1 - i0],
                )

                if J_pass == 0:
                    np.einsum(
                        "Pmn,nm->P",
                        self._Pmn_buf[: pb1 - pb0, ...],
                        Da,
                        optimize=True,
                        out=bPa[pb0:pb1],
                    )
                    np.einsum(
                        "Pmn,nm->P",
                        self._Pmn_buf[: pb1 - pb0, ...],
                        Db,
                        optimize=True,
                        out=bPb[pb0:pb1],
                    )
                elif J_pass == 1:
                    J[0] += np.einsum(
                        "Pmn,P->mn",
                        self._Pmn_buf[: pb1 - pb0, ...],
                        bPa[pb0:pb1],
                        optimize=True,
                    )
                    J[1] += np.einsum(
                        "Pmn,P->mn",
                        self._Pmn_buf[: pb1 - pb0, ...],
                        bPb[pb0:pb1],
                        optimize=True,
                    )
                pshell0 = pshell1
            np.einsum(
                "PQ,Qmi->Pmi",
                self.Mm12,
                self._Qmi_buf[:, :, : i1 - i0],
                optimize=True,
                out=self._Pmi_buf[:, :, : i1 - i0],
            )
            np.einsum(
                "PQ,Qmi->Pmi",
                self.Mm12,
                self._Qmi_buf2[:, :, : i1 - i0],
                optimize=True,
                out=self._Qmi_buf[:, :, : i1 - i0],
            )
            # Kaa contribution
            K[0] += np.einsum(
                "Pmi,Pni->mn",
                self._Pmi_buf[:, :, : i1 - i0],
                self._Pmi_buf[:, :, : i1 - i0].conj(),
                optimize=True,
            )
            # Kab contribution
            K[1] += np.einsum(
                "Pmi,Pni->mn",
                self._Pmi_buf[:, :, : i1 - i0],
                self._Qmi_buf[:, :, : i1 - i0].conj(),
                optimize=True,
            )
            # Kbb contribution
            K[2] += np.einsum(
                "Pmi,Pni->mn",
                self._Qmi_buf[:, :, : i1 - i0],
                self._Qmi_buf[:, :, : i1 - i0].conj(),
                optimize=True,
            )
            if J_pass == 0:
                bPa = self.Mm1 @ bPa
                bPb = self.Mm1 @ bPb
            J_pass += 1
            i0 = i1

        if J_pass == 1:
            # only one pass done for J, need to do the second pass
            pshell0, pshell1 = 0, 0
            while pshell0 < self.nshaux:
                pshell0, pshell1, pb0, pb1 = self._find_aux_shell_block(pshell0)
                self._fill_Pmn_buffer(pshell0, pshell1)
                J[0] += np.einsum(
                    "Pmn,P->mn",
                    self._Pmn_buf[: pb1 - pb0, ...],
                    bPa[pb0:pb1],
                    optimize=True,
                )
                J[1] += np.einsum(
                    "Pmn,P->mn",
                    self._Pmn_buf[: pb1 - pb0, ...],
                    bPb[pb0:pb1],
                    optimize=True,
                )
                pshell0 = pshell1
        return J, K

    def build_J(self, D):
        return [self._J_kernel(Di) for Di in D]

    def build_K(self, C):
        if self.system.two_component:
            self._require_complex()
            assert (
                len(C) == 1
            ), "C must be a list with one element for two-component systems."
            C = [C[0][: self.nbf, :], C[0][self.nbf :, :]]
            Kaa, Kab, Kbb = self._K_kernel_2c(C[0], C[1])
            Kba = np.copy(Kab.conj().T)
            return [Kaa, Kab, Kba, Kbb]
        else:
            return [self._K_kernel(Ci) for Ci in C]

    def build_JK(self, C):
        if self.system.two_component:
            self._require_complex()
            assert (
                len(C) == 1
            ), "C must be a list with one element for two-component systems."
            nbf = self.nbf
            Ca = C[0][:nbf, :]
            Cb = C[0][nbf:, :]
            [Jaa, Jbb], [Kaa, Kab, Kbb] = self._JK_kernel_2c(Ca, Cb)
            Kba = np.copy(Kab.conj().T)
            J_spinor = np.zeros((nbf * 2,) * 2, dtype=complex)
            K_spinor = np.zeros((nbf * 2,) * 2, dtype=complex)
            # fill the spinor J and K matrices from the spin blocks
            J_spinor[:nbf, :nbf] += Jaa + Jbb
            J_spinor[nbf:, nbf:] += Jaa + Jbb
            K_spinor[:nbf, :nbf] += Kaa
            K_spinor[nbf:, nbf:] += Kbb
            K_spinor[:nbf, nbf:] += Kab
            K_spinor[nbf:, :nbf] += Kba
            J, K = [J_spinor], [K_spinor]
        else:
            J = []
            K = []
            for Ci in C:
                Ji, Ki = self._JK_kernel(Ci)
                J.append(Ji)
                K.append(Ki)

        return J, K
