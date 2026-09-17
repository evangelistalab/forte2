from functools import cached_property

import numpy as np
import scipy as sp

from forte2 import integrals
from forte2.helpers import i_sigma_dot, invsqrt_matrix, logger, print_metric_info
from forte2.x2c.x2c import LIGHT_SPEED

# Assemble the two-component small-component B tensor in blocks of at most this
# many bytes, so that its memory footprint does not track the auxiliary basis size.
_ASSEMBLY_BLOCK_BYTES = 256 * 1024**2


class DiracFockBuilder:
    r"""
    Build four-component Coulomb (J) and exchange (K) matrices by density fitting.

    The Coulomb interaction is diagonal in the large/small component label, so the
    only surviving two-electron integral classes are :math:`(XX|YY)` with
    :math:`X, Y \in \{L, S\}`. Each factorizes through the same auxiliary basis,

    .. math::
        J^{XX} = \sum_P \rho_P B^X_P, \qquad
        K^{XY} = \sum_P B^X_P D^{XY} B^Y_P, \qquad
        \rho_P = \sum_X \mathrm{Tr}(B^X_P D^{XX}),

    where :math:`B^L` is the usual three-center tensor and :math:`B^S` is its
    small-component counterpart, built from
    :math:`(P|(\boldsymbol{\sigma}\cdot\mathbf{p})\phi_\mu
    (\boldsymbol{\sigma}\cdot\mathbf{p})\phi_\nu)` and scaled by :math:`(1/2c)^2`
    to match the scaled small component used by :func:`~forte2.scf.dirac.dirac_overlap`.

    The small-component tensor is stored as its four real Pauli components, four
    times the memory of the large-component tensor, and contracted into
    two-component form one auxiliary block at a time.

    Parameters
    ----------
    system : System
        The system for which to build the Fock matrix.
    c_light : float, optional
        The speed of light in atomic units.

    Attributes
    ----------
    B_Pmn : NDArray
        The large-component tensor with shape ``(naux, nbf, nbf)``. Lazily evaluated.
    W_Pmn : list[NDArray]
        The four Pauli components of the small-component tensor, each with shape
        ``(naux, nbf, nbf)``. Lazily evaluated.
    """

    def __init__(self, system, c_light=LIGHT_SPEED):
        self.system = system
        self.c_light = c_light
        self.nbf = system.nbf
        if system.cholesky_tei:
            raise NotImplementedError(
                "Four-component methods require density fitting; Cholesky-decomposed "
                "integrals only cover the large-component block."
            )
        if system.auxiliary_basis is None:
            raise ValueError(
                "Auxiliary basis is not defined. Define auxiliary_basis_set to run a "
                "four-component calculation."
            )

    @cached_property
    def _Mm12(self):
        M = integrals.coulomb_2c(self.system, self.system.auxiliary_basis)
        if self.system.df_ortho_rtol is not None:
            X, _, info = invsqrt_matrix(M, rtol=self.system.df_ortho_rtol)
            print_metric_info(info, "Density fitting Coulomb metric (P|Q)")
            return X
        try:
            L = sp.linalg.cholesky(M, lower=True)
        except sp.linalg.LinAlgError:
            raise ValueError(
                "Density fitting Coulomb metric (P|Q) is not positive definite.\n"
                "Please set df_ortho_rtol to a small positive value to orthogonalize "
                "the metric."
            )
        return sp.linalg.solve_triangular(L, np.eye(M.shape[0]), lower=True)

    @cached_property
    def B_Pmn(self):
        nbf, naux = self.nbf, self.system.naux
        logger.log_info1(
            f"Four-component B tensor memory: "
            f"{8 * 5 * naux * nbf**2 / 1024**3:.2f} GB"
        )
        Pmn = integrals.coulomb_3c(self.system, self.system.auxiliary_basis)
        return np.einsum("PQ,Qmn->Pmn", self._Mm12, Pmn, optimize=True)

    @cached_property
    def W_Pmn(self):
        scale = (0.5 / self.c_light) ** 2
        return [
            scale * np.einsum("PQ,Qmn->Pmn", self._Mm12, w, optimize=True)
            for w in integrals.coulomb_3c_spsp(self.system, self.system.auxiliary_basis)
        ]

    @property
    def naux(self):
        return self.B_Pmn.shape[0]

    def _aux_block_size(self):
        per_aux = 16 * (2 * self.nbf) ** 2
        return max(1, min(self.naux, _ASSEMBLY_BLOCK_BYTES // per_aux))

    def _assemble_small(self, lo, hi):
        """Return the two-component small-component tensor for one auxiliary block."""
        w0, w1, w2, w3 = (w[lo:hi] for w in self.W_Pmn)
        nbf = self.nbf
        out = np.empty((hi - lo, 2 * nbf, 2 * nbf), dtype=complex)
        for i in range(hi - lo):
            out[i] = i_sigma_dot(w0[i], w1[i], w2[i], w3[i])
        return out

    def _assemble_large(self, lo, hi):
        """Return the two-component large-component tensor for one auxiliary block."""
        nbf = self.nbf
        out = np.zeros((hi - lo, 2 * nbf, 2 * nbf), dtype=complex)
        out[:, :nbf, :nbf] = self.B_Pmn[lo:hi]
        out[:, nbf:, nbf:] = self.B_Pmn[lo:hi]
        return out

    def _density_fitted_charges(self, Dll, Dss, hermi=True):
        r"""Return :math:`\rho_P`, the fitted total charge density.

        Both components are contracted without assembling any two-component tensor:
        the spin trace collapses the large-component term, and the small-component
        term contracts each Pauli component against its own spin-block combination
        of the density.
        """
        nbf = self.nbf
        dl = Dll[:nbf, :nbf] + Dll[nbf:, nbf:]
        rho = np.einsum("Pmn,nm->P", self.B_Pmn, dl, optimize=True)

        # The Pauli decomposition B^S_P = sum_c w^c_P (x) tau_c with
        # tau = (I, i sigma_x, i sigma_y, i sigma_z) turns the spin trace into four
        # spin-block combinations of the small-component density.
        Daa, Dab, Dba, Dbb = (
            Dss[:nbf, :nbf],
            Dss[:nbf, nbf:],
            Dss[nbf:, :nbf],
            Dss[nbf:, nbf:],
        )
        ds = [Daa + Dbb, 1j * (Dab + Dba), Dba - Dab, 1j * (Daa - Dbb)]
        for w, d in zip(self.W_Pmn, ds):
            rho += np.einsum("Pmn,nm->P", w, d, optimize=True)
        # A Hermitian density traced against a Hermitian B gives a real charge.
        return rho.real if hermi else rho

    def _coulomb_from_charges(self, rho):
        """Contract the fitted charge density back into the two J blocks."""
        Jl = np.einsum("P,Pmn->mn", rho, self.B_Pmn, optimize=True)
        Js = i_sigma_dot(
            *(np.einsum("P,Pmn->mn", rho, w, optimize=True) for w in self.W_Pmn)
        )
        nbf = self.nbf
        Jl_2c = np.zeros((2 * nbf, 2 * nbf), dtype=complex)
        Jl_2c[:nbf, :nbf] = Jl
        Jl_2c[nbf:, nbf:] = Jl
        return Jl_2c, Js

    @property
    def core_energy_factor(self):
        """Prefactor relating Tr[(h + F) D] to the core energy for spinors."""
        return 0.5

    def hcore(self):
        """Return the four-component core Hamiltonian in the AO basis."""
        from forte2.scf.dirac import dirac_hcore

        return dirac_hcore(self.system, c_light=self.c_light)

    def build_JK(self, C):
        r"""
        Compute the Coulomb and exchange matrices for a set of occupied spinors.

        Parameters
        ----------
        C : list[NDArray]
            A single-element list holding the occupied coefficients, shape
            ``(4 nbf, nocc)``, matching the signature of
            :meth:`~forte2.jkbuilder.FockBuilder.build_JK`.

        Returns
        -------
        tuple(list[NDArray], list[NDArray])
            Single-element lists holding the Coulomb (J) and exchange (K) matrices.
        """
        assert (
            len(C) == 1
        ), "C must be a list with one element for four-component systems."
        J, K = self.build_JK_from_density(C[0] @ C[0].conj().T)
        return [J], [K]

    def build_JK_generalized(self, C, g1):
        r"""
        Compute Coulomb and exchange matrices for a correlated one-particle density.

        Parameters
        ----------
        C : NDArray
            Coefficients spanning the density, shape ``(4 nbf, n)``.
        g1 : NDArray
            The one-particle density matrix in that basis, shape ``(n, n)``.

        Returns
        -------
        tuple(NDArray, NDArray)
            The Coulomb (J) and exchange (K) matrices.
        """
        return self.build_JK_from_density(C @ g1 @ C.conj().T)

    def build_core_fock(self, C_core, hcore=None):
        """Build the core contribution to a generalized Fock matrix."""
        if hcore is None:
            hcore = self.hcore()
        J, K = self.build_JK_from_density(C_core @ C_core.conj().T)
        return hcore + J - K

    def build_active_fock(self, C_act, g1):
        """Build the active-density contribution to a generalized Fock matrix."""
        J, K = self.build_JK_generalized(C_act, g1)
        return J - K

    def build_generalized_fock(self, C_core, C_act, g1, hcore=None):
        """Build a multireference generalized Fock matrix in the AO basis."""
        return self.build_core_fock(C_core, hcore=hcore) + self.build_active_fock(
            C_act, g1
        )

    def B_tensor_gen_block_spinor(self, C1, C2):
        r"""
        Transform the three-center tensor into a block of four-component spinors.

        Summing the large- and small-component channels collapses the two integral
        classes into a single ``(naux, n1, n2)`` tensor, the same shape a
        two-component transform produces. That is what lets the downstream
        active-space and MCSCF machinery be reused unchanged.

        Parameters
        ----------
        C1, C2 : NDArray
            Coefficient matrices with ``4 nbf`` rows for the bra and ket indices.

        Returns
        -------
        NDArray
            The B tensor with shape ``(naux, C1.shape[1], C2.shape[1])``.
        """
        nbf = self.nbf
        n2c = 2 * nbf
        B = np.zeros((self.naux, C1.shape[1], C2.shape[1]), dtype=complex)
        # The large-component tensor is block diagonal in spin, so both spin blocks
        # contract against the same one-component tensor.
        for sl in (slice(0, nbf), slice(nbf, n2c)):
            B += np.einsum(
                "Pmn,mi,nj->Pij",
                self.B_Pmn,
                C1[sl, :].conj(),
                C2[sl, :],
                optimize=True,
            )
        C1s = np.ascontiguousarray(C1[n2c:, :].conj())
        C2s = np.ascontiguousarray(C2[n2c:, :])
        block = self._aux_block_size()
        for lo in range(0, self.naux, block):
            hi = min(lo + block, self.naux)
            B[lo:hi] += np.einsum(
                "Pmn,mi,nj->Pij",
                self._assemble_small(lo, hi),
                C1s,
                C2s,
                optimize=True,
            )
        return B

    def two_electron_integrals_gen_block_spinor(self, C1, C2, C3, C4):
        r"""Two-electron integrals :math:`\langle pq|rs\rangle` over spinor blocks."""
        return np.einsum(
            "Ppr,Pqs->pqrs",
            self.B_tensor_gen_block_spinor(C1, C3),
            self.B_tensor_gen_block_spinor(C2, C4),
            optimize=True,
        )

    def two_electron_integrals_block_spinor(self, C):
        r"""Two-electron integrals over a single set of four-component spinors."""
        return self.two_electron_integrals_gen_block_spinor(*(C,) * 4)

    def build_JK_from_density(self, D, hermi=True):
        r"""
        Compute the four-component Coulomb and exchange matrices.

        Parameters
        ----------
        D : NDArray
            The four-component density matrix, shape ``(4 nbf, 4 nbf)``.
        hermi : bool, optional, default=True
            Whether ``D`` is Hermitian, which lets the lower off-diagonal block of
            K be taken from the upper one.

        Returns
        -------
        tuple(NDArray, NDArray)
            The Coulomb (J) and exchange (K) matrices, each shape ``(4 nbf, 4 nbf)``.
        """
        n2c = 2 * self.nbf
        Dll = D[:n2c, :n2c]
        Dls = D[:n2c, n2c:]
        Dsl = D[n2c:, :n2c]
        Dss = D[n2c:, n2c:]

        J = np.zeros_like(D)
        K = np.zeros_like(D)

        rho = self._density_fitted_charges(Dll, Dss, hermi=hermi)
        J[:n2c, :n2c], J[n2c:, n2c:] = self._coulomb_from_charges(rho)

        block = self._aux_block_size()
        for lo in range(0, self.naux, block):
            hi = min(lo + block, self.naux)
            BL = self._assemble_large(lo, hi)
            BS = self._assemble_small(lo, hi)
            K[:n2c, :n2c] += np.einsum("Pms,sr,Prn->mn", BL, Dll, BL, optimize=True)
            K[:n2c, n2c:] += np.einsum("Pms,sr,Prn->mn", BL, Dls, BS, optimize=True)
            K[n2c:, n2c:] += np.einsum("Pms,sr,Prn->mn", BS, Dss, BS, optimize=True)
            if not hermi:
                K[n2c:, :n2c] += np.einsum("Pms,sr,Prn->mn", BS, Dsl, BL, optimize=True)
        if hermi:
            # A Hermitian density makes K Hermitian, so the lower block mirrors the
            # upper one.
            K[n2c:, :n2c] = K[:n2c, n2c:].conj().T
        return J, K
