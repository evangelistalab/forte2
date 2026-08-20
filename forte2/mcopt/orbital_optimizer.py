import numpy as np
import scipy as sp

from forte2.jkbuilder import FockBuilder


class OrbOptimizer:
    def __init__(
        self,
        C: np.ndarray,
        extents: list[slice],
        fock_builder: FockBuilder,
        hcore: np.ndarray,
        e_nuc: float,
        nrr: np.ndarray,
        compute_active_hessian: bool = False,
    ):
        self.core, self.actv, self.virt = extents
        self.C = C
        self.C0 = C.copy()
        self.Cgen = C
        self.Cact = C[:, self.actv]
        self.Ccore = C[:, self.core]
        self.ncore = self.Ccore.shape[1]
        self.nact = self.Cact.shape[1]
        self.nvirt = self.C.shape[1] - self.ncore - self.nact
        self.fock_builder = fock_builder
        self.hcore = hcore
        self.nrr = nrr
        self.nrot = self.nrr.sum()
        self.e_nuc = e_nuc
        self.compute_active_hessian = compute_active_hessian

        # the skew-hermitian rotation matrix, C_current = C_0 @ exp(R)
        self.R = np.zeros(self.nrot, dtype=float)
        # the unitary transformation matrix, U = exp(R)
        self.U = np.eye(self.C.shape[1], dtype=float)

    def get_eri_gaaa(self):
        self.eri_gaaa = self.fock_builder.two_electron_integrals_gen_block(
            self.Cgen, *(self.Cact,) * 3
        )
        return self.eri_gaaa

    def set_rdms(self, g1, g2):
        self.g1 = g1
        self.g2 = self._make_working_2rdm(g2)

    @staticmethod
    def _make_working_2rdm(g2):
        """Convert a spin-free 2-RDM to the orbital optimizer convention."""
        # '2RDM' defined as in [eq (6)]
        return 0.5 * (np.einsum("prqs->pqrs", g2) + np.einsum("qrps->pqrs", g2))

    def get_active_space_ints(self):
        """
        Returns the active space integrals.
        """
        return self.eri_gaaa[self.actv, ...]

    def evaluate(self, x):
        do_update_integrals = self._update_orbitals(x)
        if do_update_integrals:
            self._compute_Fcore()
            self.get_eri_gaaa()

        E_orb = self._compute_reference_energy()

        return E_orb

    def gradient(self, x):
        grad = self._compute_orbgrad()
        g = self._mat_to_vec(grad)
        return g

    def hess_diag(self, x):
        hess = self._compute_orbhess()
        h0 = self._mat_to_vec(hess)
        return h0

    def compute_orbital_hessian_vector_product(self, vector):
        r"""Apply the nonrelativistic orbital--orbital response matrix.

        Let :math:`\mathbb K=((p_I,q_I))_{I=0}^{n_\mathrm{rot}-1}` be the
        ordered orbital pairs for which ``nrr[p_I, q_I]`` is true.  Their order
        is exactly NumPy's C-order boolean-indexing order.  The input vector is
        embedded in an antisymmetric generator

        .. math::

            Z_{pq}(\mathbf z)
            = \sum_I z_I\left(
                \delta_{p p_I}\delta_{q q_I}
                - \delta_{p q_I}\delta_{q p_I}
              \right),

        and the method returns

        .. math::

            y_I
            = \sum_J (\mathcal A^{\mathrm{oo}})_{IJ}z_J
            = \left.\frac{d}{d\epsilon}
              \bar g_{p_I q_I}
              \left(Ce^{\epsilon Z(\mathbf z)};
                    \bar\gamma,\bar D\right)
              \right|_{\epsilon=0},

        where :math:`\bar g_{pq}=2(\bar A_{pq}-\bar A_{qp})` is Forte2's
        nonrelativistic orbital gradient.  Equivalently,
        :math:`(\mathcal A^{\mathrm{oo}})_{IJ}` is the derivative obtained by
        setting :math:`\mathbf z=\mathbf e_J` in the equation above.

        This definition assumes real, orthonormal restricted spatial orbitals
        and a mask containing exactly one orientation of every retained
        unordered pair.  It is a local derivative at the current ``C``.  The
        stored active 1-RDM ``g1`` and working 2-RDM ``g2`` (normally the
        state-averaged RDMs), AO Hamiltonian, DF integrals, and nuclear geometry
        are held fixed; only the orbital-dependent densities, Fock matrices,
        and MO integral transformations respond.  Thus this is the
        orbital--orbital partial Hessian, with neither CI nor nuclear response
        included.  The derivative is of the unscreened analytic gradient
        expression; the numerical ``1e-12`` screening in :meth:`gradient` is
        not differentiated.

        Parameters
        ----------
        vector : np.ndarray
            Real nonredundant orbital-rotation vector with shape ``(nrot,)``.

        Returns
        -------
        np.ndarray
            Orbital Hessian applied to ``vector``, in the same ordered-pair
            basis and with shape ``(nrot,)``.
        """
        vector = self._validate_orbital_response_vector(vector)
        intermediates = self._build_orbital_response_intermediates()
        return self._compute_orbital_hessian_vector_product(vector, intermediates)

    def compute_orbital_hessian(self):
        r"""Build the nonrelativistic orbital--orbital response matrix.

        Column :math:`j` is obtained by applying the same matrix-free response
        kernel used by :meth:`compute_orbital_hessian_vector_product` to unit
        vector :math:`j`.  The active RDMs are held fixed.

        Returns
        -------
        np.ndarray
            Dense orbital Hessian in the nonredundant rotation basis.
        """
        self._validate_nonrelativistic_orbital_response()
        intermediates = self._build_orbital_response_intermediates()
        hessian = np.empty((self.nrot, self.nrot), dtype=float)
        unit = np.zeros(self.nrot, dtype=float)
        for column in range(self.nrot):
            unit[column] = 1.0
            hessian[:, column] = self._compute_orbital_hessian_vector_product(
                unit, intermediates
            )
            unit[column] = 0.0
        return hessian

    def compute_orbital_lagrangian(self):
        r"""
        Return the symmetric CASSCF orbital Lagrangian matrix.

        The orbital optimizer forms the matrix :math:`A_{pq}` whose
        antisymmetric part is the orbital gradient,

        .. math::
            g_{pq} = 2(A_{pq} - A_{qp}).

        At a fully optimized state-specific CASSCF solution, the nonredundant
        antisymmetric part vanishes.  The symmetric part of :math:`A` is the
        molecular-orbital energy-weighted density used in the Pulay overlap
        derivative contribution,

        .. math::
            W^{S}_{\mu\nu}
            =
            C_{\mu p}
            \frac{1}{2}(A_{pq}+A_{qp})
            C_{\nu q}.

        Returns
        -------
        np.ndarray
            Symmetric orbital Lagrangian in the current MO basis.
        """
        self._compute_orbgrad()
        return 0.5 * (self.A_pq + self.A_pq.T.conj())

    def _update_orbitals(self, R):
        dR = R - self.R
        if np.max(np.abs(dR)) < 1e-12:
            # no change in orbitals, skip the update
            return False
        self.R += dR
        self.U = self.U @ self._expm(dR)

        self.C = self.C0 @ self.U
        self.Cgen = self.C
        self.Ccore = self.C[:, self.core]
        self.Cact = self.C[:, self.actv]
        return True

    def _expm(self, vec):
        M = self._vec_to_mat(vec)
        eM = sp.linalg.expm(M)
        return eM

    def _vec_to_mat(self, x):
        R = np.zeros_like(self.C)
        R[self.nrr] = x
        R += -R.T.conj()
        return R

    def _mat_to_vec(self, R):
        return R[self.nrr]

    def _validate_nonrelativistic_orbital_response(self):
        if self.fock_builder.system.two_component or np.iscomplexobj(self.C):
            raise NotImplementedError(
                "The orbital--orbital response is currently implemented only "
                "for nonrelativistic real orbitals."
            )

    def _validate_orbital_response_vector(self, vector):
        self._validate_nonrelativistic_orbital_response()
        vector = np.asarray(vector)
        if vector.shape != (self.nrot,):
            raise ValueError(
                f"Expected an orbital-response vector with shape ({self.nrot},), "
                f"got {vector.shape}."
            )
        if np.iscomplexobj(vector):
            raise TypeError("The nonrelativistic orbital-response vector must be real.")
        return vector.astype(float, copy=False)

    def _build_orbital_response_intermediates(self):
        r"""Build the current-orbital tensors shared by response applications.

        For real restricted orbitals, define the AO core and active densities

        .. math::

            P_{\mathrm{C}}=C_{\mathrm{C}}C_{\mathrm{C}}^{T},\qquad
            P_{\mathrm{A}}=C_{\mathrm{A}}\bar\gamma C_{\mathrm{A}}^{T}.

        This method returns the three tensors

        .. math::

            F_{\mathrm{C}}
            &=C^T\left(h+2J[P_{\mathrm{C}}]-K[P_{\mathrm{C}}]\right)C,\\
            \bar F_{\mathrm{A}}
            &=C^T\left(J[P_{\mathrm{A}}]-\tfrac12K[P_{\mathrm{A}}]\right)C,\\
            B^P_{pq}
            &=\sum_{\mu\nu}C_{\mu p}B^P_{\mu\nu}C_{\nu q}.

        Here ``g1`` is :math:`\bar\gamma`, :math:`B^P_{\mu\nu}` is the
        orthonormalized DF three-index tensor, and the AO maps are defined by

        .. math::

            J[D]_{\mu\nu}
            &=\sum_{P\rho\sigma}B^P_{\mu\nu}B^P_{\rho\sigma}D_{\sigma\rho},\\
            K[D]_{\mu\nu}
            &=\sum_{P\rho\sigma}B^P_{\mu\sigma}D_{\sigma\rho}B^P_{\nu\rho}.

        The first two returned arrays are in the full current MO basis; the
        third has all three-index factors transformed to that basis.

        These tensors contain no response themselves.  They are fixed base-
        point quantities that may be reused for multiple directions or all
        columns of the dense Hessian.  They are valid only while ``C``,
        ``Ccore``, ``Cact``, ``g1``, ``hcore``, and the DF tensor remain
        unchanged.  The working 2-RDM ``g2`` is not needed at this stage.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(Fcore_mo, Fact_mo, B_mo)`` with shapes ``(nmo, nmo)``,
            ``(nmo, nmo)``, and ``(naux, nmo, nmo)``, respectively.

        Raises
        ------
        NotImplementedError
            If the orbitals belong to a two-component or complex calculation.
        """
        self._validate_nonrelativistic_orbital_response()
        Fcore_ao = self.fock_builder.build_core_fock(self.Ccore, hcore=self.hcore)
        Fact_ao = self.fock_builder.build_active_fock(self.Cact, self.g1)
        Fcore_mo = self._transform_ao_operator(Fcore_ao, self.C)
        Fact_mo = self._transform_ao_operator(Fact_ao, self.C)
        B_mo = np.einsum(
            "Pmn,mp,nq->Ppq",
            self.fock_builder.B_Pmn,
            self.C,
            self.C,
            optimize=True,
        )
        return Fcore_mo, Fact_mo, B_mo

    def _build_density_fock_response(
        self, density_response, coulomb_factor, exchange_factor
    ):
        """Apply ``coulomb_factor * J[D] - exchange_factor * K[D]``."""
        B = self.fock_builder.B_Pmn
        J_response = self.fock_builder.build_J([density_response])[0]
        K_response = np.einsum("Pms,sr,Pnr->mn", B, density_response, B, optimize=True)
        return coulomb_factor * J_response - exchange_factor * K_response

    def _compute_orbital_hessian_vector_product(self, vector, intermediates):
        r"""Evaluate one fixed-RDM orbital Hessian--vector product.

        Pair ordering and the embedding :math:`\mathbf z\mapsto Z(\mathbf z)`
        are defined by :meth:`compute_orbital_hessian_vector_product`.  Starting
        from :math:`\dot C=CZ`, this kernel evaluates

        .. math::

            \dot P_{\mathrm{C}}
            &=\dot C_{\mathrm{C}}C_{\mathrm{C}}^{T}
              +C_{\mathrm{C}}\dot C_{\mathrm{C}}^{T},\\
            \dot P_{\mathrm{A}}
            &=\dot C_{\mathrm{A}}\bar\gamma C_{\mathrm{A}}^{T}
              +C_{\mathrm{A}}\bar\gamma\dot C_{\mathrm{A}}^{T},\\
            \dot F_{\mathrm{C}}^{\mathrm{AO}}
            &=2J[\dot P_{\mathrm{C}}]-K[\dot P_{\mathrm{C}}],\\
            \dot{\bar F}_{\mathrm{A}}^{\mathrm{AO}}
            &=J[\dot P_{\mathrm{A}}]-\tfrac12K[\dot P_{\mathrm{A}}].

        For :math:`X\in\{\mathrm C,\mathrm A\}`, the corresponding MO-basis
        response and the transformed DF-factor response are

        .. math::

            \dot F_X
            &=Z^T F_X+F_XZ+C^T\dot F_X^{\mathrm{AO}}C,\\
            \dot B^P_{pq}
            &=\sum_s\left(Z_{sp}B^P_{sq}+B^P_{ps}Z_{sq}\right).

        With :math:`V_{rvtw}=\sum_P B^P_{rt}B^P_{vw}`, the integral response is

        .. math::

            \dot V_{rvtw}
            =\sum_P\left(\dot B^P_{rt}B^P_{vw}
                         +B^P_{rt}\dot B^P_{vw}\right).

        Holding ``g1`` (:math:`\bar\gamma`) and the internal working 2-RDM
        ``g2`` (:math:`\bar D`) fixed, the orbital-Lagrangian response is

        .. math::

            \dot{\bar A}_{ri}
            &=2(\dot F_{\mathrm{C}}+\dot{\bar F}_{\mathrm{A}})_{ri},
              &&i\in\mathbb C,\\
            \dot{\bar A}_{ru}
            &=\sum_v(\dot F_{\mathrm{C}})_{rv}\bar\gamma_{vu}
              +\sum_{vtw}\dot V_{rvtw}\bar D_{tuvw},
              &&u\in\mathbb A,\\
            \dot{\bar A}_{re}&=0,
              &&e\in\mathbb V.

        The returned component for pair :math:`(p_I,q_I)` is therefore

        .. math::

            [\mathcal A^{\mathrm{oo}}\mathbf z]_I
            =2\left(\dot{\bar A}_{p_Iq_I}
                    -\dot{\bar A}_{q_Ip_I}\right).

        This private kernel assumes that ``vector`` is real with shape
        ``(nrot,)`` and that ``intermediates`` was built at the current orbitals
        and stored ``g1``.  It performs no validation.  The AO Hamiltonian, DF
        tensor, geometry, and both active RDMs are fixed; CI and nuclear response
        are excluded.  The analytic, unscreened gradient expression is
        differentiated.

        Parameters
        ----------
        vector : np.ndarray
            Real orbital-rotation direction in ``nrr`` C-order.
        intermediates : tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(Fcore_mo, Fact_mo, B_mo)`` returned by
            :meth:`_build_orbital_response_intermediates` at the same base point.

        Returns
        -------
        np.ndarray
            Fixed-RDM orbital Hessian applied to ``vector``, with shape
            ``(nrot,)`` and the same pair ordering.
        """
        Fcore_mo, Fact_mo, B_mo = intermediates
        Z = self._vec_to_mat(vector)
        C_response = self.C @ Z
        Ccore_response = C_response[:, self.core]
        Cact_response = C_response[:, self.actv]

        core_density_response = (
            Ccore_response @ self.Ccore.T + self.Ccore @ Ccore_response.T
        )
        active_density_response = (
            Cact_response @ self.g1 @ self.Cact.T
            + self.Cact @ self.g1 @ Cact_response.T
        )

        Fcore_ao_response = self._build_density_fock_response(
            core_density_response, coulomb_factor=2.0, exchange_factor=1.0
        )
        Fact_ao_response = self._build_density_fock_response(
            active_density_response,
            coulomb_factor=1.0,
            exchange_factor=0.5,
        )

        def transform_response(operator_mo, operator_ao_response):
            return (
                Z.T @ operator_mo
                + operator_mo @ Z
                + self._transform_ao_operator(operator_ao_response, self.C)
            )

        Fcore_response = transform_response(Fcore_mo, Fcore_ao_response)
        Fact_response = transform_response(Fact_mo, Fact_ao_response)

        B_response = np.einsum("rp,Prq->Ppq", Z, B_mo, optimize=True)
        B_response += np.einsum("Ppr,rq->Ppq", B_mo, Z, optimize=True)
        eri_response = np.einsum(
            "Prt,Pvw->rvtw",
            B_response[:, :, self.actv],
            B_mo[:, self.actv, self.actv],
            optimize=True,
        )
        eri_response += np.einsum(
            "Prt,Pvw->rvtw",
            B_mo[:, :, self.actv],
            B_response[:, self.actv, self.actv],
            optimize=True,
        )

        A_response = np.zeros_like(Fcore_response)
        A_response[:, self.core] = 2.0 * (Fcore_response + Fact_response)[:, self.core]
        A_response[:, self.actv] = np.einsum(
            "rv,vu->ru", Fcore_response[:, self.actv], self.g1, optimize=True
        )
        A_response[:, self.actv] += np.einsum(
            "rvtw,tuvw->ru", eri_response, self.g2, optimize=True
        )

        gradient_response = 2.0 * (A_response - A_response.T)
        return self._mat_to_vec(gradient_response)

    def _build_ci_orbital_response_intermediates(self):
        r"""Build fixed tensors for the orbital--CI response block.

        Returns the current inactive-core Fock matrix and transformed integral
        block

        .. math::

            F_{\mathrm C}=C^TF_{\mathrm C}^{\mathrm{AO}}C,\qquad
            V_{rvtw}=\langle rv|tw\rangle .

        Both are independent of a CI multiplier direction and can be reused
        for every column of the orbital--CI block.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The pair (Fcore_mo, eri_gaaa), with shapes (nmo, nmo) and
            (nmo, nact, nact, nact), respectively.
        """
        self._validate_nonrelativistic_orbital_response()
        Fcore_ao = self.fock_builder.build_core_fock(self.Ccore, hcore=self.hcore)
        Fcore_mo = self._transform_ao_operator(Fcore_ao, self.C)
        eri_gaaa = self.fock_builder.two_electron_integrals_gen_block(
            self.C, *(self.Cact,) * 3
        )
        return Fcore_mo, eri_gaaa

    def _compute_ci_orbital_response_from_rdms(
        self,
        overlap_response,
        g1_response,
        g2_response,
        intermediates,
    ):
        r"""Map symmetrized transition RDMs to the orbital equation.

        For a root-summed CI multiplier direction, overlap_response is s[x],
        while g1_response and g2_response are the raw spin-free, bra-plus-ket
        transition RDMs.  This routine evaluates

        .. math::

            A^{\mathrm{oc}}_{ri}
            &=2\left(s[\mathbf x](F_{\mathrm C})_{ri}
                    +(F_{\mathrm A}[\gamma[\mathbf x]])_{ri}\right),\\
            A^{\mathrm{oc}}_{ru}
            &=\sum_v(F_{\mathrm C})_{rv}\gamma_{vu}[\mathbf x]
              +\sum_{vtw}V_{rvtw}D_{tuvw}[\mathbf x],\\
            [\mathcal A^{\mathrm{oc}}\mathbf x]_I
            &=2\left(A^{\mathrm{oc}}_{p_Iq_I}
                    -A^{\mathrm{oc}}_{q_Ip_I}\right).

        The first equation applies to a core column i, the second to an active
        column u, and virtual columns are zero.  The working density D[x] is
        formed from g2_response using the same permutation and symmetrization
        as set_rdms.

        Parameters
        ----------
        overlap_response : float
            Root-summed bra-plus-ket transition overlap.
        g1_response : np.ndarray
            Root-summed spin-free transition 1-RDM, shape (nact, nact).
        g2_response : np.ndarray
            Root-summed spin-free transition 2-RDM, shape
            (nact, nact, nact, nact).
        intermediates : tuple[np.ndarray, np.ndarray]
            The pair (Fcore_mo, eri_gaaa) returned by
            _build_ci_orbital_response_intermediates at the same base point.

        Returns
        -------
        np.ndarray
            Orbital part of the orbital--CI action, with shape (nrot,) and
            nrr C-order.
        """
        Fcore_mo, eri_gaaa = intermediates
        active_density_response = self.Cact @ g1_response @ self.Cact.T
        Fact_ao_response = self._build_density_fock_response(
            active_density_response,
            coulomb_factor=1.0,
            exchange_factor=0.5,
        )
        Fact_response = self._transform_ao_operator(Fact_ao_response, self.C)
        g2_working_response = self._make_working_2rdm(g2_response)

        A_response = np.zeros_like(Fcore_mo)
        A_response[:, self.core] = (
            2.0 * (overlap_response * Fcore_mo + Fact_response)[:, self.core]
        )
        A_response[:, self.actv] = np.einsum(
            "rv,vu->ru", Fcore_mo[:, self.actv], g1_response, optimize=True
        )
        A_response[:, self.actv] += np.einsum(
            "rvtw,tuvw->ru",
            eri_gaaa,
            g2_working_response,
            optimize=True,
        )

        gradient_response = 2.0 * (A_response - A_response.T)
        return self._mat_to_vec(gradient_response)

    def _compute_reference_energy(self):
        energy = self.Ecore + self.e_nuc
        energy += np.einsum("uv,uv->", self.Fcore[self.actv, self.actv], self.g1)
        energy += 0.5 * np.einsum("tvuw,tuvw->", self.get_active_space_ints(), self.g2)
        return energy

    @staticmethod
    def _transform_ao_operator(operator, C):
        return np.einsum(
            "mp,mn,nq->pq",
            C.conj(),
            operator,
            C,
            optimize=True,
        )

    def _compute_Fcore(self):
        # Compute the core Fock matrix [eq (3)], also return the core energy
        Fcore_ao = self.fock_builder.build_core_fock(self.Ccore, hcore=self.hcore)
        self.Fcore = self._transform_ao_operator(Fcore_ao, self.Cgen)

        core_factor = 0.5 if self.fock_builder.system.two_component else 1.0
        self.Ecore = core_factor * np.trace(
            self._transform_ao_operator(
                self.hcore + Fcore_ao,
                self.Ccore,
            )
        )

    def _compute_Fact(self):
        # [eq (13)]
        Fact_ao = self.fock_builder.build_active_fock(self.Cact, self.g1)
        self.Fact = self._transform_ao_operator(Fact_ao, self.Cgen)

    def _compute_orbgrad(self):
        self._compute_Fact()
        orbgrad = np.zeros_like(self.Fcore)

        self.A_pq = np.zeros_like(self.Fcore)
        self.Fock = self.Fcore + self.Fact

        # compute A_ri (mo, core) block, [eq (10)]
        self.A_pq[:, self.core] = 2.0 * self.Fock[:, self.core]

        # compute A_ru (mo, active) block, [eq (11)]
        self.A_pq[:, self.actv] = np.einsum(
            "rv,vu->ru", self.Fcore[:, self.actv], self.g1
        )
        # (rt|vw) D_tu,vw, where (rt|vw) = <rv|tw>
        self.A_pq[:, self.actv] += np.einsum("rvtw,tuvw->ru", self.eri_gaaa, self.g2)

        # screen small gradients to prevent symmetry breaking
        self.A_pq[np.abs(self.A_pq) < 1e-12] = 0.0

        # compute g_rk (mo, core + active) block of gradient, [eq (9)]
        orbgrad = 2 * (self.A_pq - self.A_pq.T.conj())
        orbgrad *= self.nrr

        return orbgrad

    def _compute_orbhess(self):
        """Diagonal orbital Hessian"""
        orbhess = np.zeros_like(self.Fcore)
        diag_F = np.diag(self.Fock)
        diag_g1 = np.diag(self.g1)
        diag_grad = np.diag(self.A_pq)

        # The VC, VA, AC blocks are based on Theor. Chem. Acc. 97, 88-95 (1997)
        # compute virtual-core block
        orbhess[self.virt, self.core] = 4.0 * (
            diag_F[self.virt, None] - diag_F[None, self.core]
        )

        # compute virtual-active block
        orbhess[self.virt, self.actv] = 2.0 * (
            diag_F[self.virt, None] * diag_g1[None, :] - diag_grad[None, self.actv]
        )

        # compute active-core block
        orbhess[self.actv, self.core] = 4.0 * (
            diag_F[self.actv, None] - diag_F[None, self.core]
        )
        orbhess[self.actv, self.core] += 2.0 * (
            diag_F[None, self.core] * diag_g1[:, None] - diag_grad[self.actv, None]
        )

        # if GAS: compute active-active block [see SI of J. Chem. Phys. 152, 074102 (2020)]
        if self.compute_active_hessian:
            eri_actv = self.get_active_space_ints()
            # A. G^{uu}_{vv}
            Guu_ = np.einsum("uxuy,vvxy->uv", eri_actv, self.g2)
            Guu_ += 2.0 * np.einsum("uuxy,vxvy->uv", eri_actv, self.g2)
            Guu_ += np.diag(self.Fcore)[self.actv, None] * diag_g1[None, :]

            # B. G^{uv}_{vu}
            Guv_ = self.Fcore[self.actv, self.actv] * self.g1.T
            Guv_ += np.einsum("uxvy,vuxy->uv", eri_actv, self.g2)
            Guv_ += 2.0 * np.einsum("uvxy,vxuy->uv", eri_actv, self.g2)

            # compute diagonal hessian
            orbhess[self.actv, self.actv] = 2.0 * (Guu_ + Guu_.T)
            orbhess[self.actv, self.actv] -= 2.0 * (Guv_ + Guv_.T)
            orbhess[self.actv, self.actv] -= 2.0 * (
                diag_grad[self.actv, None] + diag_grad[None, self.actv]
            )
        orbhess *= self.nrr

        return orbhess


class RelOrbOptimizer(OrbOptimizer):
    def __init__(
        self,
        C: np.ndarray,
        extents: list[slice],
        fock_builder: FockBuilder,
        hcore: np.ndarray,
        e_nuc: float,
        nrr: np.ndarray,
        compute_active_hessian: bool = False,
    ):
        super().__init__(
            C,
            extents,
            fock_builder,
            hcore,
            e_nuc,
            nrr,
            compute_active_hessian,
        )
        self.R = self.R.astype(np.complex128)
        self.U = self.U.astype(np.complex128)

    def get_eri_gaaa(self):
        self.eri_gaaa = self.fock_builder.two_electron_integrals_gen_block_spinor(
            self.Cgen, *(self.Cact,) * 3
        )
        return self.eri_gaaa

    def set_rdms(self, g1, g2):
        self.g1 = g1
        # '2RDM' defined as in [eq (6)]
        self.g2 = g2.swapaxes(1, 2)

    def compute_orbital_lagrangian(self):
        """Return the Hermitian two-component CASSCF orbital Lagrangian."""
        self._compute_orbgrad()
        # RelOrbOptimizer stores its generalized Fock matrix with the
        # Lagrangian indices transposed relative to the AO transformation.
        return 0.5 * (self.Fock + self.Fock.T.conj()).T

    def _compute_reference_energy(self):
        energy = self.Ecore + self.e_nuc
        energy += np.einsum("uv,uv->", self.Fcore[self.actv, self.actv], self.g1)
        energy += 0.5 * np.einsum("tvuw,tuvw->", self.get_active_space_ints(), self.g2)
        return energy

    def _compute_orbgrad(self):
        self._compute_Fact()
        orbgrad = np.zeros_like(self.Fcore)

        self.Fock = np.zeros_like(self.Fcore)
        self.Fock1 = self.Fcore + self.Fact

        # compute A_ri (mo, core) block, [eq (10)]
        self.Fock[self.core, :] += self.Fock1[:, self.core].T

        # compute A_ru (mo, active) block, [eq (11)]
        self.Fock2 = np.zeros_like(self.Fcore)
        self.Fock2[self.actv, :] = np.einsum(
            "tu,qu->tq", self.g1, self.Fcore[:, self.actv], optimize=True
        )
        # (rt|vw) D_tu,vw, where (rt|vw) = <rv|tw>
        self.Fock2[self.actv, :] += np.einsum(
            "tuvw,qvuw->tq", self.g2, self.eri_gaaa, optimize=True
        )
        self.Fock[self.actv, :] += self.Fock2[self.actv, :]

        # screen small gradients to prevent symmetry breaking
        self.Fock[np.abs(self.Fock) < 1e-12] = 0.0

        orbgrad = -2 * (self.Fock - self.Fock.T.conj()).conj()
        orbgrad *= self.nrr

        return orbgrad

    def _compute_orbhess(self):
        """Diagonal orbital Hessian"""
        orbhess = np.zeros_like(self.Fcore)
        diag_F = np.diag(self.Fock1)
        diag_F2 = np.diag(self.Fock2)
        diag_g1 = np.diag(self.g1)
        diag_grad = np.diag(self.Fock)

        # The VC, VA, AC blocks are based on Theor. Chem. Acc. 97, 88-95 (1997)
        # compute virtual-core block
        orbhess[self.virt, self.core] += 2.0 * (
            diag_F[self.virt, None] - diag_F[None, self.core]
        )

        # compute virtual-active block
        orbhess[self.virt, self.actv] += 2.0 * (
            diag_F[self.virt, None] * diag_g1[None, :] - diag_F2[None, self.actv]
        )

        # compute active-core block
        orbhess[self.actv, self.core] += 2.0 * (
            diag_F[self.actv, None] - diag_F[None, self.core]
        )
        orbhess[self.actv, self.core] += 2.0 * (
            diag_g1[:, None] * diag_F[None, self.core] - diag_F2[self.actv, None]
        )

        # if GAS: compute active-active block [see SI of J. Chem. Phys. 152, 074102 (2020)]
        if self.compute_active_hessian:
            eri_actv = self.get_active_space_ints()
            # A. G^{uu}_{vv}
            Guu_ = np.einsum("uxuy,vvxy->uv", eri_actv, self.g2)
            Guu_ += 2.0 * np.einsum("uuxy,vxvy->uv", eri_actv, self.g2)
            Guu_ += np.diag(self.Fcore)[self.actv, None] * diag_g1[None, :]

            # B. G^{uv}_{vu}
            Guv_ = self.Fcore[self.actv, self.actv] * self.g1.T.conj()
            Guv_ += np.einsum("uxvy,vuxy->uv", eri_actv, self.g2)
            Guv_ += 2.0 * np.einsum("uvxy,vxuy->uv", eri_actv, self.g2)

            # compute diagonal hessian
            orbhess[self.actv, self.actv] = 2.0 * (Guu_ + Guu_.T.conj())
            orbhess[self.actv, self.actv] -= 2.0 * (Guv_ + Guv_.T.conj())
            orbhess[self.actv, self.actv] -= 2.0 * (
                diag_grad[self.actv, None] + diag_grad[None, self.actv]
            )
        orbhess = orbhess * self.nrr

        return orbhess
