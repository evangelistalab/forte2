import numpy as np
import scipy

from forte2 import integrals
from forte2.helpers import (
    logger,
    block_diag_2x2,
    i_sigma_dot,
    sigma_dot,
    canonical_orth,
    invsqrt_matrix,
    print_metric_info,
)
from forte2.system.build_basis import build_sap_potential_basis, decontract_basis

LIGHT_SPEED = 137.03599917697
SAP_BASIS_NAME = "sap_grasp_large"
ROW_Z_START = np.array([1, 3, 11, 19, 37, 55, 87])


def _row_given_Z(Z):
    return np.searchsorted(ROW_Z_START, Z, side="right")


class X2CHelper:
    """
    Helper class to compute the X2C one-electron Hamiltonian for a given system.

    Parameters
    ----------
    system : System
        The molecular system for which to compute the X2C Hamiltonian.

    Attributes
    ----------
    X : NDArray
        The decoupling matrix used in the X2C transformation.
    R : NDArray
        The renormalization matrix used in the X2C transformation.
    nbf : int
        The number of basis functions in the decontracted basis.

    Notes
    -----
    Implementation follows the general algorithm of J. Chem. Phys. 135, 084114 (2011),
    but adopts some numerical tricks from J. Chem. Phys. 131, 031104 (2009), especially
    for the spin-orbit case. See also PySCF's x2c module for reference. When
    ``system.x2c_model == "sap"``, the decoupling
    transformation follows the SAP-X2C
    Hamiltonian of Surjuse and Valeev, J. Chem. Theory Comput. 22, 3443--3452 (2026),
    https://doi.org/10.1021/acs.jctc.6c00032.
    """

    def __init__(self, system):
        self.system = system
        self.overlap_ortho_rtol = system.overlap_ortho_rtol
        self.x2c_type = system.x2c_type
        self.x2c_model = system.x2c_model
        self.snso_type = system.snso_type
        self.snso_target = system.snso_target

        logger.log_info1(f"Number of contracted basis functions: {self.system.nbf}")

        self.xbasis = decontract_basis(system.basis)
        self._spin_operator = None

        self.proj = scipy.linalg.solve(
            integrals.overlap(self.system, self.xbasis),
            integrals.overlap(self.system, self.xbasis, self.system.basis),
            assume_a="pos",
        )

        nbf_decon = len(self.xbasis)
        logger.log_info1(f"Number of decontracted basis functions: {nbf_decon}")

        self.S = integrals.overlap(self.system, self.xbasis)
        self.T = integrals.kinetic(self.system, self.xbasis)
        # the V and W integrals know about Gaussian nuclear charges
        self.V = integrals.nuclear(self.system, self.xbasis)
        self.W = integrals.opVop(self.system, self.xbasis)
        self.V_e = None
        self.W_e = None
        if self.x2c_model == "sap":
            logger.log_info1(
                f"Building the SAP-X2C screening potential with {SAP_BASIS_NAME}."
            )
            sap_basis = build_sap_potential_basis(
                SAP_BASIS_NAME, self.system.geom_helper
            )
            self.V_e = np.einsum(
                "Pmn->mn",
                integrals.coulomb_3c(
                    self.system,
                    sap_basis,
                    self.xbasis,
                    preserve_density_norm=True,
                ),
                optimize=True,
            )
            if integrals.LIBCINT_AVAILABLE:
                self.W_e = integrals.cint_coulomb_3c_opVop(
                    self.system, sap_basis, self.xbasis
                )
            else:
                self.W_e = integrals.coulomb_3c_opVop(
                    self.system, sap_basis, self.xbasis
                )

            # Enforce the exact permutation symmetry of the Pauli components.
            self.V_e = 0.5 * (self.V_e + self.V_e.T)
            self.W_e[0] = 0.5 * (self.W_e[0] + self.W_e[0].T)
            for component in range(1, 4):
                self.W_e[component] = 0.5 * (
                    self.W_e[component] - self.W_e[component].T
                )

        # Get orthonormal transformation for X2C
        self.Xorth_l, self.Xorthm1_l, self.orth_info = canonical_orth(
            self.S, self.overlap_ortho_rtol
        )
        print_metric_info(self.orth_info)
        logger.log_info1(
            f"Number of orthogonalized decontracted basis functions: {self.orth_info['n_kept']}"
        )

    def hcore_x2c(self):
        """
        Return the one-electron X2C core Hamiltonian matrix for the given system.

        Returns
        -------
        NDArray
            The X2C core Hamiltonian matrix in the contracted basis.
        """
        S, T, V, W = self._get_integrals()

        # build and solve the one-electron matrix Dirac equation
        _, c_dirac = self._solve_dirac_eq(S, T, V, W)

        # build the decoupling matrix X
        self.X = self._get_decoupling_matrix(c_dirac)

        # build the transformation matrix R
        self.R = self._get_transformation_matrix(S, T)

        # the picture-change correction of any property is tied to this X and R
        self._spin_operator = None

        # build the Foldy-Wouthuysen Hamiltonian
        h_fw = self._build_foldy_wouthuysen_hamiltonian(T, V, W)

        # Remove the untransformed screening potential to avoid double counting it in
        # subsequent mean-field or correlated treatments (SAP-X2C, Eq. 22).
        if self.x2c_model == "sap":
            h_fw -= self._get_sap_screening_potential()

        # return to original non-orthogonal AO basis
        _, Xorthm1 = self._get_Xorth()
        h_fw = Xorthm1.conj().T @ h_fw @ Xorthm1

        h_fw = self._apply_snso_to_hcore(h_fw)

        # project back to the contracted basis
        proj = self._get_projection_matrix()
        h_fw = proj.conj().T @ h_fw @ proj

        return h_fw

    def hcore_gradient(self, density):
        r"""Contract the analytic X2C Hamiltonian derivative with ``density``."""
        from .x2c_grad import compute_hcore_gradient

        return compute_hcore_gradient(self, density)

    def electric_dipole_moment(self, origin=None):
        """
        Compute the electric dipole moment integrals with picture change correction.

        Parameters
        ----------
        origin : array-like, optional
            The origin for the dipole operator. If None, defaults to [0, 0, 0].

        Returns
        -------
        list[NDArray]
            The picture-change-corrected electric dipole moment integrals along
            the x, y, z directions, in the contracted basis.
        """
        # the large-large (mu_ll) and small-small (mu_ss) dipole integrals are
        # built in the decontracted basis, matching the basis in which X and R
        # were constructed by hcore_x2c().
        if self.x2c_type == "sf":
            _, *mu_ll = integrals.emultipole1(self.system, self.xbasis, origin=origin)
            # only the spin-free (identity) part of the small-component dipole
            # is needed for the spin-free case: components [x, I2], [y, I2], [z, I2]
            mu_ss = integrals.cint_sprsp(self.system, self.xbasis, origin=origin)[
                [3, 7, 11]
            ]
            mu_ss = mu_ss * (0.25 / LIGHT_SPEED**2)
            mu_ss = list(mu_ss)
        else:  # so
            _, *mu_ll = integrals.emultipole1(self.system, self.xbasis, origin=origin)
            mu_ll = [block_diag_2x2(mu) for mu in mu_ll]
            mu_ss_so = integrals.cint_sprsp(self.system, self.xbasis, origin=origin)
            mu_ss_so = mu_ss_so * (0.25 / LIGHT_SPEED**2)
            # [x, y, z, I2] spin components for each Cartesian dipole direction
            mu_ss = [
                i_sigma_dot(mu_ss_so[3], *mu_ss_so[:3]),
                i_sigma_dot(mu_ss_so[7], *mu_ss_so[4:7]),
                i_sigma_dot(mu_ss_so[11], *mu_ss_so[8:11]),
            ]

        mu_pc = self.picture_change_even_operator(mu_ll, mu_ss)

        # project back to the contracted basis
        proj = self._get_projection_matrix()
        return [proj.conj().T @ mu_pc_i @ proj for mu_pc_i in mu_pc]

    def spin_operator(self):
        r"""
        Compute the spin operator matrices with picture change correction.

        Returns
        -------
        list[NDArray]
            The picture-change-corrected :math:`\hat{s}_x`, :math:`\hat{s}_y`, and
            :math:`\hat{s}_z` matrices in the two-component contracted basis, each of
            shape (2 * nbf, 2 * nbf).

        Notes
        -----
        :math:`\hat{s}_k = \sigma_k / 2` is an even operator, with large-large block
        :math:`\sigma_k S / 2` and small-small block
        :math:`(\sigma\cdot\hat{p}) \sigma_k (\sigma\cdot\hat{p}) / (8 c^2)`.

        For ``x2c_type == "sf"`` the decoupling matrices are spin-free, so the spin
        structure factors out of the transformation and each of the nine spatial blocks
        is picture-changed on its own before being recombined.
        """
        if self._spin_operator is not None:
            return self._spin_operator

        nbf = len(self.xbasis)
        ovlp = integrals.overlap(self.system, self.xbasis)
        # (sigma.p) sigma_k (sigma.p) as [k][sigma_j]; the four I2 blocks vanish
        ss = integrals.cint_spsigmasp(self.system, self.xbasis).reshape(3, 4, nbf, nbf)
        # 1/2 from s_k = sigma_k / 2, 1/(4 c^2) from the small-component metric
        fac = 0.5 * 0.25 / LIGHT_SPEED**2
        zero = np.zeros_like(ovlp)

        if self.x2c_type == "so":
            ints_LL, ints_SS = [], []
            for k in range(3):
                comp = [zero] * 3
                comp[k] = 0.5 * ovlp
                ints_LL.append(sigma_dot(*comp))
                ints_SS.append(fac * sigma_dot(*ss[k, :3]))
            s_pc = self.picture_change_even_operator(ints_LL, ints_SS)
            proj = self._get_projection_matrix()
            self._spin_operator = [proj.conj().T @ s @ proj for s in s_pc]
        else:
            ints_LL, ints_SS = [], []
            for k in range(3):
                for j in range(3):
                    ints_LL.append(0.5 * ovlp if j == k else zero)
                    ints_SS.append(fac * ss[k, j])
            s_pc = self.picture_change_even_operator(ints_LL, ints_SS)
            s_pc = [self.proj.conj().T @ s @ self.proj for s in s_pc]
            self._spin_operator = [
                sigma_dot(*s_pc[3 * k : 3 * k + 3]) for k in range(3)
            ]

        return self._spin_operator

    def magnetic_dipole_moment(self, origin=None):
        r"""
        Compute the magnetic dipole moment integrals with picture change correction.

        Parameters
        ----------
        origin : array-like, optional
            The gauge origin. If None, defaults to [0, 0, 0]. The uniform-field magnetic
            moment is gauge-origin dependent, so this is part of the definition of the
            property, not a numerical detail.

        Returns
        -------
        list[NDArray]
            The picture-change-corrected magnetic dipole moment integrals along x, y and
            z, in the two-component contracted basis, in atomic units.

        Notes
        -----
        A magnetic field enters the Dirac equation through the minimal substitution
        :math:`\hat{p} \to \hat{p} + \mathbf{A}/c`, which contributes
        :math:`\alpha\cdot\mathbf{A}`. The Dirac alpha matrices are block off-diagonal, so
        the magnetic moment :math:`\hat{m}_j = -\frac{1}{2}(\mathbf{r}\times\alpha)_j` is
        an **odd** operator, with large-small block
        :math:`-(\sigma\cdot\mathbf{A}^{(10)}_j)(\sigma\cdot\hat{p}) / 2c`.

        The :math:`1/2c` is the small-component normalization; it is what makes the Bohr
        magneton :math:`\mu_B = 1/2c` appear, so that the nonrelativistic limit is
        :math:`-(\hat{L}_j + 2\hat{S}_j)/2c`.

        Restricted kinetic balance expands the small component in
        :math:`(\sigma\cdot\hat{p})\chi`, which is adequate for the field-free problem
        but not for the magnetic response. Measured against the analytic Dirac g-factor of
        a hydrogenic 1s(1/2) level, this recovers about 63% of the relativistic correction
        to g, nearly independently of Z, and decontracting the basis does not improve it.
        A finite-difference solution of the four-component equation in the same
        representation reproduces the same value, so the shortfall belongs to restricted
        kinetic balance rather than to the transformation; removing it needs restricted
        magnetic balance. The nonrelativistic limit is exact.
        """
        nbf = len(self.xbasis)
        om = integrals.cint_cg_sa10sp(self.system, self.xbasis, origin=origin).reshape(
            3, 4, nbf, nbf
        )
        fac = -0.5 / LIGHT_SPEED

        if self.x2c_type == "so":
            # (sigma.A)(sigma.p) = A.p + i sigma.(A x p): a single p flips the reality of
            # the two halves relative to opVop, so the I2 block carries the i and the
            # three sigma blocks do not, and libcint returns the latter negated.
            ints_LS = [
                fac * (1j * block_diag_2x2(om[j, 3]) - sigma_dot(*om[j, :3]))
                for j in range(3)
            ]
            m_pc = self.picture_change_odd_operator(ints_LS)
            proj = self._get_projection_matrix()
            return [proj.conj().T @ m @ proj for m in m_pc]

        # spin-free: X and R carry no spin structure, so the identity channel and the
        # three sigma channels each transform on their own. Taking the conjugate
        # transpose channel by channel is the same as taking it of the assembled block.
        ints_LS = []
        for j in range(3):
            ints_LS += [fac * 1j * om[j, 3], *(fac * om[j, k] for k in range(3))]
        m_pc = self.picture_change_odd_operator(ints_LS)
        m_pc = [self.proj.conj().T @ m @ self.proj for m in m_pc]
        return [
            block_diag_2x2(m_pc[4 * j]) - sigma_dot(*m_pc[4 * j + 1 : 4 * j + 4])
            for j in range(3)
        ]

    def picture_change_even_operator(self, ints_LL, ints_SS):
        """
        Apply the picture change correction to integrals of an even operator, i.e.,
        one of the form [[M^LL, 0], [0, M^SS]]. Most "non-relativistic" property
        operators are even operators.

        Parameters
        ----------
        ints_LL : list[NDArray]
            The integrals corresponding to the large-large matrix block, in the
            (block-diagonalized, for the spin-orbit case) decontracted basis.
        ints_SS : list[NDArray]
            The integrals corresponding to the small-small matrix block, in the
            same basis as ``ints_LL``.

        Returns
        -------
        list[NDArray]
            The picture-change-corrected integrals in the (non-orthogonal)
            decontracted basis.

        Notes
        -----
        This operation is identical to the decoupling transformation of the 1e
        Dirac Hamiltonian: we compute R^+ M_NESC R, where M_NESC = M^LL + X^+ M^SS X.
        Because X and R are built in the orthonormal basis (see hcore_x2c), the
        input integrals are first transformed into that basis with Xorth and the
        result is transformed back with Xorthm1, mirroring how the core
        Hamiltonian is handled.
        """
        assert hasattr(self, "X") and hasattr(self, "R"), (
            "The X2C transformation has not been executed yet; "
            "call hcore_x2c() before requesting picture-change-corrected properties."
        )
        assert len(ints_LL) == len(
            ints_SS
        ), "ints_LL and ints_SS must have the same length."

        # X and R live in the orthonormal basis, so bring the operator integrals
        # into that basis, apply the decoupling, then return to the AO basis.
        Xorth, Xorthm1 = self._get_Xorth()
        res = []
        for M_LL, M_SS in zip(ints_LL, ints_SS):
            M_LL_o = Xorth.conj().T @ M_LL @ Xorth
            M_SS_o = Xorth.conj().T @ M_SS @ Xorth
            M_NESC = M_LL_o + self.X.conj().T @ M_SS_o @ self.X
            M_pc = self.R.conj().T @ M_NESC @ self.R
            res.append(Xorthm1.conj().T @ M_pc @ Xorthm1)
        return res

    def picture_change_odd_operator(self, ints_LS, ints_SL=None):
        """
        Apply the picture change correction to integrals of an odd operator, i.e.,
        one of the form [[0, M^LS], [M^SL, 0]]. Operators that couple the large and
        small components, such as magnetic property operators, are odd operators.

        Parameters
        ----------
        ints_LS : list[NDArray]
            The integrals corresponding to the large-small matrix block, in the
            (block-diagonalized, for the spin-orbit case) decontracted basis.
            Any prefactor required by the small-component metric (typically
            :math:`1/2c`) must already be folded in, as for ``ints_SS`` in
            :meth:`picture_change_even_operator`.
        ints_SL : list[NDArray], optional
            The integrals corresponding to the small-large matrix block, in the
            same basis as ``ints_LS``. If None, the operator is assumed Hermitian
            and each block is taken as the conjugate transpose of ``ints_LS``.

        Returns
        -------
        list[NDArray]
            The picture-change-corrected integrals in the (non-orthogonal)
            decontracted basis.

        Notes
        -----
        The picture change correction of a general operator is
        R^+ (M^LL + M^LS X + X^+ M^SL + X^+ M^SS X) R. For an odd operator the
        diagonal blocks vanish, leaving R^+ (M^LS X + X^+ M^SL) R. As in
        :meth:`picture_change_even_operator`, the integrals are transformed into the
        orthonormal basis that X and R are built in and the result is transformed
        back.
        """
        assert hasattr(self, "X") and hasattr(self, "R"), (
            "The X2C transformation has not been executed yet; "
            "call hcore_x2c() before requesting picture-change-corrected properties."
        )
        if ints_SL is None:
            ints_SL = [M_LS.conj().T for M_LS in ints_LS]
        assert len(ints_LS) == len(
            ints_SL
        ), "ints_LS and ints_SL must have the same length."

        Xorth, Xorthm1 = self._get_Xorth()
        res = []
        for M_LS, M_SL in zip(ints_LS, ints_SL):
            M_LS_o = Xorth.conj().T @ M_LS @ Xorth
            M_SL_o = Xorth.conj().T @ M_SL @ Xorth
            M_NESC = M_LS_o @ self.X + self.X.conj().T @ M_SL_o
            M_pc = self.R.conj().T @ M_NESC @ self.R
            res.append(Xorthm1.conj().T @ M_pc @ Xorthm1)
        return res

    @staticmethod
    def _build_nesc_matrix(T, V, W, X):
        return (
            T @ X
            + X.conj().T @ T
            - X.conj().T @ T @ X
            + V
            + (0.25 / LIGHT_SPEED**2) * X.conj().T @ W @ X
        )

    def _get_projection_matrix(self):
        return self.proj if self.x2c_type == "sf" else block_diag_2x2(self.proj)

    def _get_Xorth(self):
        if self.x2c_type == "sf":
            return self.Xorth_l, self.Xorthm1_l
        elif self.x2c_type == "so":
            return block_diag_2x2(self.Xorth_l), block_diag_2x2(self.Xorthm1_l)

    def _get_northo(self):
        if self.x2c_type == "sf":
            return self.orth_info["n_kept"]
        elif self.x2c_type == "so":
            return self.orth_info["n_kept"] * 2

    def _get_integrals(self):
        Xorth, _ = self._get_Xorth()
        V_ao = self.V
        W_ao = self.W
        if self.x2c_model == "sap":
            V_ao = self.V + self.V_e
            W_ao = [W + W_e for W, W_e in zip(self.W, self.W_e)]
        if self._snso_on_w():
            # Scale the spin-orbit part of W before the decoupling (the SNSO(W) ansatz),
            # so X and R -- and hence every picture-changed property -- see the screening.
            # _apply_snso_scaling writes in place, so hand it copies.
            W_ao = [W_ao[0]] + [self._apply_snso_scaling(W.copy()) for W in W_ao[1:]]
        if self.x2c_type == "sf":
            S = np.eye(Xorth.shape[1])
            T = Xorth.conj().T @ self.T @ Xorth
            V = Xorth.conj().T @ V_ao @ Xorth
            W = Xorth.conj().T @ W_ao[0] @ Xorth
        elif self.x2c_type == "so":
            S = np.eye(Xorth.shape[1], dtype=complex)
            T = Xorth.conj().T @ block_diag_2x2(self.T) @ Xorth
            V = Xorth.conj().T @ block_diag_2x2(V_ao) @ Xorth
            W = Xorth.conj().T @ i_sigma_dot(*W_ao) @ Xorth

        return S, T, V, W

    def _get_sap_screening_potential(self):
        """Return the untransformed SAP screening potential in the OAO basis."""
        Xorth, _ = self._get_Xorth()
        V_e = self.V_e
        if self.x2c_type == "so":
            V_e = block_diag_2x2(V_e)
        return Xorth.conj().T @ V_e @ Xorth

    def _solve_dirac_eq(self, S, T, V, W):
        dtype = np.float64 if self.x2c_type == "sf" else np.complex128
        north = self._get_northo()
        D = np.zeros((north * 2,) * 2, dtype=dtype)
        M = np.zeros((north * 2,) * 2, dtype=dtype)
        D[:north, :north] = V
        D[north:, north:] = (0.25 / LIGHT_SPEED**2) * W - T
        D[:north, north:] = T
        D[north:, :north] = T
        M[:north, :north] = S
        M[north:, north:] = (0.5 / LIGHT_SPEED**2) * T
        return scipy.linalg.eigh(D, M)

    def _get_decoupling_matrix(self, c_dirac):
        north = self._get_northo()
        clpos = c_dirac[:north, north:]
        cspos = c_dirac[north:, north:]
        return cspos @ scipy.linalg.pinv(clpos)

    def _get_transformation_matrix(self, S, T):
        """
        This implementation follows eqs 26-34 of J. Chem. Phys. 131, 031104 (2009),
        which avoids doing matrix inversions and leads to a more numerically stable transformation.
        """
        S_tilde = S + (0.5 / LIGHT_SPEED**2) * self.X.conj().T @ T @ self.X
        # S is guaranteed to be identity in the orthonormal basis
        # so we just need to compute the inverse square root of S_tilde
        # the tolerance used here isn't self.overlap_ortho_rtol because we're already in the
        # orthonormal basis, it's just an additional numerical guard against division by zero.
        S_tilde_m12, *_ = invsqrt_matrix(S_tilde, rtol=1e-12)
        return S_tilde_m12 @ S
        # This was the old way (Cheng and Gauss), worked fine for sfx2c1e, but seems unusable for sox2c1e
        # S_tilde = S + (0.5 / c0**2) * X.conj().T @ T @ X
        # Ssqrt = scipy.linalg.sqrtm(S)
        # S12 = forte2.helpers.invsqrt_matrix(S, tol=tol)
        # SSS = S12 @ S_tilde @ S12
        # SSS12 = forte2.helpers.invsqrt_matrix(SSS, tol=tol)
        # return S12 @ SSS12 @ Ssqrt

    def _build_foldy_wouthuysen_hamiltonian(self, T, V, W):
        L = self._build_nesc_matrix(T, V, W, self.X)
        return self.R.conj().T @ L @ self.R

    def _snso_on_w(self):
        """Whether the SNSO scaling is applied to W rather than to the Hamiltonian."""
        return (
            self.snso_type is not None
            and self.snso_target == "w"
            and self.x2c_model == "1e"
            and self.x2c_type == "so"
        )

    def _apply_snso_to_hcore(self, hcore):
        # SAP-X2C already screens the spin-orbit interaction, so SNSO is 1e-only.
        if self.x2c_model != "1e" or self.x2c_type != "so" or self.snso_type is None:
            return hcore
        if self.snso_target == "w":
            # already folded into W before the decoupling
            return hcore

        nbf = len(self.xbasis)
        haa = hcore[:nbf, :nbf]
        hab = hcore[:nbf, nbf:]
        hba = hcore[nbf:, :nbf]
        hbb = hcore[nbf:, nbf:]
        h0 = (haa + hbb) / 2
        h1 = self._apply_snso_scaling((hab + hba) / 2)
        h2 = self._apply_snso_scaling((hab - hba) / (-2j))
        h3 = self._apply_snso_scaling((haa - hbb) / 2)
        return np.block([[h0 + h3, h1 - 1j * h2], [h1 + 1j * h2, h0 - h3]])

    def _apply_snso_scaling(self, ints):
        """
        Apply the 'screened-nuclear-spin-orbit' (SNSO) scaling to the core Hamiltonian.
        Original paper ('Boettger'): Phys. Rev. B 62, 7809 (2000)
        Re-parameterized schemes ('DC'/'DCB'/'Row-dependent'): J. Chem. Theory Comput. 19, 5785 (2023)
        """
        # applied in the decontracted basis before recontraction (if requested)
        basis = self.xbasis
        atoms = self.system.atoms

        if self.snso_type is None:
            return ints
        if basis.max_l > 7:
            raise RuntimeError(
                "SNSO scaling is not implemented for basis sets with l > 7."
            )
        match self.snso_type:
            case "boettger":
                Ql = np.array([0.0, 2.0, 10.0, 28.0, 60.0, 110.0, 182.0, 280.0])
            case "dc":
                Ql = np.array([0.0, 2.32, 10.64, 28.38, 60.0, 110.0, 182.0, 280.0])
            case "dcb":
                Ql = np.array([0.0, 2.97, 11.93, 29.84, 64.0, 115.0, 188.0, 287.0])
            case "row-dependent":
                Ql = {
                    1: np.array([0.0, 2.97, 11.93, 29.84, 64.0, 115.0, 188.0, 287.0]),
                    2: np.array([0.0, 2.80, 11.93, 29.84, 64.0, 115.0, 188.0, 287.0]),
                    3: np.array([0.0, 2.95, 11.93, 29.84, 64.0, 115.0, 188.0, 287.0]),
                    4: np.array([0.0, 3.09, 11.49, 29.84, 64.0, 115.0, 188.0, 287.0]),
                    5: np.array([0.0, 3.02, 11.91, 29.84, 64.0, 115.0, 188.0, 287.0]),
                    6: np.array([0.0, 2.85, 12.31, 30.61, 64.0, 115.0, 188.0, 287.0]),
                    7: np.array([0.0, 2.85, 12.31, 30.61, 64.0, 115.0, 188.0, 287.0]),
                }
            case _:
                raise ValueError(
                    f"Invalid SNSO type: {self.snso_type}. Must be 'boettger', 'dc', 'dcb', or 'row-dependent'."
                )

        center_first = np.array([_[0] for _ in basis.center_first_and_last_shell])
        center_given_shell = (
            lambda ishell: np.searchsorted(center_first, ishell, side="right") - 1
        )

        iptr = jptr = 0
        for ishell in range(basis.nshells):
            isize = basis[ishell].size
            li = int(basis[ishell].l)
            if li == 0:
                iptr += isize
                jptr = 0
                continue
            Zi = atoms[center_given_shell(ishell)][0]
            if isinstance(Ql, dict):
                Ql_i = Ql[_row_given_Z(Zi)][li]
            else:
                Ql_i = Ql[li]
            for jshell in range(basis.nshells):
                jsize = basis[jshell].size
                lj = int(basis[jshell].l)
                if lj == 0:
                    jptr += jsize
                    continue
                Zj = atoms[center_given_shell(jshell)][0]
                if isinstance(Ql, dict):
                    Ql_j = Ql[_row_given_Z(Zj)][lj]
                else:
                    Ql_j = Ql[lj]
                snso_factor = 1 - np.sqrt(Ql_i * Ql_j / (Zi * Zj))
                ints[iptr : iptr + isize, jptr : jptr + jsize] *= snso_factor
                jptr += jsize
            iptr += isize
            jptr = 0

        return ints
