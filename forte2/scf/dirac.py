import numpy as np

from forte2 import integrals
from forte2.helpers import block_diag_2x2, canonical_orth, i_sigma_dot, logger
from forte2.jkbuilder.dirac_jkbuilder import DiracFockBuilder
from forte2.system.build_basis import build_sap_potential_basis
from forte2.x2c.x2c import LIGHT_SPEED

SAP_BASIS_NAME = "sap_helfem_large"


def dirac_overlap(system, basis=None, c_light=LIGHT_SPEED):
    r"""
    Build the four-component overlap matrix in a restricted kinetically balanced basis.

    .. math::
        \mathbf{S} = \begin{bmatrix}
        \mathbf{S} \otimes \mathbf{I}_2 & \mathbf{0}\\
        \mathbf{0} & \mathbf{T} \otimes \mathbf{I}_2 / 2c^2
        \end{bmatrix}

    The small component is scaled by :math:`2c`, which is what makes the
    small-component metric :math:`\mathbf{T}/2c^2` rather than
    :math:`\langle\boldsymbol{\sigma}\cdot\mathbf{p}\chi|
    \boldsymbol{\sigma}\cdot\mathbf{p}\chi\rangle`. The two are equal because
    :math:`(\boldsymbol{\sigma}\cdot\mathbf{p})^2=p^2`, so no dedicated
    ``int1e_spsp`` integral is needed.

    Parameters
    ----------
    system : System
        The molecular system.
    basis : BasisSet, optional
        The orbital basis. If None, defaults to ``system.basis``.
    c_light : float, optional
        The speed of light in atomic units.

    Returns
    -------
    NDArray
        The ``(4 nbf, 4 nbf)`` overlap matrix.
    """
    S = block_diag_2x2(integrals.overlap(system, basis))
    T = block_diag_2x2(integrals.kinetic(system, basis))
    n2c = S.shape[0]
    out = np.zeros((2 * n2c,) * 2, dtype=complex)
    out[:n2c, :n2c] = S
    out[n2c:, n2c:] = (0.5 / c_light**2) * T
    return out


def dirac_hcore(system, basis=None, c_light=LIGHT_SPEED):
    r"""
    Build the four-component one-electron Dirac Hamiltonian.

    .. math::
        \mathbf{h} = \begin{bmatrix}
        \mathbf{V} \otimes \mathbf{I}_2 & \mathbf{T} \otimes \mathbf{I}_2\\
        \mathbf{T} \otimes \mathbf{I}_2 & \mathbf{W}/4c^2 - \mathbf{T} \otimes \mathbf{I}_2
        \end{bmatrix}

    where :math:`\mathbf{W}` assembles the four Pauli components of
    :math:`(\boldsymbol{\sigma}\cdot\mathbf{p})V(\boldsymbol{\sigma}\cdot\mathbf{p})`.
    This is the matrix Dirac equation that :class:`~forte2.x2c.x2c.X2CHelper`
    decouples.

    Parameters
    ----------
    system : System
        The molecular system.
    basis : BasisSet, optional
        The orbital basis. If None, defaults to ``system.basis``.
    c_light : float, optional
        The speed of light in atomic units.

    Returns
    -------
    NDArray
        The ``(4 nbf, 4 nbf)`` core Hamiltonian.
    """
    T = block_diag_2x2(integrals.kinetic(system, basis))
    V = block_diag_2x2(integrals.nuclear(system, basis))
    W = i_sigma_dot(*integrals.opVop(system, basis))
    n2c = T.shape[0]
    out = np.zeros((2 * n2c,) * 2, dtype=complex)
    out[:n2c, :n2c] = V
    out[:n2c, n2c:] = T
    out[n2c:, :n2c] = T
    out[n2c:, n2c:] = (0.25 / c_light**2) * W - T
    return out


def dirac_sap_hcore(system, basis=None, c_light=LIGHT_SPEED):
    """
    Build a Dirac Hamiltonian screened by a superposition of atomic potentials.

    The screening potential enters the large-large block as a plain three-center
    potential and the small-small block through its ``opVop`` counterpart, giving a
    four-component analogue of the SAP guess used by the non-relativistic SCF
    methods.

    Parameters
    ----------
    system : System
        The molecular system.
    basis : BasisSet, optional
        The orbital basis. If None, defaults to ``system.basis``.
    c_light : float, optional
        The speed of light in atomic units.

    Returns
    -------
    NDArray
        The ``(4 nbf, 4 nbf)`` screened Hamiltonian.
    """
    sap_basis = build_sap_potential_basis(SAP_BASIS_NAME, system.geom_helper)
    V_e = np.einsum(
        "Pmn->mn",
        integrals.coulomb_3c(system, sap_basis, basis, preserve_density_norm=True),
        optimize=True,
    )
    W_e = integrals.cint_coulomb_3c_opVop(system, sap_basis, basis)

    h = dirac_hcore(system, basis, c_light)
    n2c = h.shape[0] // 2
    h[:n2c, :n2c] += block_diag_2x2(V_e)
    h[n2c:, n2c:] += (0.25 / c_light**2) * i_sigma_dot(*W_e)
    return h


def dirac_orthogonalizer(
    system, basis=None, c_light=LIGHT_SPEED, rtol=None, kinetic_rtol=1e-12
):
    r"""
    Canonically orthogonalize the large and small metric blocks separately.

    The two blocks must be handled independently: the small-component metric is
    :math:`\mathbf{T}/2c^2`, some seven orders of magnitude below the large-component
    overlap, so orthogonalizing the assembled four-component metric with a single
    relative threshold silently discards small-component functions. When that
    happens, electronic states leak below the negative-energy branch and the aufbau
    occupation picks the wrong orbitals.

    Parameters
    ----------
    system : System
        The molecular system.
    basis : BasisSet, optional
        The orbital basis. If None, defaults to ``system.basis``.
    c_light : float, optional
        The speed of light in atomic units.
    rtol : float, optional
        Relative threshold for discarding large-component metric eigenvalues. If
        None, defaults to ``system.overlap_ortho_rtol``.
    kinetic_rtol : float, optional
        Relative threshold for the small-component metric. This is deliberately
        tighter than ``rtol``: the two thresholds answer different questions. The
        overlap threshold is a chemistry choice, discarding near-redundant diffuse
        combinations to stabilize the SCF. The kinetic metric instead spans the
        basis exponent range, so an uncontracted heavy-element basis gives it a
        condition number of 1e10 or more with no redundancy at all. Reusing the
        overlap threshold there discards small-component functions that are
        perfectly well determined, so this threshold asks only whether a direction
        is numerically present.

    Returns
    -------
    X : NDArray
        The ``(4 nbf, n_large + n_small)`` orthogonalization matrix.
    n_large : int
        The number of retained large-component functions.
    n_small : int
        The number of retained small-component functions, which is also the number
        of negative-energy solutions the Dirac equation will have in this basis.
    """
    if rtol is None:
        rtol = system.overlap_ortho_rtol
    XL, _, info_l = canonical_orth(integrals.overlap(system, basis), rtol)
    XS, _, info_s = canonical_orth(
        (0.5 / c_light**2) * integrals.kinetic(system, basis), kinetic_rtol
    )
    XL = block_diag_2x2(XL)
    XS = block_diag_2x2(XS)

    n2c, n_large = XL.shape
    n_small = XS.shape[1]
    X = np.zeros((2 * n2c, n_large + n_small), dtype=complex)
    X[:n2c, :n_large] = XL
    X[n2c:, n_large:] = XS

    if info_l["n_discarded"] or info_s["n_discarded"]:
        logger.log_info1(
            f"Discarded {info_l['n_discarded']} large-component and "
            f"{info_s['n_discarded']} small-component functions."
        )
    if n_large != n_small:
        # Restricted kinetic balance puts the two spaces in one-to-one
        # correspondence, so unequal dimensions mean one metric was truncated
        # differently from the other.
        logger.log_warning(
            f"The electronic ({n_large}) and negative-energy ({n_small}) spaces "
            "have different dimensions, so one of the two metrics was truncated "
            "more aggressively than the other. Check overlap_ortho_rtol."
        )
    return X, n_large, n_small


class DiracHamiltonian:
    """
    AO-basis Hamiltonian data for four-component methods.

    This stands in for :class:`~forte2.System` wherever a method asks for
    one-electron integrals, a Fock builder or the nuclear repulsion. Keeping it
    separate from the System leaves ``System.two_component`` and the
    one/two-component Fock builder untouched, so a four-component calculation can
    share a System with other methods.

    Parameters
    ----------
    system : System
        The molecular system.
    c_light : float, optional
        The speed of light in atomic units.
    """

    def __init__(self, system, c_light=LIGHT_SPEED):
        self.system = system
        self.c_light = c_light
        self.fock_builder = DiracFockBuilder(system, c_light=c_light)
        self.two_component = True
        self.four_component = True
        self._hcore = None

    @property
    def nbf(self):
        return self.system.nbf

    @property
    def nmo(self):
        return self.system.nmo

    @property
    def ao_dim(self):
        """Row dimension of a four-component AO coefficient matrix."""
        return 4 * self.system.nbf

    @property
    def nuclear_repulsion(self):
        return self.system.nuclear_repulsion

    @property
    def point_group(self):
        return self.system.point_group

    def ints_hcore(self):
        if self._hcore is None:
            self._hcore = dirac_hcore(self.system, c_light=self.c_light)
        return self._hcore

    def ints_overlap(self):
        return dirac_overlap(self.system, c_light=self.c_light)
