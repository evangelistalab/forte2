from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

# Prevents circular imports during type checking
if TYPE_CHECKING:
    from forte2 import System

# module-level tokens
o, v = object(), object()


@dataclass
class RestrictedMOIntegrals:
    r"""
    Class to compute molecular orbital integrals for a given set of restricted orbitals.

    Parameters
    ----------
    system : System
        The system for which to compute the integrals.
    C : NDArray
        The coefficient matrix for the molecular orbitals.
    orbitals : list[int]
        Subspace of the orbitals for which to compute the integrals.
    core_orbitals : list[int], optional
        Subspace of doubly occupied orbitals.
    use_aux_corr : bool, optional, default=False
        If True, use ``system.auxiliary_basis_corr``, else use ``system.auxiliary_basis``.
    antisymmetrize : bool, optional, default=False
        If True, antisymmetrize the two-electron integrals.

    Attributes
    ----------
    E : float
        Nuclear repulsion plus the core energy contribution.
    H : NDArray
        The effective one-electron integrals.
    V : NDArray
        The two-electron integrals stored in physicist's convention: V[p,q,r,s] = :math:`\langle pq | rs \rangle`.

    Examples
    --------
    >>> ints = RestrictedMOIntegrals(system=system, C=rhf.C[0], orbitals=orbitals, core_orbitals=core_orbitals)
    >>> ints.H  # one-electron integrals in the MO basis
    >>> ints.V  # two-electron integrals in the MO basis
    """

    system: "System"
    C: NDArray
    orbitals: list[int] | range
    core_orbitals: list[int] | range = field(default_factory=list)
    use_aux_corr: bool = False
    antisymmetrize: bool = False

    def __post_init__(self):
        self.norb = len(self.orbitals)
        if self.use_aux_corr:
            jkbuilder = self.system.fock_builder_corr
        else:
            jkbuilder = self.system.fock_builder
        C = np.ascontiguousarray(self.C[:, self.orbitals])

        # nuclear repulsion energy contribution to the energy
        self.E = self.system.nuclear_repulsion

        # build one-electron integrals
        H_ao = self.system.ints_hcore()

        # one-electron contributions to the one-electron integrals
        self.H = np.einsum("mi,mn,nj->ij", C, H_ao, C)

        if self.core_orbitals:
            # compute the J and K matrices contributions from the core orbitals
            Ccore = np.ascontiguousarray(self.C[:, self.core_orbitals])

            # one-electron contributions to the energy
            self.E += 2.0 * np.einsum("mi,mn,ni->", Ccore, H_ao, Ccore)
            J, K = jkbuilder.build_JK([Ccore])

            # two-electron contributions to the energy
            self.E += np.einsum("mi,mn,ni->", Ccore, 2 * J[0] - K[0], Ccore)

            # two-electron contributions to the one-electron integrals
            self.H += np.einsum("mi,mn,nj->ij", C, 2 * J[0] - K[0], C)

        # two-electron integrals
        self.V = jkbuilder.two_electron_integrals_block(C)
        if self.antisymmetrize:
            self.V -= self.V.swapaxes(2, 3)


@dataclass
class SpinorbitalIntegrals:
    r"""
    Class to compute molecular orbital integrals for a given set of restricted orbitals.

    Parameters
    ----------
    system : System
        The system for which to compute the integrals.
    C : NDArray, shape (2*nbf, *)
        The coefficient matrix for the spinorbitals.
    spinorbitals : list[int] | range
        Subspace of the spinorbitals for which to compute the integrals.
    core_spinorbitals : list[int] | range, optional
        Subspace of doubly occupied spinorbitals.
    use_aux_corr : bool, optional, default=False
        If True, use ``system.auxiliary_basis_corr``, else use ``system.auxiliary_basis``.
    antisymmetrize : bool, optional, default=False
        If True, antisymmetrize the two-electron integrals.

    Attributes
    ----------
    E : float
        Nuclear repulsion plus the core energy contribution.
    H : NDArray
        The effective one-electron integrals.
    V : NDArray
        The two-electron integrals stored in physicist's convention: V[p,q,r,s] = :math:`\langle pq | rs \rangle`.
    """

    system: "System"
    C: NDArray
    spinorbitals: list[int] | range
    core_spinorbitals: list[int] | range = field(default_factory=list)
    use_aux_corr: bool = False
    antisymmetrize: bool = False

    def __post_init__(self):
        assert self.system.two_component, "System must be two-component."
        assert (
            self.C.shape[0] == self.system.nbf * 2
        ), "C must be in the spinorbital basis."
        self.norb = len(self.spinorbitals)
        if self.use_aux_corr:
            jkbuilder = self.system.fock_builder_corr
        else:
            jkbuilder = self.system.fock_builder
        C = np.ascontiguousarray(self.C[:, self.spinorbitals])

        # nuclear repulsion energy contribution to the energy
        self.E = self.system.nuclear_repulsion

        # build one-electron integrals
        H_ao = self.system.ints_hcore()

        # one-electron contributions to the one-electron integrals
        self.H = np.einsum("mi,mn,nj->ij", C.conj(), H_ao, C)

        if len(self.core_spinorbitals) > 0:
            # compute the J and K matrices contributions from the core orbitals
            Ccore = np.ascontiguousarray(self.C[:, self.core_spinorbitals])

            # one-electron contributions to the energy
            self.E += np.einsum("mi,mn,ni->", Ccore.conj(), H_ao, Ccore)
            J, K = jkbuilder.build_JK([Ccore])

            # two-electron contributions to the energy
            self.E += 0.5 * np.einsum("mi,mn,ni->", Ccore.conj(), J[0] - K[0], Ccore)

            # two-electron contributions to the one-electron integrals
            self.H += np.einsum("mi,mn,nj->ij", C.conj(), J[0] - K[0], C)

        # two-electron integrals
        self.V = jkbuilder.two_electron_integrals_block_spinor(C)
        if self.antisymmetrize:
            self.V -= self.V.swapaxes(2, 3)
