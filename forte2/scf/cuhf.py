from dataclasses import dataclass
import numpy as np

from .scf_base import SCFBase
from .uhf import UHF


@dataclass
class CUHF(SCFBase):
    """
    A class that runs constrained unrestricted Hartree-Fock calculations.
    Equivalent to ROHF but uses UHF machinery.
    See J. Chem. Phys. 133, 141102 (2010) (10.1063/1.3503173)

    Parameters
    ----------
    ms : float
        Spin projection. Must be a multiple of 0.5.
    guess_mix : bool, optional, default=False
        If True, will mix the HOMO and LUMO orbitals to try to break alpha-beta degeneracy if ms is 0.0.
    """

    ms: float = None
    guess_mix: bool = False  # only used if ms == 0

    _parse_state = UHF._parse_state
    _build_density_matrix = UHF._build_density_matrix
    _initial_guess = UHF._initial_guess
    _build_ao_grad = UHF._build_ao_grad
    _diagonalize_fock = UHF._diagonalize_fock
    _spin = UHF._spin
    _energy = UHF._energy
    _diis_update = UHF._diis_update
    _build_total_density_matrix = UHF._build_total_density_matrix
    _get_occupation = UHF._get_occupation
    _print_orbital_energies = UHF._print_orbital_energies
    _assign_orbital_symmetries = UHF._assign_orbital_symmetries

    def __call__(self, system):
        system.two_component = False
        self = super().__call__(system)
        self._parse_state()
        return self

    def _build_fock(self, H, fock_builder, S):
        F, _ = UHF._build_fock(self, H, fock_builder, S)

        F_canon = self._build_canonical_fock(F, S)

        return F, F_canon

    def _build_canonical_fock(self, F, S):

        # Projection matrices for core (doubly occupied), open (singly occupied, alpha), and virtual (unoccupied) MO spaces
        U_core = np.dot(self.D[1], S)
        U_open = np.dot(self.D[0] - self.D[1], S)
        U_virt = np.eye(self.nbf) - np.dot(self.D[0], S)

        # Closed-shell Fock
        F_cs = 0.5 * (F[0] + F[1])

        def _project(u, v, f):
            return np.einsum("ur,uv,vt->rt", u, f, v, optimize=True)

        def _build_fock_eff(f):
            # these are scaled by 0.5 to account for fock + fock.T below
            fock = _project(U_core, U_core, f) * 0.5  # cc
            fock += _project(U_open, U_open, f) * 0.5  # oo
            fock += _project(U_virt, U_virt, f) * 0.5  # vv
            # off-diagonal blocks
            fock += _project(U_open, U_core, f)  # oc
            fock += _project(U_open, U_virt, f)  # ov
            fock += _project(U_virt, U_core, F_cs)  # Replace cv sector with fcore
            fock = fock + fock.conj().T
            return fock

        return [_build_fock_eff(f) for f in F]
