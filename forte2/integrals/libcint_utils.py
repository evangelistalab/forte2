import numpy as np
from scipy.special import gamma

from forte2.data import ATOM_DATA

# Slots of a libcint atm row
PTR_ZETA = 3
ATM_SLOTS = 6
# Slots of a libcint bas row
BAS_SLOTS = 8
# Reserved env pointers
PTR_COMMON_ORIG = 1
PTR_RINV_ORIG = 4
PTR_RINV_ZETA = 7
PTR_ENV_START = 20
# Nuclear models
NUC_POINT = 1
NUC_GAUSS = 2


class CintEnv:
    """
    Assemble the libcint ``atm``, ``bas``, and ``env`` arrays.

    Each basis is placed on its own shell centers as chargeless atoms, so
    bases at different geometries can share one environment. Nuclei that
    generate potentials are added separately with :meth:`add_nuclei`.
    """

    def __init__(self):
        self._atm = []
        self._bas = []
        self._env = [np.zeros(PTR_ENV_START)]
        self._env_size = PTR_ENV_START
        self._shell_ranges = []

    def add_nuclei(self, atoms, gaussian=False):
        """
        Add nuclear charges as libcint atoms.

        Parameters
        ----------
        atoms : list[tuple[int, list[float]]]
            The atomic numbers and positions (in bohr) of the nuclei.
        gaussian : bool, optional, default=False
            Use Gaussian nuclear charge distributions instead of point charges.
        """
        for charge, center in atoms:
            zeta = dyall_nuc_mod(charge) if gaussian else 0.0
            self._add_atom(charge, center, zeta)

    def add_basis(self, basis):
        """
        Add the shells of a basis set.

        Adding a basis that is already present returns its existing shells.

        Parameters
        ----------
        basis : Basis
            The basis set to add.

        Returns
        -------
        tuple[int, int]
            The first and past-the-last libcint shell indices of the basis.
        """
        for known, shell_range in self._shell_ranges:
            if known is basis:
                return shell_range
        first = len(self._bas)
        for first_shell, last_shell in basis.center_first_and_last_shell:
            atom = self._add_atom(0, basis[first_shell].center)
            for ishell in range(first_shell, last_shell):
                shell = basis[ishell]
                exponents = np.asarray(shell.exponents)
                coeffs = _normalize_contraction(
                    shell.l, exponents, np.asarray(shell.coeff)
                )
                ptr_exp = self._push(exponents)
                ptr_coeff = self._push(coeffs)
                # (atom, l, nprim, nctr, kappa, ptr_exp, ptr_coeff, unused)
                self._bas.append(
                    [atom, shell.l, len(exponents), 1, 0, ptr_exp, ptr_coeff, 0]
                )
        shell_range = (first, len(self._bas))
        self._shell_ranges.append((basis, shell_range))
        return shell_range

    def arrays(self, common_origin=None):
        """
        Return the libcint arrays.

        Parameters
        ----------
        common_origin : array-like, optional
            The origin of position operators. If None, the origin is (0, 0, 0).

        Returns
        -------
        tuple[ndarray, ndarray, ndarray]
            The ``atm``, ``bas``, and ``env`` arrays.
        """
        env = np.concatenate(self._env)
        if common_origin is not None:
            env[PTR_COMMON_ORIG : PTR_COMMON_ORIG + 3] = common_origin
        atm = np.asarray(self._atm, dtype=np.int32).reshape(-1, ATM_SLOTS)
        bas = np.asarray(self._bas, dtype=np.int32).reshape(-1, BAS_SLOTS)
        return atm, bas, env

    def _push(self, values):
        values = np.asarray(values, dtype=float).ravel()
        ptr = self._env_size
        self._env.append(values)
        self._env_size += values.size
        return ptr

    def _add_atom(self, charge, center, zeta=0.0):
        ptr = self._push([*center, zeta])
        nuc_mod = NUC_GAUSS if zeta > 0.0 else NUC_POINT
        # (charge, ptr_coord, nuc_mod, ptr_zeta, unused, unused)
        self._atm.append([charge, ptr, nuc_mod, ptr + 3, 0, 0])
        return len(self._atm) - 1


def _normalize_contraction(l, exponents, coeffs):
    # Radial self-overlap: int_0^inf r^(2l+2) exp(-(a+b) r^2) dr
    n1 = l + 1.5
    radial = gamma(n1) / (2.0 * (exponents[:, None] + exponents[None, :]) ** n1)
    return coeffs / np.sqrt(coeffs @ radial @ coeffs)


def dyall_nuc_mod(nuc_charge):
    """
    Return the Gaussian nuclear charge exponent of an element.

    The nuclear charge distribution is
    rho(r) = nuc_charge * (zeta / pi)^(3/2) * exp(-zeta * r^2).

    Ref. L. Visscher and K. Dyall, At. Data Nucl. Data Tables, 67, 207 (1997)
    """
    mass = ATOM_DATA[nuc_charge]["mass_number"]
    r = (0.836 * mass ** (1.0 / 3) + 0.570) / 52917.7249
    return 1.5 / (r**2)
