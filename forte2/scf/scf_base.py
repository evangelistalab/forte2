from dataclasses import dataclass, field
from abc import abstractmethod
import time

import numpy as np
from forte2.system import System, ModelSystem, BasisInfo
from forte2.base_classes import Method, MO
from forte2.helpers import logger, DIIS
from forte2.symmetry.symmetry_basis import SymmetryBasis
from forte2.symmetry.sym_utils import COTTON_LABELS
from .occupations import OccupationPolicy, validate_occupation_options


@dataclass
class SCFBase(Method):
    """
    Abstract base class for SCF calculations.

    Parameters
    ----------
    charge : int
        Charge of the system.
    do_diis : bool, optional, default=True
        Whether to perform DIIS acceleration.
    diis_start : int, optional, default=1
        Which iteration to start collecting DIIS error vectors.
    diis_nvec : int, optional, default=8
        How many DIIS error vectors to keep.
    diis_min : int, optional, default=2
        Minimum number of DIIS vectors to perform extrapolation.
    e_tol : float, optional, default=1e-9
        Energy convergence tolerance.
    d_tol : float, optional, default=1e-6
        RMS density change convergence tolerance.
    maxiter : int, optional, default=100
        Maximum iteration for SCF.
    guess_type : str, optional, default="minao"
        Initial guess type for the SCF calculation. Can be "minao" (SAP) or "hcore".
    level_shift : float, optional
        Level shift for the SCF calculation. If None, no level shift is applied.
    level_shift_thresh : float, optional, default=1e-5
        If energy change is below this threshold, level shift is turned off.
    die_if_not_converged : bool, optional, default=True
        Whether to raise an error if the SCF calculation does not converge.
    target_symmetry : str | int | None, optional
        Total determinant irrep, specified by label or irrep index. Not supported by GHF.
    irrep_occupations : dict | None, optional
        Occupied spatial orbitals per irrep for RHF, or (alpha, beta) counts for
        ROHF/UHF/CUHF. Unlisted irreps have zero occupation. Not supported by GHF.

    Attributes
    ----------
    C : list[NDArray]
        The MO coefficients.
    D : list[NDArray]
        The density matrices.
    E : float
        The total energy of the system.
    F : list[NDArray]
        The Fock matrices.
    eps : list[NDArray]
        The orbital energies.
    orbital_point_group : str
        Point group used for orbital labels and occupation constraints. GHF uses C1.
    state_symmetry : str
        Total determinant irrep.

    Raises
    ------
    RuntimeError
        If the SCF calculation does not converge within the maximum number of iterations.
    """

    charge: int
    do_diis: bool = True
    diis_start: int = 1
    diis_nvec: int = 8
    diis_min: int = 2
    e_tol: float = 1e-9
    d_tol: float = 1e-6
    maxiter: int = 100
    guess_type: str = "minao"
    level_shift: float = None
    level_shift_thresh: float = 1e-5
    die_if_not_converged: bool = True
    target_symmetry: str | int | None = None
    irrep_occupations: dict | None = None

    executed: bool = field(default=False, init=False)
    converged: bool = field(default=False, init=False)

    _occupation_type = None

    def __post_init__(self):
        self.provides = {"system", "mos", "eps"}
        validate_occupation_options(
            self.target_symmetry, self.irrep_occupations, self._occupation_type
        )

    def __call__(self, system):
        assert isinstance(
            system, (System, ModelSystem)
        ), "System must be an instance of forte2.System"
        self.system = system
        self.method = self._scf_type().upper()
        self.nel = self.system.Zsum - self.charge
        assert self.nel >= 0, "Number of electrons must be non-negative."

        if self.method != "GHF" and self.system.x2c_type == "so":
            raise ValueError(
                "SO-X2C is only available for GHF. Use SF-X2C for RHF/UHF."
            )

        self.C = None
        self.Xorth = self.system.get_Xorth()
        self._symmetry_basis = None
        self._orbital_irreps = None
        self._occupation_policy = None
        self.state_symmetry = None
        self.orbital_point_group = system.point_group
        self._validate_level_shift()
        self.called = True
        return self

    def _validate_level_shift(self):
        """Validate (and, for UHF, normalize) the configured level_shift."""
        if self.level_shift is not None:
            if isinstance(self.level_shift, (int, float)) and self.level_shift < 0.0:
                raise ValueError("level_shift must be non-negative.")
            if isinstance(self.level_shift, tuple) and self.method != "UHF":
                raise ValueError("Tuple level_shift is only valid for UHF.")
            if isinstance(self.level_shift, float) and self.method == "UHF":
                self.level_shift = (self.level_shift, self.level_shift)
            if isinstance(self.level_shift, tuple) and len(self.level_shift) != 2:
                raise ValueError("Tuple level_shift must have length 2 for UHF.")

    def _eigh(self, F):
        self._last_eigh_irreps = None
        if self._symmetry_basis is not None:
            eps, C, self._last_eigh_irreps = self._symmetry_basis.eigh(F)
            return eps, C
        Ftilde = self.Xorth.T @ F @ self.Xorth
        e, c = np.linalg.eigh(Ftilde)
        return e, self.Xorth @ c

    def _configure_occupation_constraints(self):
        validate_occupation_options(
            self.target_symmetry, self.irrep_occupations, self._occupation_type
        )
        self._occupation_policy = None
        if self.target_symmetry is None and self.irrep_occupations is None:
            return
        nelec = (
            (self.na,) if self._occupation_type == "restricted" else (self.na, self.nb)
        )
        self._occupation_policy = OccupationPolicy(
            self._occupation_type,
            self.orbital_point_group,
            nelec,
            self.target_symmetry,
            self.irrep_occupations,
        )

    def _initial_symmetry_eigh(self, F):
        # Initial guesses use the same symmetry blocks as subsequent iterations.
        if self._symmetry_basis is not None:
            eps, C, irreps = self._symmetry_basis.eigh(F)
        else:
            eps, C = SCFBase._eigh(self, F)
            irreps = np.zeros(len(eps), dtype=int)
        self._guess_eps = eps
        self._guess_irreps = irreps
        return eps, C

    def _prepare_initial_orbitals(self):
        if self._guess_eps is not None:
            # Guesses built here are already symmetry-adapted and in aufbau order.
            if self._occupation_policy is None:
                return
            eps = [self._guess_eps] * len(self.C)
            irreps = [self._guess_irreps] * len(self.C)
        elif self._symmetry_basis is not None:
            # Symmetry-adapt a supplied guess, ranking orbitals by their occupation in it.
            adapted = [
                self._symmetry_basis.adapt(C, -w)
                for C, w in zip(self.C, self._guess_occupations())
            ]
            eps, self.C, irreps = (list(x) for x in zip(*adapted))
        elif self._occupation_policy is not None:
            eps = [-w for w in self._guess_occupations()]
            irreps = [np.zeros(len(e), dtype=int) for e in eps]
        else:
            return
        self._orbital_irreps = irreps
        self.eps, self.C = self._apply_occupation_constraints(eps, self.C)

    def _guess_occupations(self):
        """Occupation numbers of a supplied guess, whose occupied orbitals come first."""
        index = np.arange(self.C[0].shape[1])
        if len(self.C) == 1:
            return [(index < self.na).astype(float) + (index < self.nb)]
        return [(index < n).astype(float) for n in (self.na, self.nb)]

    def _apply_occupation_constraints(self, eps, C):
        if self._occupation_policy is None:
            return eps, C
        orders = self._occupation_policy.permutations(eps, self._orbital_irreps)
        self._orbital_irreps = [
            h[order] for h, order in zip(self._orbital_irreps, orders)
        ]
        return [e[order] for e, order in zip(eps, orders)], [
            c[:, order] for c, order in zip(C, orders)
        ]

    def _setup_orbital_symmetry(self, S, H):
        """Build an orthonormal symmetry basis once per SCF run."""
        self._symmetry_basis = None
        if self.orbital_point_group != "C1":
            self._symmetry_basis = SymmetryBasis.build(
                self.system, self.basis_info, S, self.Xorth
            )
            # A symmetric density then gives a symmetric Fock matrix at every iteration.
            self._symmetry_basis.check_symmetric(H, "core Hamiltonian")
        if self._occupation_policy is not None:
            irreps = (
                self._symmetry_basis.irreps
                if self._symmetry_basis is not None
                else np.zeros(self.Xorth.shape[1], dtype=int)
            )
            self._occupation_policy.validate_capacity(irreps)

    def _scf_type(self):
        return type(self).__name__.upper()

    def run(self):
        """
        Run the SCF calculation.

        Returns
        -------
            self : SCFBase
                The SCF object.
        """
        self._validate_level_shift()
        self._configure_occupation_constraints()
        self._current_level_shift = self.level_shift
        start = time.monotonic()

        diis = DIIS(
            diis_start=self.diis_start,
            diis_nvec=self.diis_nvec,
            diis_min=self.diis_min,
            do_diis=self.do_diis,
        )
        Vnn = self._get_nuclear_repulsion()
        S = self._get_overlap()
        H = self._get_hcore()
        fock_builder = self.system.fock_builder

        self.nbf = self.system.nbf
        self.naux = self.system.naux
        self.nmo = self.system.nmo

        if isinstance(self.system, ModelSystem):
            self.basis_info = None
        else:
            self.basis_info = BasisInfo(self.system, self.system.basis)
        self._setup_orbital_symmetry(S, H)

        logger.log_info1(f"Number of electrons: {self.nel}")
        if self._scf_type() != "GHF":  # not good quantum numbers for GHF
            logger.log_info1(f"Number of alpha electrons: {self.na}")
            logger.log_info1(f"Number of beta electrons: {self.nb}")
            logger.log_info1(f"Ms: {self.ms}")
        logger.log_info1(f"Total charge: {self.charge}")
        logger.log_info1(f"Number of basis functions: {self.nbf}")
        logger.log_info1(f"Number of orthogonalized basis functions: {self.nmo}")
        logger.log_info1(f"Number of auxiliary basis functions: {self.naux}")
        logger.log_info1(f"Energy convergence criterion: {self.e_tol:e}")
        logger.log_info1(f"Density convergence criterion: {self.d_tol:e}")
        logger.log_info1(f"DIIS acceleration: {diis.do_diis}")
        logger.log_info1(f"\n==> {self.method} SCF ROUTINE <==")
        self.iter = 0
        self._guess_eps = None
        self._guess_irreps = None
        if self.C is None:
            self.C = self._initial_guess(H, guess_type=self.guess_type)
        self._prepare_initial_orbitals()
        self.D = self._build_density_matrix()
        F, F_canon = self._build_fock(H, fock_builder, S)
        self.F = F_canon
        self.E = Vnn + self._energy(H, F)

        Eold = self.E
        Dold = self.D
        self.iter += 1

        width = 81
        logger.log_info1("=" * width)
        logger.log_info1(
            f"{'Iter':>4s} {'Energy':>20s} {'ΔE':>12} {'||ΔD||':>12} {'||AO grad||':>12} {'<S^2>':>10} {'DIIS':>5s}"
        )
        logger.log_info1("-" * width)
        for iter in range(self.maxiter):
            # 1. Get the extrapolated Fock matrix
            AO_grad = self._build_ao_grad(S, F_canon)
            F_canon = self._diis_update(diis, F_canon, AO_grad)
            F_canon = self._apply_level_shift(F_canon, S)
            # 2. Diagonalize the extrapolated Fock
            self.eps, self.C = self._diagonalize_fock(F_canon)
            # 3. Build new density matrix
            self.D = self._build_density_matrix()
            # 4. Build the (non-extrapolated) Fock matrix
            # (there is a slot for canonicalized F to accommodate ROHF and CUHF methods - admittedly weird for RHF/UHF)
            F, F_canon = self._build_fock(H, fock_builder, S)
            self.F = F_canon
            # 5. Compute new HF energy from the non-extrapolated Fock matrix
            self.E = Vnn + self._energy(H, F)

            # check convergence parameters
            deltaE = self.E - Eold
            if np.abs(deltaE) < self.level_shift_thresh:
                self._current_level_shift = None
            deltaD = sum([np.linalg.norm(d - dold) for d, dold in zip(self.D, Dold)])
            self.S2 = self._spin(S)

            # print iteration
            logger.log_info1(
                f"{iter+1:4d} {self.E:20.12f} {deltaE:12.4e} {deltaD:12.4e} {np.linalg.norm(AO_grad):12.4e} {self.S2:10.5f} {diis.status:>5s}"
            )

            if np.abs(deltaE) < self.e_tol and deltaD < self.d_tol:
                logger.log_info1("=" * width)
                logger.log_info1(f"{self.method} iterations converged\n")
                # perform final iteration
                self.eps, self.C = self._diagonalize_fock(F_canon)
                self.D = self._build_density_matrix()
                F, F_canon = self._build_fock(H, fock_builder, S)
                self.F = F_canon
                self.E = Vnn + self._energy(H, F)
                logger.log_info1(f"Final {self.method} Energy: {self.E:20.12f}")
                AO_grad = self._build_ao_grad(S, F_canon)
                logger.log_info1(f"Final ||AO grad||: {np.linalg.norm(AO_grad):.4e}")
                self.converged = True
                break

            # reset old parameters
            Eold = self.E
            Dold = self.D
            self.iter += 1
        else:
            logger.log_info1("=" * width)
            logger.log_info1(f"{self.method} iterations did not converge")
            if self.die_if_not_converged:
                raise RuntimeError(
                    f"{self.method} did not converge in {self.maxiter} iterations."
                )
            else:
                logger.log_warning(
                    f"{self.method} did not converge in {self.maxiter} iterations."
                )

        end = time.monotonic()
        logger.log_info1(f"{self.method} time: {end - start:.2f} seconds")

        self._post_process()
        self.mos = MO(self.C, self.two_component, self.irrep_labels, self.irrep_indices)

        self.executed = True
        return self

    def _get_hcore(self):
        return self.system.ints_hcore()

    def _get_overlap(self):
        return self.system.ints_overlap()

    def _get_nuclear_repulsion(self):
        return self.system.nuclear_repulsion

    def _post_process(self):
        self._get_occupation()
        self._assign_orbital_symmetries()
        self._print_orbital_energies()
        self._print_ao_composition()

    @abstractmethod
    def _build_fock(self, H, fock_builder, S): ...

    @abstractmethod
    def _build_density_matrix(self): ...

    @abstractmethod
    def _initial_guess(self, H, guess_type="minao"): ...

    @abstractmethod
    def _build_ao_grad(self, S, F): ...

    def _diagonalize_fock(self, F):
        eps, C, irreps = [], [], []
        for f in F:
            e, c = self._eigh(f)
            eps.append(e)
            C.append(c)
            irreps.append(
                self._last_eigh_irreps
                if self._last_eigh_irreps is not None
                else np.zeros(len(e), dtype=int)
            )
        self._orbital_irreps = irreps
        return self._apply_occupation_constraints(eps, C)

    @abstractmethod
    def _spin(self, S): ...

    @abstractmethod
    def _energy(self, H, F): ...

    @abstractmethod
    def _diis_update(self, diis, F, AO_grad): ...

    @abstractmethod
    def _get_occupation(self): ...

    @abstractmethod
    def _print_orbital_energies(self): ...

    def _assign_orbital_symmetries(self):
        if self._orbital_irreps is None:
            raise RuntimeError(
                "Orbital symmetry metadata is missing after SCF diagonalization."
            )
        names = {
            index: label
            for label, index in COTTON_LABELS[self.orbital_point_group].items()
        }
        self.irrep_indices = [h.tolist() for h in self._orbital_irreps]
        self.irrep_labels = [[names[index] for index in h] for h in self.irrep_indices]
        self._assign_determinant_symmetry(names)
        if self._occupation_policy is not None:
            target = self._occupation_policy.target
            if target is not None and self.state_symmetry != names[target]:
                raise RuntimeError("HF determinant does not have target_symmetry.")
        logger.log_info1(f"HF determinant symmetry: {self.state_symmetry}")

    def _assign_determinant_symmetry(self, names):
        symmetry = 0
        occupations = (
            (self.nel,)
            if self.two_component
            else (
                (self.na, self.nb)
                if len(self._orbital_irreps) == 2
                else (max(self.na, self.nb),)
            )
        )
        for h, nocc in zip(self._orbital_irreps, occupations):
            if len(self._orbital_irreps) == 1 and not self.two_component:
                h = h[min(self.na, self.nb) : nocc]
            else:
                h = h[:nocc]
            symmetry ^= int(np.bitwise_xor.reduce(h, initial=0))
        self.state_symmetry = names[symmetry]

    @abstractmethod
    def _apply_level_shift(self, F, S): ...

    @abstractmethod
    def _print_ao_composition(self): ...
