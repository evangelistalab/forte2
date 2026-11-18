import importlib.util
import os
import shutil
import tempfile
import weakref
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np

from forte2.lib import cpp_helpers
from forte2.state import State, MOSpace
from forte2.helpers import logger
from forte2.jkbuilder import RestrictedMOIntegrals
from forte2.base_classes import CIBase
from forte2.base_classes.params import DMRGParams
from forte2.ci.ci_utils import make_cumulant_from_rdms, validate_single_state_rdm
from .dmrg_utils import block2_2pdm_to_sf_2rdm

# Locate pyblock2 without importing it, so `import forte2` never loads block2's
# libraries. The import happens when the first DMRG calculation runs.
BLOCK2_AVAILABLE = importlib.util.find_spec("pyblock2") is not None


@dataclass
class _DMRGSingleStateSolver:
    """
    A spin-adapted DMRG solver for a single ``State``, backed by block2.

    This is the per-state worker of ``DMRGSolver``. Each worker owns a private
    scratch directory, so the workers of a state-averaged calculation never share
    MPS files.

    Parameters
    ----------
    mo_space : MOSpace
        The core and active orbitals. Only a single active space is supported.
    state : State
        The electronic state to solve for.
    ints : RestrictedMOIntegrals
        The active-space integrals.
    nroot : int
        The number of roots. More than one root gives a state-averaged MPS.
    active_orbsym : list[list[int]]
        The irrep of each active orbital, one list per GAS.
    dmrg_params : DMRGParams, optional
        The DMRG sweep parameters.
    die_if_not_converged : bool, optional, default=False
        If True, raise an error if the sweeps do not converge.
    log_level : int, optional
        The logging level. Defaults to ``logger.VERBOSITY_DEBUG``.

    Attributes
    ----------
    evals : NDArray
        The energy of each root.
    converged : bool
        Whether the energy changed by less than ``dmrg_params.tol`` over the last sweep.
    """

    mo_space: MOSpace
    state: State
    ints: RestrictedMOIntegrals
    nroot: int
    active_orbsym: list[list[int]]
    dmrg_params: DMRGParams = field(default_factory=DMRGParams)
    die_if_not_converged: bool = False
    log_level: int = logger.VERBOSITY_DEBUG

    ### Non-init attributes
    executed: bool = field(default=False, init=False)
    converged: bool = field(default=False, init=False)

    _rdm_orders: ClassVar[tuple[int, ...]] = (1, 2)
    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("sf",)
    _cumulant_orders: ClassVar[tuple[int, ...]] = (2,)
    _cumulant_spin_types: ClassVar[tuple[str, ...]] = ("sf",)

    def __post_init__(self):
        if self.mo_space.ngas > 1:
            raise ValueError(
                f"DMRGSolver supports a single active space, got {self.mo_space.ngas} GASes."
            )
        self.norb = self.mo_space.nactv
        self.ncore = self.mo_space.ncore + self.mo_space.nfrozen_core
        self.nactel = self.state.nel - 2 * self.ncore
        assert (
            self.nactel >= 0
        ), f"Number of active electrons {self.nactel} must be non-negative."
        self.orb_sym = [int(irrep) for irrep in self.active_orbsym[0]]
        # The active orbital at each site of the MPS chain
        self._order = np.arange(self.norb)
        # Single-root MPS tags, one per root, used to reload each root for its RDMs
        self._root_tags = []
        self._scratch = None

    def _make_driver(self):
        """
        Create a block2 driver on this worker's scratch directory.

        Note to devs: each block2 DMRGDriver allocates a DataFrame,
        which has stack_mem amount of memory reserved.

        block2 keeps one process-wide data frame, which a new driver takes over. A
        sibling worker may have taken it over since this worker last ran, so every
        block2 operation starts from a fresh driver.
        """
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes

        if self._scratch is None:
            if self.dmrg_params.scratch is not None:
                os.makedirs(self.dmrg_params.scratch, exist_ok=True)
            self._scratch = tempfile.mkdtemp(
                prefix="forte2_dmrg_", dir=self.dmrg_params.scratch
            )
            # clean up on python exit or garbage collection
            weakref.finalize(self, shutil.rmtree, self._scratch, ignore_errors=True)

        driver = DMRGDriver(
            stack_mem=self.dmrg_params.stack_memory << 20,
            scratch=self._scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=cpp_helpers.get_num_threads(),
        )
        driver.initialize_system(
            n_sites=self.norb,
            n_elec=self.nactel,
            spin=self.state.multiplicity - 1,
            pg_irrep=self.state.symmetry,
            orb_sym=[self.orb_sym[i] for i in self._order],
        )
        return driver

    def run(self):
        if self.norb == 0:
            self.evals = np.full(self.nroot, float(self.ints.E))
            self.converged = True
            self.executed = True
            return self

        h1e = np.asarray(self.ints.H)
        # block2 takes chemist-notation integrals, (pq|rs) = <pr|qs>; this is a view
        g2e = np.asarray(self.ints.V).transpose(0, 2, 1, 3)
        driver = self._make_driver()
        if self.dmrg_params.orbital_ordering != "original":
            self._order = np.asarray(
                driver.orbital_reordering(
                    h1e, g2e, method=self.dmrg_params.orbital_ordering
                )
            )
            # The orbital irreps the driver was set up with must follow the new order
            driver = self._make_driver()
            logger.log(
                f"Using orbital ordering {self._order} (active space relative indexing), "
                f"generated with the '{self.dmrg_params.orbital_ordering}' algorithm",
                self.log_level,
            )
        else:
            logger.log("Using original orbital ordering", self.log_level)
        o = self._order
        mpo = driver.get_qc_mpo(
            h1e=np.ascontiguousarray(h1e[np.ix_(o, o)]),
            # Fancy indexing returns a new contiguous array, the only copy of V made
            g2e=g2e[np.ix_(o, o, o, o)],
            ecore=float(self.ints.E),
            iprint=self.dmrg_params.iprint,
        )
        driver.bw.b.Random.rand_seed(self.dmrg_params.random_seed)
        ket = driver.get_random_mps(
            tag="GS", bond_dim=self.dmrg_params.bond_dims[0], nroots=self.nroot
        )
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=self.dmrg_params.n_sweeps,
            twosite_to_onesite=self.dmrg_params.twosite_to_onesite,
            tol=self.dmrg_params.tol,
            bond_dims=self.dmrg_params.bond_dims,
            noises=self.dmrg_params.noises,
            thrds=self.dmrg_params.davidson_tols,
            iprint=self.dmrg_params.iprint,
        )
        self.converged = self._sweeps_converged(driver)

        if self.nroot == 1:
            self._root_tags = ["GS"]
            self.evals = np.array([float(energy)])
        else:
            self._root_tags = [f"GS-{r}" for r in range(self.nroot)]
            for r, tag in enumerate(self._root_tags):
                driver.split_mps(ket, r, tag=tag)
            self.evals = np.array([float(e) for e in energy])

        if not self.converged and self.die_if_not_converged:
            raise RuntimeError(
                f"DMRG sweeps for state {self.state} did not converge to "
                f"tol={self.dmrg_params.tol}."
            )

        for i, e in enumerate(self.evals):
            logger.log(f"Final DMRG Energy Root {i}: {e:20.12f} [Eh]", self.log_level)

        self.executed = True
        return self

    def _sweeps_converged(self, driver):
        """Whether the energy of every root changed by less than tol over the last sweep."""
        # block2 keeps its per-sweep statistics on driver._dmrg
        energies = driver._dmrg.energies
        if len(energies) < 2:
            return True
        last = np.array(energies[-1], dtype=float)
        prev = np.array(energies[-2], dtype=float)
        return bool(np.max(np.abs(last - prev)) < self.dmrg_params.tol)

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: Literal[1, 2],
        spin_type: Literal["sf"],
    ):
        """
        Make the spin-free RDM of the given order for one DMRG root.

        Parameters
        ----------
        left_root : int
            The root index.
        right_root : int | None, optional
            Must be None or equal to ``left_root``.
        order : int
            The RDM order (1 or 2).
        spin_type : str
            "sf" (spin-free). The aliases "spin_free" and "spin-free" are also accepted.

        Returns
        -------
        NDArray
            The RDM.
        """
        validate_single_state_rdm(
            self,
            left_root,
            right_root,
            order,
            self._rdm_orders,
            spin_type,
            self._rdm_spin_types,
        )
        if right_root is not None and right_root != left_root:
            raise NotImplementedError(
                "DMRGSolver does not support transition RDMs yet."
            )
        if self.norb == 0:
            return np.zeros((0,) * (2 * order))
        driver = self._make_driver()
        ket = driver.load_mps(self._root_tags[left_root], nroots=1)
        pdm = driver.get_npdm(ket, pdm_type=order)
        # Back from MPS chain order to active-space order
        inv = np.argsort(self._order)
        pdm = pdm[np.ix_(*[inv] * pdm.ndim)]
        if order == 1:
            return np.ascontiguousarray(pdm)
        return block2_2pdm_to_sf_2rdm(pdm)

    def make_cumulant(
        self,
        root: int,
        *,
        order: Literal[2],
        spin_type: Literal["sf"],
    ):
        """
        Make the spin-free cumulant of the given order for one DMRG root.

        Parameters
        ----------
        root : int
            The root index.
        order : int
            The cumulant order (2).
        spin_type : str
            "sf" (spin-free). The aliases "spin_free" and "spin-free" are also accepted.

        Returns
        -------
        NDArray
            The cumulant.
        """
        spin_type = validate_single_state_rdm(
            self,
            root,
            None,
            order,
            self._cumulant_orders,
            spin_type,
            self._cumulant_spin_types,
        )
        return make_cumulant_from_rdms(self, root, order=order, spin_type=spin_type)

    def compute_natural_occupation_numbers(self):
        """
        Compute the natural occupation numbers from the 1-RDMs.

        Returns
        -------
        (norb, nroot) NDArray
            The natural occupation numbers of each root, in descending order.
        """
        if not self.executed:
            raise RuntimeError("DMRG solver has not been executed yet.")
        no = np.zeros((self.norb, self.nroot))
        for i in range(self.nroot):
            no[:, i] = np.linalg.eigvalsh(self.make_rdm(i, order=1, spin_type="sf"))[
                ::-1
            ]
        return no

    def set_ints(self, scalar, oei, tei):
        """Set the active-space integrals."""
        self.ints.E = scalar
        self.ints.H = oei
        self.ints.V = tei

    def reset_eigensolver(self, guess=None):
        """
        Do nothing. Each run starts from a random MPS, so there is no eigensolver
        state to reset, and ``guess`` is ignored.
        """

    def cleanup(self):
        """Remove the scratch directory."""
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None
        self._root_tags = []


@dataclass
class DMRGSolver(CIBase):
    """
    A spin-adapted density matrix renormalization group (DMRG) active-space solver,
    backed by block2.

    It takes the place of ``CISolver`` under the ``CI`` driver and supports state
    averaging over several states and roots. It requires block2 (``pyblock2``),
    which is not installed with forte2.

    Parameters
    ----------
    dmrg_params : DMRGParams, optional
        The DMRG sweep parameters. If not provided, default parameters are used.
    log_level : int, optional
        The logging level. Defaults to ``logger.VERBOSITY_DEBUG``.

    Attributes
    ----------
    sub_solvers : list[_DMRGSingleStateSolver]
        One DMRG solver per state.
    evals_per_solver : list[NDArray]
        The energies computed by each sub-solver.
    evals_flat, E : NDArray
        The energies of all roots of all states.
    E_avg : float
        The state-averaged energy.
    """

    dmrg_params: DMRGParams = field(default_factory=DMRGParams)

    # Active-space integral class
    _integrals_cls: ClassVar[type] = RestrictedMOIntegrals
    # Single state solver class
    _ss_solver_cls: ClassVar[type] = _DMRGSingleStateSolver

    _rdm_orders: ClassVar[tuple[int, ...]] = (1, 2)
    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("sf",)
    # each state has its own MPS, so there are no RDMs between roots of different states
    _rdm_cross_state_orders: ClassVar[tuple[int, ...]] = ()
    _cumulant_orders: ClassVar[tuple[int, ...]] = (2,)
    _cumulant_spin_types: ClassVar[tuple[str, ...]] = ("sf",)

    def __post_init__(self):
        if not BLOCK2_AVAILABLE:
            raise ImportError(
                "DMRGSolver requires block2, which is not installed. See "
                "'Install block2 for DMRG' in the Forte2 quickstart."
            )
        super().__post_init__()

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: Literal[1, 2],
        spin_type: Literal["sf"],
    ):
        """Spin-free RDM of the given order for one absolute DMRG root."""
        left_state, _, left_root_in_state, right_root_in_state, spin_type = (
            self._validate_rdm_inputs(
                left_root,
                right_root,
                order,
                self._rdm_orders,
                spin_type,
                self._rdm_spin_types,
                self._rdm_cross_state_orders,
            )
        )
        return self.sub_solvers[left_state].make_rdm(
            left_root_in_state, right_root_in_state, order=order, spin_type=spin_type
        )

    def get_convergence_status(self):
        """
        Get the convergence status of each sub-solver.

        Returns
        -------
        list[bool]
            Whether each sub-solver has converged.
        """
        return [dmrg_solver.converged for dmrg_solver in self.sub_solvers]

    def get_top_determinants(self, n=5):
        """
        Return an empty list for each root.

        An MPS has no explicit determinant expansion. This keeps the ``CISolver``
        interface that the drivers print from.
        """
        return [[] for _ in range(self.sa_info.nroots_sum)]

    def cleanup(self):
        """Remove the scratch directories of the sub-solvers."""
        for dmrg_solver in self.sub_solvers:
            dmrg_solver.cleanup()
