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
from forte2.ci.ci_utils import (
    make_cumulant_from_rdms,
    pack_sd_rdm,
    validate_single_state_rdm,
)
from .dmrg_utils import block2_pdm_to_rdm

# Locate pyblock2 without importing it, so `import forte2` never loads block2's
# libraries. The import happens when the first DMRG calculation runs.
BLOCK2_AVAILABLE = importlib.util.find_spec("pyblock2") is not None


@dataclass
class _DMRGSingleStateSolver:
    """
    A spin-adapted DMRG solver for a single ``State``, backed by block2.

    This is the per-state worker of ``DMRGSolver``. The workers of one solver share its
    scratch directory, with MPS tags that start with each worker's own prefix, so that
    one driver can load the MPS of any state.

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
    scratch : str
        The scratch directory, shared with the other workers of the solver.
    tag_prefix : str
        The prefix of this worker's MPS tags.
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
    scratch: str
    tag_prefix: str
    dmrg_params: DMRGParams = field(default_factory=DMRGParams)
    die_if_not_converged: bool = False
    log_level: int = logger.VERBOSITY_DEBUG

    ### Non-init attributes
    executed: bool = field(default=False, init=False)
    converged: bool = field(default=False, init=False)

    _rdm_orders: ClassVar[tuple[int, ...]] = (1, 2, 3)
    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("sd", "sf")
    _cumulant_orders: ClassVar[tuple[int, ...]] = (2, 3)
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
        # The tag of the MPS of all roots, which a warm restart reloads
        self._tag = f"{self.tag_prefix}GS"
        # Single-root MPS tags, one per root, used to reload each root for its RDMs
        self._root_tags = []
        # Whether the MPS of a previous run is on disk to restart from
        self._has_mps = False
        # The active orbital at each site of the MPS chain. The solver sets the same one
        # on all its workers before a cold start, so all states share one chain.
        self._order = None

    def _make_driver(self):
        """
        Create a block2 driver on the shared scratch directory.

        Note to devs: each block2 DMRGDriver allocates a DataFrame,
        which has stack_mem amount of memory reserved.

        block2 keeps one process-wide data frame, which a new driver takes over. A
        sibling worker may have taken it over since this worker last ran, so every
        block2 operation starts from a fresh driver.
        """
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes

        driver = DMRGDriver(
            stack_mem=self.dmrg_params.stack_memory << 20,
            scratch=self.scratch,
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
        p = self.dmrg_params
        warm = self._has_mps
        driver = self._make_driver()
        if warm:
            logger.log("Restarting DMRG from the previous MPS", self.log_level)
        o = self._order
        mpo = driver.get_qc_mpo(
            h1e=np.ascontiguousarray(h1e[np.ix_(o, o)]),
            # Fancy indexing returns a new contiguous array, the only copy of V made
            g2e=g2e[np.ix_(o, o, o, o)],
            ecore=float(self.ints.E),
            iprint=p.iprint,
        )
        twosite_to_onesite = p.twosite_to_onesite
        if self.norb == 2:
            # A two-site step solves the problem exactly
            twosite_to_onesite = None
        if warm:
            # Repeat only the final stretch of the schedule, so that the previous MPS
            # isn't first truncated to the smaller bond dimensions of the ramp
            start = self._final_stretch_start()
            onesite = twosite_to_onesite is not None and twosite_to_onesite <= start
            if twosite_to_onesite is not None:
                twosite_to_onesite = None if onesite else twosite_to_onesite - start
            ket = driver.load_mps(self._tag, nroots=self.nroot)
            ket, forward = driver.adjust_mps(ket, dot=1 if onesite else 2)
            if ket.n_sites == ket.dot:
                # adjust_mps takes center 0 to mean the left end, but here it is also
                # the right end. Sweep away from the end the wavefunction is on.
                forward = ket.canonical_form[-1] in "LR"
        else:
            start = 0
            forward = None
            driver.bw.b.Random.rand_seed(p.random_seed)
            ket = driver.get_random_mps(
                tag=self._tag, bond_dim=p.bond_dims[0], nroots=self.nroot
            )
        energy = driver.dmrg(
            mpo,
            ket,
            n_sweeps=p.n_sweeps - start,
            twosite_to_onesite=twosite_to_onesite,
            tol=p.tol,
            bond_dims=p.bond_dims[start:],
            noises=p.noises[start:],
            noise_type=p.noise_type,
            thrds=p.davidson_tols[start:],
            forward=forward,
            iprint=p.iprint,
        )
        self._has_mps = True
        last_change = self._last_sweep_change(driver)
        self.converged = last_change is None or last_change < p.tol

        if self.nroot == 1:
            self._root_tags = [self._tag]
            self.evals = np.array([float(energy)])
        else:
            self._root_tags = [f"{self._tag}-{r}" for r in range(self.nroot)]
            for r, tag in enumerate(self._root_tags):
                driver.split_mps(ket, r, tag=tag)
            self.evals = np.array([float(e) for e in energy])

        if not self.converged:
            message = (
                f"DMRG sweeps for state {self.state} did not converge: the energy "
                f"changed by {last_change:.1e} over the last sweep, above "
                f"tol={p.tol:.1e}."
            )
            if self.die_if_not_converged:
                raise RuntimeError(message)
            logger.log_warning(message)

        for i, e in enumerate(self.evals):
            logger.log(f"Final DMRG Energy Root {i}: {e:20.12f} [Eh]", self.log_level)

        self.executed = True
        return self

    def _final_stretch_start(self):
        """
        The first sweep from which the bond dimension, noise, and Davidson tolerance
        stay at their final values.
        """
        p = self.dmrg_params
        schedule = list(zip(p.bond_dims, p.noises, p.davidson_tols))[: p.n_sweeps]
        start = len(schedule) - 1
        while start > 0 and schedule[start - 1] == schedule[-1]:
            start -= 1
        return start

    def _last_sweep_change(self, driver):
        """
        The largest change in a root's energy over the last sweep, or None after a
        single sweep.
        """
        # block2 keeps its per-sweep statistics on driver._dmrg
        energies = driver._dmrg.energies
        if len(energies) < 2:
            return None
        last = np.array(energies[-1], dtype=float)
        prev = np.array(energies[-2], dtype=float)
        return float(np.max(np.abs(last - prev)))

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: Literal[1, 2, 3],
        spin_type: Literal["sd", "sf"],
        packed: bool = True,
    ):
        """
        Make the RDM of the given order and representation for two DMRG roots.

        Parameters
        ----------
        left_root : int
            The root for the bra state.
        right_root : int | None, optional (default=left_root)
            The root for the ket state.
        order : int
            The RDM order (1, 2, or 3).
        spin_type : str
            "sd" (spin-dependent) or "sf" (spin-free). The aliases "spin_dependent",
            "spin-dependent", "spin_free", and "spin-free" are also accepted.
        packed : bool, optional, default=True
            For spin_type=sd, if True, the same-spin index pairs and triplets are packed
            (see ``pack_sd_rdm``); if False, every block is a full tensor.

        Returns
        -------
        NDArray or tuple[NDArray, ...]
            spin_type=sd -> (a, b) at order 1, (aa, ab, bb) at order 2, and
            (aaa, aab, abb, bbb) at order 3; spin_type=sf -> a single full tensor.
        """
        spin_type = validate_single_state_rdm(
            self,
            left_root,
            right_root,
            order,
            self._rdm_orders,
            spin_type,
            self._rdm_spin_types,
        )
        if right_root is None:
            right_root = left_root
        return self._rdm(self, left_root, right_root, order, spin_type, packed)

    def _rdm(self, bra_solver, left_root, right_root, order, spin_type, packed):
        """
        Make the RDM between root left_root of bra_solver, which may be another worker
        of the same solver, and root right_root of this worker, in forte2's layout.
        """
        if self.norb == 0:
            nblocks = order + 1 if spin_type == "sd" else 1
            pdms = [np.zeros((0,) * (2 * order)) for _ in range(nblocks)]
        else:
            pdms = self._npdms(
                bra_solver, left_root, right_root, order, spin_type == "sd"
            )
        # Back from MPS chain order to active-space order
        inv = np.argsort(self._order)
        rdms = [block2_pdm_to_rdm(pdm[np.ix_(*[inv] * pdm.ndim)]) for pdm in pdms]
        if spin_type == "sf":
            return rdms[0]
        return pack_sd_rdm(rdms) if packed else tuple(rdms)

    def _npdms(self, bra_solver, left_root, right_root, order, spin_resolved):
        """
        Compute the block2 PDMs between root left_root of bra_solver and root right_root
        of this worker: the spin-summed one, or the spin blocks for this state's Ms, in
        MPS chain order. The driver is set up for this worker's state.
        """
        from pyblock2.driver.core import SymmetryTypes

        driver = self._make_driver()
        tags = [bra_solver._root_tags[left_root], self._root_tags[right_root]]
        bra, ket = [driver.load_mps(tag, nroots=1) for tag in tags]
        if spin_resolved:
            # Take the Ms component of each root into a non-spin-adapted MPS. block2
            # converts only an MPS with its center at the first site, so move the
            # center on a copy, leaving the MPS that a warm restart reloads unchanged.
            # The component has less than unit norm unless Ms is the only one, so
            # normalize it.
            converted = []
            for mps, tag in zip((bra, ket), tags):
                mps = driver.copy_mps(mps, f"SU2-{tag}")
                driver.align_mps_center(mps, ref=0)
                converted.append(
                    driver.mps_change_to_sz(mps, f"SZ-{tag}", sz=self.state.twice_ms)
                )
            bra, ket = converted
            driver.symm_type = SymmetryTypes.SZ
            driver.initialize_system(
                n_sites=self.norb,
                n_elec=self.nactel,
                spin=self.state.twice_ms,
                pg_irrep=self.state.symmetry,
                orb_sym=[self.orb_sym[i] for i in self._order],
            )
            identity = driver.get_identity_mpo()
            for mps in (bra, ket):
                mps.iscale(1 / np.sqrt(driver.expectation(mps, identity, mps)))
        pdms = driver.get_npdm(
            ket, pdm_type=order, bra=None if tags[0] == tags[1] else bra
        )
        return list(pdms) if spin_resolved else [pdms]

    def make_cumulant(
        self,
        root: int,
        *,
        order: Literal[2, 3],
        spin_type: Literal["sf"],
    ):
        """
        Make the spin-free cumulant of the given order for one DMRG root.

        Parameters
        ----------
        root : int
            The root index.
        order : int
            The cumulant order (2 or 3).
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
        Start the next run from a random MPS, with an orbital ordering the solver
        chooses again, rather than from the previous MPS. ``guess`` is ignored.
        """
        self._has_mps = False
        self._order = None

    def cleanup(self):
        """Forget the MPS, whose files the solver removes with its scratch directory."""
        self._root_tags = []
        self._has_mps = False
        self._order = None


@dataclass
class DMRGSolver(CIBase):
    """
    A spin-adapted density matrix renormalization group (DMRG) active-space solver,
    using block2.

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

    _rdm_orders: ClassVar[tuple[int, ...]] = (1, 2, 3)
    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("sd", "sf")
    _rdm_cross_state_orders: ClassVar[tuple[int, ...]] = (1,)
    _cumulant_orders: ClassVar[tuple[int, ...]] = (2, 3)
    _cumulant_spin_types: ClassVar[tuple[str, ...]] = ("sf",)

    def __post_init__(self):
        if not BLOCK2_AVAILABLE:
            raise ImportError(
                "DMRGSolver requires block2, which is not installed. See "
                "'Install block2 for DMRG' in the Forte2 quickstart."
            )
        super().__post_init__()
        self._scratch = None

    def _startup(self):
        if self._scratch is None:
            if self.dmrg_params.scratch is not None:
                os.makedirs(self.dmrg_params.scratch, exist_ok=True)
            self._scratch = tempfile.mkdtemp(
                prefix="forte2_dmrg_", dir=self.dmrg_params.scratch
            )
            # Remove it when the solver is garbage-collected or Python exits
            weakref.finalize(self, shutil.rmtree, self._scratch, ignore_errors=True)
        super()._startup()

    def _extra_worker_kwargs(self, index, state):
        return {"scratch": self._scratch, "tag_prefix": f"S{index}-"}

    def run(self):
        if self.first_run:
            self._startup()
            self.first_run = False
        # Before a cold start, choose one orbital ordering for every state's MPS
        if self.sub_solvers[0]._order is None:
            order = self._choose_orbital_ordering()
            for dmrg_solver in self.sub_solvers:
                dmrg_solver._order = order
        return super().run()

    def _choose_orbital_ordering(self):
        """Choose the active orbital at each site of the MPS chain."""
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes

        p = self.dmrg_params
        if p.orbital_ordering == "original" or self.norb == 0:
            logger.log("Using original orbital ordering", self.log_level)
            return np.arange(self.norb)
        ints = self.sub_solvers[0].ints
        driver = DMRGDriver(
            stack_mem=p.stack_memory << 20,
            scratch=self._scratch,
            symm_type=SymmetryTypes.SU2,
            n_threads=cpp_helpers.get_num_threads(),
        )
        order = np.asarray(
            driver.orbital_reordering(
                np.asarray(ints.H),
                # block2 takes chemist-notation integrals, (pq|rs) = <pr|qs>
                np.asarray(ints.V).transpose(0, 2, 1, 3),
                method=p.orbital_ordering,
            )
        )
        logger.log(
            f"Using orbital ordering {order} (active space relative indexing), "
            f"generated with the '{p.orbital_ordering}' algorithm",
            self.log_level,
        )
        return order

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: Literal[1, 2, 3],
        spin_type: Literal["sd", "sf"],
        packed: bool = True,
    ):
        """
        RDM of the given order and representation for two absolute DMRG roots. Requests
        between roots of different states are only supported at order=1. See
        ``_DMRGSingleStateSolver.make_rdm``.
        """
        left_state, right_state, left_root_in_state, right_root_in_state, spin_type = (
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
        left_solver = self.sub_solvers[left_state]
        right_solver = self.sub_solvers[right_state]
        if left_state == right_state:
            return right_solver.make_rdm(
                left_root_in_state,
                right_root_in_state,
                order=order,
                spin_type=spin_type,
                packed=packed,
            )
        if (
            spin_type == "sf"
            and left_solver.state.multiplicity != right_solver.state.multiplicity
        ):
            # A spin-free operator doesn't connect states of different total spin
            return np.zeros((self.norb, self.norb))
        return right_solver._rdm(
            left_solver,
            left_root_in_state,
            right_root_in_state,
            order,
            spin_type,
            packed,
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
        """Remove the scratch directory with the MPS of every state."""
        for dmrg_solver in self.sub_solvers:
            dmrg_solver.cleanup()
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
