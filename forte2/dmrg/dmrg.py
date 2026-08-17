import shutil
import tempfile
import weakref
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from forte2.state import State, MOSpace
from forte2.helpers import logger
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.base_classes import CIBase, RelCIBase
from forte2.base_classes.params import DMRGParams
from forte2.orbitals import Semicanonicalizer, NaturalOrbitals
from forte2.ci.ci import CISolver
from forte2.ci.ci_utils import (
    pretty_print_ci_summary,
    pretty_print_ci_nat_occ_numbers,
    pretty_print_ci_transition_props,
)
from .dmrg_utils import (
    physicist_to_chemist_g2e,
    block2_2pdm_to_sf_2rdm,
    block2_3pdm_to_sf_3rdm,
)


@dataclass
class _DMRGSingleStateSolver:
    """
    A DMRG (block2) active-space solver for a single ``State``.

    This is the per-state worker used by ``DMRGSolver`` (analogous to
    ``_CISingleStateSolver`` for the CI solver). It owns a block2
    ``DMRGDriver``, builds the quantum-chemistry MPO from the active-space
    integrals, optimizes a (possibly multi-root) MPS, and returns spin-free
    1- and 2-RDMs in forte2's convention. Although possible, it is not
    recommended to instantiate this class directly; use ``DMRGSolver`` instead.

    Parameters
    ----------
    mo_space : MOSpace
        Specifies the GASes and core orbitals.
    state : State
        The electronic state for which the DMRG is solved.
    ints : RestrictedMOIntegrals
        The molecular orbital integrals for the active space.
    nroot : int
        The number of roots to compute (state-averaged MPS if > 1).
    active_orbsym : list[int]
        Per-GAS list of orbital symmetries for the active orbitals.
    dmrg_params : DMRGParams, optional
        Parameters controlling the block2 DMRG calculation.
    log_level : int, optional
        The logging level for the solver.
    die_if_not_converged : bool, optional, default=False
        If True, raise an error if the DMRG sweeps do not converge.

    Attributes
    ----------
    evals : NDArray
        The DMRG energies for each root.
    """

    mo_space: MOSpace
    state: State
    ints: RestrictedMOIntegrals
    nroot: int
    active_orbsym: list[int]
    dmrg_params: DMRGParams = field(default_factory=DMRGParams)
    log_level: int = field(default=logger.get_verbosity_level())
    die_if_not_converged: bool = False

    ### Non-init attributes
    executed: bool = field(default=False, init=False)

    def __post_init__(self):
        self.norb = self.mo_space.nactv
        self.ncore = self.mo_space.ncore + self.mo_space.nfrozen_core
        self.two_component = False
        self.dtype = float

        # block2 objects, built in run()
        self._driver = None
        self._mpo = None
        # per-root single-root MPS tags on disk, used to reload for RDMs
        self._root_tags = []
        self._converged = False
        # whether an optimized "GS" MPS exists on disk to warm-start from
        self._has_mps = False

        # scratch directory management: each worker gets its own unique
        # subdirectory so that state-averaged runs (multiple workers sharing the
        # same DMRGParams) never collide on MPS files/tags on disk.
        self._scratch_root = self.dmrg_params.scratch
        self._scratch = None

    # ------------------------------------------------------------------
    # block2 orbital symmetry
    # ------------------------------------------------------------------
    def _flat_orbsym(self):
        """Flatten the per-GAS active orbital symmetries into a single list."""
        flat = []
        for gas in self.active_orbsym:
            flat.extend(gas)
        # block2 SU2/point-group irreps; C1 -> all zeros
        return [int(s) for s in flat]

    # ------------------------------------------------------------------
    # driver lifecycle
    # ------------------------------------------------------------------
    def _symm_type(self):
        """The block2 SymmetryTypes for this worker (SU2 for the base solver)."""
        from pyblock2.driver.core import SymmetryTypes

        return getattr(SymmetryTypes, self.dmrg_params.symm_type.upper())

    def _target(self):
        """
        The (n_elec, spin) target passed to ``initialize_system``. For the
        spin-adapted (SU2) solver each doubly occupied core orbital holds two
        electrons and the target 2S = multiplicity - 1.
        """
        nactel = self.state.nel - 2 * self.ncore
        assert nactel >= 0, f"Number of active electrons {nactel} must be non-negative."
        spin = self.state.multiplicity - 1
        return nactel, spin

    def _make_driver(self):
        import os

        from pyblock2.driver.core import DMRGDriver

        # allocate a unique scratch subdirectory for this worker
        if self._scratch is None:
            if self._scratch_root is not None:
                os.makedirs(self._scratch_root, exist_ok=True)
                self._scratch = tempfile.mkdtemp(
                    prefix="state_", dir=self._scratch_root
                )
            else:
                self._scratch = tempfile.mkdtemp(prefix="forte2_dmrg_")
            # Safety net: cleanup() is the primary way to remove this
            # directory, but nothing calls it automatically (the same worker
            # may be re-run many times across MCSCF macroiterations, so this
            # can't just happen at the end of run()). Fall back to removing
            # it when this worker is garbage-collected, so an un-cleaned-up
            # DMRG solver doesn't leak scratch directories indefinitely. Binds
            # the path (not self) so the finalizer doesn't keep self alive.
            weakref.finalize(self, shutil.rmtree, self._scratch, ignore_errors=True)

        driver = DMRGDriver(
            scratch=self._scratch,
            symm_type=self._symm_type(),
            n_threads=self.dmrg_params.n_threads,
        )
        nactel, spin = self._target()
        driver.initialize_system(
            n_sites=self.norb,
            n_elec=nactel,
            spin=spin,
            orb_sym=self._flat_orbsym(),
        )
        return driver

    def _ecore(self):
        """The scalar (core) energy passed to block2 (real for SU2)."""
        return float(self.ints.E)

    def _build_mpo(self):
        g2e = physicist_to_chemist_g2e(self.ints.V)
        return self._driver.get_qc_mpo(
            h1e=np.ascontiguousarray(self.ints.H),
            g2e=g2e,
            ecore=self._ecore(),
            reorder="fiedler" if self.dmrg_params.reorder_orbitals else None,
            iprint=self.dmrg_params.iprint,
        )

    def _pad_schedule(self, schedule):
        """Pad a per-sweep schedule to n_sweeps by repeating the last entry."""
        n = self.dmrg_params.n_sweeps
        if len(schedule) >= n:
            return list(schedule)
        return list(schedule) + [schedule[-1]] * (n - len(schedule))

    def run(self):
        if self.norb == 0:
            # No active orbitals: energy is just the (core) scalar term.
            self.evals = np.array([float(self.ints.E)] * self.nroot)
            self._converged = True
            self.executed = True
            return self

        # A fresh driver bound to this worker's scratch. block2 keeps a single
        # process-global frame, so recreating the driver here also re-activates
        # this worker's scratch (important for state-averaged runs where
        # sibling workers each create their own driver).
        self._driver = self._make_driver()
        self._mpo = self._build_mpo()

        max_bond = max(self.dmrg_params.bond_dims)
        # Warm-start from the MPS optimized on the previous set of integrals if
        # one is available (e.g. across MCSCF macroiterations). Otherwise start
        # from a fresh random MPS. reset_eigensolver() clears the warm start.
        if self._can_warm_start():
            ket = self._driver.load_mps("GS", nroots=self.nroot)
        else:
            ket = self._driver.get_random_mps(
                tag="GS",
                bond_dim=max_bond,
                nroots=self.nroot,
            )

        energy = self._driver.dmrg(
            self._mpo,
            ket,
            n_sweeps=self.dmrg_params.n_sweeps,
            tol=self.dmrg_params.tol,
            bond_dims=self._pad_schedule(self.dmrg_params.bond_dims),
            noises=self._pad_schedule(self.dmrg_params.noises),
            thrds=self._pad_schedule(self.dmrg_params.thrds),
            iprint=self.dmrg_params.iprint,
        )
        self._converged = self._assess_convergence()
        self._has_mps = True

        # Persist a single-root MPS per root to this worker's scratch so RDMs
        # can be computed later even after a sibling worker has taken over the
        # global block2 frame (see _load_root_ket).
        if self.nroot == 1:
            self._root_tags = ["GS"]
            self.evals = np.array([float(energy)])
        else:
            self._root_tags = []
            for r in range(self.nroot):
                tag = f"GS-{r}"
                self._driver.split_mps(ket, r, tag=tag)
                self._root_tags.append(tag)
            self.evals = np.array([float(e) for e in energy])

        if not self._converged and self.die_if_not_converged:
            raise RuntimeError(
                f"DMRG sweeps for state {self.state} did not converge to "
                f"tol={self.dmrg_params.tol}."
            )

        for i, e in enumerate(self.evals):
            logger.log(f"Final DMRG Energy Root {i}: {e:20.12f} [Eh]", self.log_level)

        self.executed = True
        return self

    def _can_warm_start(self):
        """Whether a previously optimized "GS" MPS is available on disk to
        continue from (set by a prior run, cleared by reset_eigensolver)."""
        return self._has_mps

    def _assess_convergence(self):
        """
        Judge convergence from the block2 sweep-energy history: the change in
        the (state-averaged) energy over the last two sweeps must be below the
        requested tolerance. Falls back to True when the history is too short
        to compare (e.g. a single-determinant/1-sweep case).
        """
        try:
            energies = self._driver._dmrg.energies
        except AttributeError:
            # self._driver._dmrg is a private block2 attribute with no
            # documented stability guarantee; if it's ever renamed/removed,
            # fail loud rather than silently reporting convergence.
            logger.log_warning(
                "Could not read block2's internal sweep-energy history "
                "(self._driver._dmrg.energies is unavailable); assuming "
                "converged without verifying. This likely indicates a block2 "
                "version incompatibility."
            )
            return True
        if len(energies) < 2:
            return True
        last = np.array(energies[-1], dtype=float)
        prev = np.array(energies[-2], dtype=float)
        return bool(np.max(np.abs(last - prev)) < self.dmrg_params.tol)

    @property
    def converged(self):
        return self._converged

    # ------------------------------------------------------------------
    # per-root MPS access
    # ------------------------------------------------------------------
    def _load_root_ket(self, root):
        """
        Reactivate this worker's driver and reload the single-root MPS from its
        private scratch. Reactivation is required because a sibling worker may
        have replaced block2's process-global frame since this worker ran.
        """
        # Always recreate the driver: constructing a DMRGDriver re-activates
        # block2's process-global frame onto this worker's scratch. Merely
        # holding a stale driver object is not enough, because a sibling worker
        # may have moved the global frame elsewhere since this worker ran.
        self._driver = self._make_driver()
        return self._driver.load_mps(self._root_tags[root], nroots=1)

    def _load_root_kets(self, left_root, right_root):
        """
        Reactivate this worker's driver and reload the single-root MPS for both
        the bra (``left_root``) and ket (``right_root``) roots from its private
        scratch, for computing (transition) RDMs. A single driver/frame is
        (re)activated for both MPS.
        """
        self._driver = self._make_driver()
        bra = self._driver.load_mps(self._root_tags[left_root], nroots=1)
        ket = self._driver.load_mps(self._root_tags[right_root], nroots=1)
        return bra, ket

    # ------------------------------------------------------------------
    # RDMs (spin-free, forte2 convention)
    # ------------------------------------------------------------------
    def _load_root_kets_for_rdm(self, left_root, right_root):
        """
        Resolve the (bra, ket) MPS pair for an RDM request.

        Returns ``(ket, None)`` for a diagonal RDM (``right_root is None`` or
        equal to ``left_root``) and ``(ket, bra)`` for a cross-root transition
        RDM, where ``bra`` corresponds to ``left_root`` and ``ket`` to
        ``right_root``.
        """
        if right_root is None or right_root == left_root:
            return self._load_root_ket(left_root), None
        bra, ket = self._load_root_kets(left_root, right_root)
        return ket, bra

    def make_sf_1rdm(self, left_root, right_root=None):
        r"""
        Make the spin-free one-particle (transition) RDM for DMRG roots.

        Parameters
        ----------
        left_root : int
            The bra root index.
        right_root : int | None, optional
            The ket root index. Defaults to ``left_root`` (diagonal RDM). When
            different, a transition 1-RDM ``<left_root| E_pq |right_root>`` is
            returned.

        Returns
        -------
        NDArray
            Spin-free one-particle (transition) RDM.

        Notes
        -----
        Transition RDMs carry an overall phase (sign) uncertainty inherent to
        block2's MPS and are only physically meaningful between non-degenerate
        roots (within a degenerate manifold the RDM depends on the arbitrary
        basis chosen inside the manifold).
        """
        if self.norb == 0:
            return np.zeros((0, 0), dtype=self.dtype)
        ket, bra = self._load_root_kets_for_rdm(left_root, right_root)
        pdm1 = self._get_1pdm(ket, bra=bra)
        return np.ascontiguousarray(pdm1)

    def make_sf_2rdm(self, left_root, right_root=None):
        r"""
        Make the spin-free two-particle (transition) RDM for DMRG roots.

        Parameters
        ----------
        left_root : int
            The bra root index.
        right_root : int | None, optional
            The ket root index. Defaults to ``left_root`` (diagonal RDM). When
            different, a transition 2-RDM is returned.

        Returns
        -------
        NDArray
            Spin-free two-particle (transition) RDM in forte2's convention.

        Notes
        -----
        See :meth:`make_sf_1rdm` for the phase/degeneracy caveats that apply to
        transition RDMs.
        """
        if self.norb == 0:
            return np.zeros((0, 0, 0, 0), dtype=self.dtype)
        ket, bra = self._load_root_kets_for_rdm(left_root, right_root)
        pdm2 = self._get_2pdm(ket, bra=bra)
        return block2_2pdm_to_sf_2rdm(pdm2)

    def make_sf_3rdm(self, left_root, right_root=None):
        r"""
        Make the spin-free three-particle (transition) RDM for DMRG roots.

        Parameters
        ----------
        left_root : int
            The bra root index.
        right_root : int | None, optional
            The ket root index. Defaults to ``left_root`` (diagonal RDM). When
            different, a transition 3-RDM is returned.

        Returns
        -------
        NDArray
            Spin-free three-particle (transition) RDM in forte2's convention.

        Notes
        -----
        See :meth:`make_sf_1rdm` for the phase/degeneracy caveats that apply to
        transition RDMs.
        """
        if self.norb == 0:
            return np.zeros((0,) * 6, dtype=self.dtype)
        ket, bra = self._load_root_kets_for_rdm(left_root, right_root)
        pdm3 = self._get_3pdm(ket, bra=bra)
        return block2_3pdm_to_sf_3rdm(pdm3)

    def _get_1pdm(self, ket, bra=None):
        """Extract the block2 1-particle density matrix (SU2, spin-summed)."""
        return self._driver.get_npdm(ket, pdm_type=1, bra=bra)

    def _get_2pdm(self, ket, bra=None):
        """Extract the block2 2-particle density matrix (SU2, chemist order)."""
        return self._driver.get_npdm(ket, pdm_type=2, bra=bra)

    def _get_3pdm(self, ket, bra=None):
        """Extract the block2 3-particle density matrix (SU2, chemist order)."""
        return self._driver.get_npdm(ket, pdm_type=3, bra=bra)

    # DMRG supports spin-free 1-, 2-, and 3-RDMs; the state-averaged RDM
    # machinery in CIBase calls make_{1,2,3}rdm on the sub-solvers.
    make_1rdm = make_sf_1rdm
    make_2rdm = make_sf_2rdm
    make_3rdm = make_sf_3rdm

    def compute_natural_occupation_numbers(self):
        """Compute natural occupation numbers from the spin-free 1-RDMs."""
        if not self.executed:
            raise RuntimeError("DMRG solver has not been executed yet.")
        no = np.zeros((self.norb, self.nroot))
        for i in range(self.nroot):
            g1 = self.make_sf_1rdm(i)
            no[:, i] = np.linalg.eigvalsh(g1)[::-1]
        return no

    # ------------------------------------------------------------------
    # integral / solver management
    # ------------------------------------------------------------------
    def set_ints(self, scalar, oei, tei):
        """Set the active-space integrals for the DMRG solver."""
        self.ints.E = scalar
        self.ints.H = oei
        self.ints.V = tei

    def reset_eigensolver(self):
        """
        Discard the warm-start MPS so the next run starts from a fresh random
        MPS (e.g. after a discontinuous change of orbitals/integrals).
        """
        self._mpo = None
        self._has_mps = False
        self._root_tags = []

    def cleanup(self):
        """Remove the per-worker scratch directory."""
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None
        self._has_mps = False
        self._root_tags = []


@dataclass
class DMRGSolver(CIBase):
    """
    A DMRG (block2) active-space solver, drop-in compatible with ``CISolver``.

    This solver can be called iteratively, e.g. in an MCSCF loop, and plugs into
    ``MCOptimizer`` exactly like ``CISolver``. It supports state averaging over
    multiple ``State`` objects and multiple roots per state, using one
    ``_DMRGSingleStateSolver`` worker per state.

    Parameters
    ----------
    dmrg_params : DMRGParams, optional
        Parameters for the DMRG calculation. If not provided, default
        parameters are used.
    log_level : int, optional
        The logging level for the solver. Defaults to warning level so the
        solver is quiet when used inside a loop.

    Attributes
    ----------
    sub_solvers : list[_DMRGSingleStateSolver]
        A per-state list of DMRG workers.
    evals_per_solver : list[NDArray]
        The eigenvalues computed by each sub-solver.
    evals_flat, E : NDArray
        The flattened array of eigenvalues from all sub-solvers.
    E_avg : float
        The state-averaged energy.
    """

    orbital_rotation_invariant: ClassVar[bool] = False

    dmrg_params: DMRGParams = field(default_factory=DMRGParams)
    # If used as a solver, log at warning level
    log_level: int = field(default=logger.get_verbosity_level() + 1)

    def _startup(self):
        super()._startup()
        self.norb = self.mo_space.nactv
        # no distinction between core and frozen core in the DMRG solver
        self.core_indices = (
            self.mo_space.frozen_core_indices + self.mo_space.core_indices
        )
        self.active_indices = self.mo_space.active_indices

        ints = RestrictedMOIntegrals(
            self.system,
            self.mos.C[0],
            self.active_indices,
            self.core_indices,
        )

        self.sub_solvers = []
        active_orbsym = [
            [self.mos.irrep_indices[0][i] for i in active_space]
            for active_space in self.mo_space.active_orbitals
        ]
        for i, state in enumerate(self.sa_info.states):
            kwargs = self._collect_child_kwargs(_DMRGSingleStateSolver)
            # these are needed by _DMRGSingleStateSolver but not present as
            # attributes of DMRGSolver
            kwargs.update(
                {
                    "ints": ints,
                    "state": state,
                    "nroot": self.sa_info.nroots[i],
                    "active_orbsym": active_orbsym,
                }
            )
            self.sub_solvers.append(_DMRGSingleStateSolver(**kwargs))

    def run(self):
        if self.first_run:
            self._startup()
            self.first_run = False

        self.evals_per_solver = []
        for dmrg_solver in self.sub_solvers:
            dmrg_solver.run()
            self.evals_per_solver.append(dmrg_solver.evals)

        self.evals_flat = np.concatenate(self.evals_per_solver)
        self.E_avg = self.compute_average_energy()
        self.E = self.evals_flat

        self.executed = True
        return self

    def reset_eigensolver(self):
        """
        Reset each sub-solver so the MPO/MPS are rebuilt on the next run.
        Useful whenever the integrals have changed (e.g. after semicanonicalization).
        """
        for dmrg_solver in self.sub_solvers:
            dmrg_solver.reset_eigensolver()

    def set_ints(self, scalar, oei, tei):
        """
        Set the active-space integrals for the DMRG solver.

        Parameters
        ----------
        scalar : float
            The scalar energy term.
        oei : NDArray
            One-electron active-space integrals in the MO basis.
        tei : NDArray
            Two-electron active-space integrals in the MO basis.
        """
        for dmrg_solver in self.sub_solvers:
            dmrg_solver.set_ints(scalar, oei, tei)

    def set_maxiter(self, maxiter):
        """
        Set the maximum number of DMRG sweeps for each sub-solver.

        Parameters
        ----------
        maxiter : int
            The maximum number of sweeps to set.
        """
        self.maxiter = maxiter
        for dmrg_solver in self.sub_solvers:
            dmrg_solver.dmrg_params.n_sweeps = maxiter

    def get_convergence_status(self):
        """
        Get the convergence status of each sub-solver.

        Returns
        -------
        list[bool]
            A list of booleans indicating whether each sub-solver has converged.
        """
        return [dmrg_solver.converged for dmrg_solver in self.sub_solvers]

    def compute_natural_occupation_numbers(self):
        """
        Compute the natural occupation numbers for the DMRG states.
        The first columns correspond to each root; if state-averaging over more
        than one state, the last column is from the average 1-RDM.

        Returns
        -------
        (norb, nroot) NDArray
            The natural occupation numbers for each root.
        """
        nos = []
        for dmrg_solver in self.sub_solvers:
            nos.append(dmrg_solver.compute_natural_occupation_numbers())
        if self.ncis > 1:
            g1_avg = self.make_average_1rdm()
            nos.append(np.linalg.eigvalsh(g1_avg)[::-1][:, np.newaxis])
        self.nat_occs = np.concatenate(nos, axis=1)
        return self.nat_occs

    def cleanup(self):
        """Remove scratch directories created by the sub-solvers."""
        for dmrg_solver in self.sub_solvers:
            dmrg_solver.cleanup()

    def get_top_determinants(self, n=5):
        """
        Return the top determinants for each root.

        DMRG represents the wavefunction as an MPS and has no explicit
        determinant expansion, so this returns an empty list per root. It is
        provided for API compatibility with ``CISolver`` (e.g. so that
        ``MCOptimizer`` post-processing works unchanged).
        """
        return [[] for _ in range(self.sa_info.nroots_sum)]

    def make_sf_1rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> NDArray:
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state != right_state:
            raise NotImplementedError(
                "Cross-state RDMs are not supported for DMRG."
            )
        return self.sub_solvers[left_state].make_sf_1rdm(
            left_root_in_state, right_root_in_state
        )

    def make_sf_2rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> NDArray:
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state != right_state:
            raise NotImplementedError(
                "Cross-state RDMs are not supported for DMRG."
            )
        return self.sub_solvers[left_state].make_sf_2rdm(
            left_root_in_state, right_root_in_state
        )

    def make_sf_3rdm(
        self,
        left_root: int,
        right_root: int | None = None,
    ) -> NDArray:
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state != right_state:
            raise NotImplementedError(
                "Cross-state RDMs are not supported for DMRG."
            )
        return self.sub_solvers[left_state].make_sf_3rdm(
            left_root_in_state, right_root_in_state
        )

    make_1rdm = make_sf_1rdm
    make_2rdm = make_sf_2rdm
    make_3rdm = make_sf_3rdm

    # RDM-consuming orchestration (state/root bookkeeping, dipole integrals) is
    # representation-agnostic; only make_1rdm above differs between DMRG and CI.
    compute_transition_properties = CISolver.compute_transition_properties


@dataclass
class DMRG(DMRGSolver):
    """
    DMRG solver specialized for a single DMRG calculation (i.e., not used in a
    loop). See ``DMRGSolver`` for all parameters and attributes.
    """

    die_if_not_converged: bool = True
    final_orbitals: str = "original"
    do_transition_dipole: bool = False
    log_level: int = field(default=logger.get_verbosity_level())

    def __post_init__(self):
        super().__post_init__()
        if self.final_orbitals not in ["original", "semicanonical", "natural"]:
            raise ValueError(
                f"Invalid value for final_orbitals: {self.final_orbitals}. "
                "Must be 'original', 'semicanonical', or 'natural'."
            )

    def run(self):
        super().run()
        self._post_process()
        if self.final_orbitals in ("semicanonical", "natural"):
            irrep_indices = np.array(self.mos.irrep_indices[0])[
                self.mo_space.orig_to_contig
            ]
            C_contig = self.mos.C[0][:, self.mo_space.orig_to_contig].copy()
            g1_act = self.make_average_1rdm()

            # Semicanonicalize the orbital subspaces (except the active space,
            # for natural orbitals).
            semi = Semicanonicalizer(
                mo_space=self.mo_space,
                system=self.system,
                irrep_indices=irrep_indices,
                do_active=(self.final_orbitals == "semicanonical"),
            )
            semi.semi_canonicalize(g1=g1_act, C_contig=C_contig)
            C_final = semi.C_semican

            if self.final_orbitals == "natural":
                natural_orbital = NaturalOrbitals(
                    self.mo_space, irrep_indices=irrep_indices
                )
                natural_orbital.make_natural_orbitals(
                    g1_act=g1_act, C_contig=C_final
                )
                C_final = natural_orbital.C_natural

            self.mos.C[0] = C_final[:, self.mo_space.contig_to_orig].copy()

            # recompute the wavefunction in the final orbital basis
            ints = RestrictedMOIntegrals(
                self.system,
                self.mos.C[0],
                self.active_indices,
                self.core_indices,
            )
            self.set_ints(ints.E, ints.H, ints.V)
            self.reset_eigensolver()
            super().run()

        return self

    def _post_process(self):
        pretty_print_ci_summary(self.sa_info, self.evals_per_solver)
        self.compute_natural_occupation_numbers()
        pretty_print_ci_nat_occ_numbers(self.sa_info, self.mo_space, self.nat_occs)

        if self.do_transition_dipole:
            self.compute_transition_properties()
            pretty_print_ci_transition_props(
                self.sa_info,
                self.transition_dipoles,
                self.oscillator_strengths,
                self.evals_per_solver,
            )


@dataclass
class _RelDMRGSingleStateSolver(_DMRGSingleStateSolver):
    """
    A relativistic (two-component, complex) DMRG worker for a single ``State``.

    Drives block2 in general-spin + complex mode
    (``SymmetryTypes.SGF | SymmetryTypes.CPX``). Everything about the lifecycle
    (private scratch, driver reactivation, warm start, convergence detection) is
    inherited from ``_DMRGSingleStateSolver``; only the block2 symmetry, the
    active-electron count, complex ``ecore``, and the PDM extraction differ.
    """

    def __post_init__(self):
        super().__post_init__()
        self.two_component = True
        self.dtype = complex

    def _symm_type(self):
        from pyblock2.driver.core import SymmetryTypes

        return SymmetryTypes.SGF | SymmetryTypes.CPX

    def _target(self):
        # In the spin-orbital (general-spin) representation each spinor holds a
        # single electron, so the active electron count is nel - ncore (NOT
        # nel - 2*ncore). block2's general-spin mode does no spin adaptation, so
        # the spin argument is unused.
        nactel = self.state.nel - self.ncore
        assert nactel >= 0, f"Number of active electrons {nactel} must be non-negative."
        return nactel, 0

    def _ecore(self):
        # The core energy is real up to numerical noise, but block2's complex
        # MPO builder wants a complex scalar.
        return complex(self.ints.E)

    def _get_1pdm(self, ket, bra=None):
        # site_type=2 keeps the 2-dot MPS form; the default (site_type=0) splits
        # to 1-dot and triggers a zero-dimension zgemm crash in the complex
        # general-spin PDM path of the block2 wheels.
        return self._driver.get_npdm(ket, pdm_type=1, bra=bra, site_type=2)

    def _get_2pdm(self, ket, bra=None):
        return self._driver.get_npdm(ket, pdm_type=2, bra=bra, site_type=2)

    def _get_3pdm(self, ket, bra=None):
        return self._driver.get_npdm(ket, pdm_type=3, bra=bra, site_type=2)

    def compute_natural_occupation_numbers(self):
        """Natural occupation numbers from the (complex Hermitian) 1-RDMs."""
        if not self.executed:
            raise RuntimeError("DMRG solver has not been executed yet.")
        no = np.zeros((self.norb, self.nroot))
        for i in range(self.nroot):
            g1 = self.make_sf_1rdm(i)
            no[:, i] = np.linalg.eigvalsh(g1)[::-1]
        return no

    # For two-component CI/DMRG the "spin-free" RDM accessors are the
    # spin-orbital RDMs; expose make_1rdm/make_2rdm as the primary API to match
    # RelCISolver.
    make_1rdm = _DMRGSingleStateSolver.make_sf_1rdm
    make_2rdm = _DMRGSingleStateSolver.make_sf_2rdm
    make_3rdm = _DMRGSingleStateSolver.make_sf_3rdm


@dataclass
class RelDMRGSolver(RelCIBase):
    """
    A relativistic (two-component) DMRG active-space solver, drop-in compatible
    with ``RelCISolver``. Drives block2 in general-spin + complex mode.

    Requires a two-component system and a block2 build with complex +
    general-spin support (``SymmetryTypes.SGF | SymmetryTypes.CPX``).

    Parameters
    ----------
    dmrg_params : DMRGParams, optional
        Parameters for the DMRG calculation. The ``symm_type`` field is ignored;
        the relativistic solver always uses SGF|CPX.
    log_level : int, optional
        The logging level for the solver.

    Attributes
    ----------
    sub_solvers : list[_RelDMRGSingleStateSolver]
        A per-state list of relativistic DMRG workers.
    evals_per_solver, evals_flat, E, E_avg
        As in ``DMRGSolver``.
    """

    orbital_rotation_invariant: ClassVar[bool] = False

    dmrg_params: DMRGParams = field(default_factory=DMRGParams)
    log_level: int = field(default=logger.get_verbosity_level() + 1)

    # Methods that are representation-agnostic are reused verbatim from the
    # non-relativistic DMRGSolver (mirrors how RelCISolver reuses CISolver).
    reset_eigensolver = DMRGSolver.reset_eigensolver
    set_ints = DMRGSolver.set_ints
    set_maxiter = DMRGSolver.set_maxiter
    get_convergence_status = DMRGSolver.get_convergence_status
    compute_natural_occupation_numbers = DMRGSolver.compute_natural_occupation_numbers
    cleanup = DMRGSolver.cleanup
    get_top_determinants = DMRGSolver.get_top_determinants

    def _startup(self):
        super()._startup()
        self.norb = self.mo_space.nactv
        self.core_indices = (
            self.mo_space.frozen_core_indices + self.mo_space.core_indices
        )
        self.active_indices = self.mo_space.active_indices

        ints = SpinorbitalIntegrals(
            self.system,
            self.mos.C[0],
            self.active_indices,
            self.core_indices,
        )

        self.sub_solvers = []
        active_orbsym = [
            [self.mos.irrep_indices[0][i] for i in active_space]
            for active_space in self.mo_space.active_orbitals
        ]
        for i, state in enumerate(self.sa_info.states):
            kwargs = self._collect_child_kwargs(_RelDMRGSingleStateSolver)
            kwargs.update(
                {
                    "ints": ints,
                    "state": state,
                    "nroot": self.sa_info.nroots[i],
                    "active_orbsym": active_orbsym,
                }
            )
            self.sub_solvers.append(_RelDMRGSingleStateSolver(**kwargs))

    def run(self, use_asym_ints=False):
        if use_asym_ints:
            raise NotImplementedError(
                "Antisymmetrized integrals are not supported for RelDMRG."
            )
        if self.first_run:
            self._startup()
            self.first_run = False

        self.evals_per_solver = []
        for dmrg_solver in self.sub_solvers:
            dmrg_solver.run()
            self.evals_per_solver.append(dmrg_solver.evals)

        self.evals_flat = np.concatenate(self.evals_per_solver)
        self.E_avg = self.compute_average_energy()
        self.E = self.evals_flat

        self.executed = True
        return self

    def make_1rdm(self, left_root: int, right_root: int | None = None) -> NDArray:
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state != right_state:
            raise NotImplementedError(
                "Cross-state RDMs are not supported for RelDMRG."
            )
        return self.sub_solvers[left_state].make_1rdm(
            left_root_in_state, right_root_in_state
        )

    def make_2rdm(self, left_root: int, right_root: int | None = None) -> NDArray:
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state != right_state:
            raise NotImplementedError(
                "Cross-state RDMs are not supported for RelDMRG."
            )
        return self.sub_solvers[left_state].make_2rdm(
            left_root_in_state, right_root_in_state
        )

    def make_3rdm(self, left_root: int, right_root: int | None = None) -> NDArray:
        left_state, right_state, left_root_in_state, right_root_in_state = (
            self._validate_rdm_inputs(left_root, right_root)
        )
        if left_state != right_state:
            raise NotImplementedError(
                "Cross-state RDMs are not supported for RelDMRG."
            )
        return self.sub_solvers[left_state].make_3rdm(
            left_root_in_state, right_root_in_state
        )

    # Same rationale as DMRGSolver: only the RDM primitives above are
    # representation-specific, so this is reused verbatim from CISolver (mirrors
    # RelCISolver.compute_transition_properties = CISolver.compute_transition_properties).
    compute_transition_properties = CISolver.compute_transition_properties


@dataclass
class RelDMRG(RelDMRGSolver):
    """
    Relativistic DMRG solver specialized for a single calculation (not in a
    loop). See ``RelDMRGSolver`` for all parameters and attributes.
    """

    die_if_not_converged: bool = True
    final_orbitals: str = "original"
    do_transition_dipole: bool = False
    log_level: int = field(default=logger.get_verbosity_level())

    def __post_init__(self):
        super().__post_init__()
        if self.final_orbitals not in ["original", "semicanonical", "natural"]:
            raise ValueError(
                f"Invalid value for final_orbitals: {self.final_orbitals}. "
                "Must be 'original', 'semicanonical', or 'natural'."
            )

    def run(self, use_asym_ints=False):
        super().run(use_asym_ints=use_asym_ints)
        self._post_process()
        if self.final_orbitals in ("semicanonical", "natural"):
            irrep_indices = np.array(self.mos.irrep_indices[0])[
                self.mo_space.orig_to_contig
            ]
            C_contig = self.mos.C[0][:, self.mo_space.orig_to_contig].copy()
            g1_act = self.make_average_1rdm()

            semi = Semicanonicalizer(
                mo_space=self.mo_space,
                system=self.system,
                irrep_indices=irrep_indices,
                do_active=(self.final_orbitals == "semicanonical"),
            )
            semi.semi_canonicalize(g1=g1_act, C_contig=C_contig)
            C_final = semi.C_semican

            if self.final_orbitals == "natural":
                natural_orbital = NaturalOrbitals(
                    self.mo_space, irrep_indices=irrep_indices
                )
                natural_orbital.make_natural_orbitals(
                    g1_act=g1_act, C_contig=C_final
                )
                C_final = natural_orbital.C_natural

            self.mos.C[0] = C_final[:, self.mo_space.contig_to_orig].copy()

            ints = SpinorbitalIntegrals(
                self.system,
                self.mos.C[0],
                self.active_indices,
                self.core_indices,
            )
            self.set_ints(ints.E, ints.H, ints.V)
            self.reset_eigensolver()
            super().run(use_asym_ints=use_asym_ints)

        return self

    _post_process = DMRG._post_process
