import shutil
import tempfile
import weakref
from dataclasses import dataclass, field
from typing import ClassVar, Literal

import numpy as np

from forte2.state import State, MOSpace
from forte2.helpers import logger
from forte2.jkbuilder import RestrictedMOIntegrals, SpinorbitalIntegrals
from forte2.base_classes import CIBase, RelCIBase
from forte2.base_classes.params import DMRGParams
from forte2.ci.ci_utils import make_cumulant_from_rdms, validate_single_state_rdm
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
    die_if_not_converged : bool, optional, default=False
        If True, raise an error if the DMRG sweeps do not converge.
    log_level : int, optional
        The logging level for the solver. Defaults to ``logger.VERBOSITY_DEBUG``.

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
    die_if_not_converged: bool = False
    log_level: int = logger.VERBOSITY_DEBUG

    ### Non-init attributes
    executed: bool = field(default=False, init=False)

    ### These will be overridden by _RelDMRGSingleStateSolver
    two_component: ClassVar[bool] = False
    dtype: ClassVar[type] = float

    _rdm_orders: ClassVar[tuple[int, ...]] = (1, 2, 3)
    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("sf",)
    _cumulant_orders: ClassVar[tuple[int, ...]] = (2, 3)
    _cumulant_spin_types: ClassVar[tuple[str, ...]] = ("sf",)

    def __post_init__(self):
        self.norb = self.mo_space.nactv
        self.ncore = self.mo_space.ncore + self.mo_space.nfrozen_core

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

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: Literal[1, 2, 3],
        spin_type: Literal["sf", "so"],
    ):
        r"""
        Make the (transition) RDM of the given order for two DMRG roots.

        Parameters
        ----------
        left_root : int
            The bra root index.
        right_root : int | None, optional
            The ket root index. Defaults to ``left_root`` (diagonal RDM). When
            different, a transition RDM ``<left_root| ... |right_root>`` is
            returned.
        order : int
            The RDM order (1, 2, or 3).
        spin_type : str
            "sf" (spin-free) for the one-component solver, "so" (spin-orbital)
            for the two-component one. The longer spellings are also accepted.

        Returns
        -------
        NDArray
            The (transition) RDM in forte2's convention.

        Notes
        -----
        Transition RDMs carry an overall phase (sign) uncertainty inherent to
        block2's MPS and are only physically meaningful between non-degenerate
        roots (within a degenerate manifold the RDM depends on the arbitrary
        basis chosen inside the manifold).
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
        if self.norb == 0:
            return np.zeros((0,) * (2 * order), dtype=self.dtype)
        ket, bra = self._load_root_kets_for_rdm(left_root, right_root)
        if order == 1:
            return np.ascontiguousarray(self._get_1pdm(ket, bra=bra))
        if order == 2:
            return block2_2pdm_to_sf_2rdm(self._get_2pdm(ket, bra=bra))
        return block2_3pdm_to_sf_3rdm(self._get_3pdm(ket, bra=bra))

    def make_cumulant(
        self,
        root: int,
        *,
        order: Literal[2, 3],
        spin_type: Literal["sf", "so"],
    ):
        """
        Make the cumulant of the given order for one DMRG root.

        Parameters
        ----------
        root : int
            The root index.
        order : int
            The cumulant order (2 or 3).
        spin_type : str
            "sf" (spin-free) for the one-component solver, "so" (spin-orbital)
            for the two-component one.

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

    def _get_1pdm(self, ket, bra=None):
        """Extract the block2 1-particle density matrix (SU2, spin-summed)."""
        return self._driver.get_npdm(ket, pdm_type=1, bra=bra)

    def _get_2pdm(self, ket, bra=None):
        """Extract the block2 2-particle density matrix (SU2, chemist order)."""
        return self._driver.get_npdm(ket, pdm_type=2, bra=bra)

    def _get_3pdm(self, ket, bra=None):
        """Extract the block2 3-particle density matrix (SU2, chemist order)."""
        return self._driver.get_npdm(ket, pdm_type=3, bra=bra)

    def compute_natural_occupation_numbers(self):
        """
        Compute the natural occupation numbers from the 1-RDMs.

        Returns
        -------
        (norb, nroot) NDArray
            The natural occupation numbers for each root.
        """
        if not self.executed:
            raise RuntimeError("DMRG solver has not been executed yet.")
        spin_type = "so" if self.two_component else "sf"
        no = np.zeros((self.norb, self.nroot))
        for i in range(self.nroot):
            no[:, i] = np.linalg.eigvalsh(
                self.make_rdm(i, order=1, spin_type=spin_type)
            )[::-1]

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
        The logging level for the solver. Defaults to ``logger.VERBOSITY_DEBUG``.

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

    # Active-space integral class
    _integrals_cls: ClassVar[type] = RestrictedMOIntegrals
    # Single state solver class
    _ss_solver_cls: ClassVar[type] = _DMRGSingleStateSolver

    _rdm_orders: ClassVar[tuple[int, ...]] = (1, 2, 3)
    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("sf",)
    # each state is optimized by its own block2 driver, so RDMs between roots of
    # different states are not available
    _rdm_cross_state_orders: ClassVar[tuple[int, ...]] = ()
    _cumulant_orders: ClassVar[tuple[int, ...]] = (2, 3)
    _cumulant_spin_types: ClassVar[tuple[str, ...]] = ("sf",)

    def get_convergence_status(self):
        """
        Get the convergence status of each sub-solver.

        Returns
        -------
        list[bool]
            A list of booleans indicating whether each sub-solver has converged.
        """
        return [dmrg_solver.converged for dmrg_solver in self.sub_solvers]

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

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: Literal[1, 2, 3],
        spin_type: Literal["sf"],
    ):
        """Spin-free RDM of the given order for two absolute DMRG roots (same-state only)."""
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

    two_component: ClassVar[bool] = True
    dtype: ClassVar[type] = complex

    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("so",)
    _cumulant_spin_types: ClassVar[tuple[str, ...]] = ("so",)

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
        The logging level for the solver. Defaults to ``logger.VERBOSITY_DEBUG``.

    Attributes
    ----------
    sub_solvers : list[_RelDMRGSingleStateSolver]
        A per-state list of relativistic DMRG workers.
    evals_per_solver, evals_flat, E, E_avg
        As in ``DMRGSolver``.
    """

    orbital_rotation_invariant: ClassVar[bool] = False

    dmrg_params: DMRGParams = field(default_factory=DMRGParams)

    # Methods that are representation-agnostic are reused verbatim from the
    # non-relativistic DMRGSolver (mirrors how RelCISolver reuses CISolver).
    get_convergence_status = DMRGSolver.get_convergence_status
    cleanup = DMRGSolver.cleanup
    get_top_determinants = DMRGSolver.get_top_determinants

    # Active-space integral class
    _integrals_cls: ClassVar[type] = SpinorbitalIntegrals
    # Single state solver class
    _ss_solver_cls: ClassVar[type] = _RelDMRGSingleStateSolver

    _rdm_orders: ClassVar[tuple[int, ...]] = (1, 2, 3)
    _rdm_spin_types: ClassVar[tuple[str, ...]] = ("so",)
    # each state is optimized by its own block2 driver, so RDMs between roots of
    # different states are not available
    _rdm_cross_state_orders: ClassVar[tuple[int, ...]] = ()
    _cumulant_orders: ClassVar[tuple[int, ...]] = (2, 3)
    _cumulant_spin_types: ClassVar[tuple[str, ...]] = ("so",)

    def make_rdm(
        self,
        left_root: int,
        right_root: int | None = None,
        *,
        order: Literal[1, 2, 3],
        spin_type: Literal["so"],
    ):
        """Spin-orbital RDM of the given order for two absolute DMRG roots (same-state only)."""
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
