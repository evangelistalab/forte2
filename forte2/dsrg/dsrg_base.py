from dataclasses import dataclass, field
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray

from forte2.base_classes import ActiveSpaceDriver, Method
from forte2.helpers import logger
from forte2.orbitals import Semicanonicalizer
from forte2.ci.ci_utils import pretty_print_ci_summary
from .dsrg_common import _DSRGHelper
from .rel_dsrg_common import _RelDSRGHelper

# Largest off-diagonal Fock element tolerated within an orbital block.
_SEMICANONICAL_TOL = 1e-8


@dataclass
class DSRGBase(Method):
    """Base class for DSRG methods."""

    flow_param: float = 0.5

    # Reference relaxation options
    relax_reference: int | str | bool = False
    relax_maxiter: int = 10
    relax_tol: float = 1e-6

    # options to freeze orbitals
    frozen_core_orbitals: int | list[int] = None
    frozen_virtual_orbitals: int | list[int] = None

    # Build the active-space effective Hamiltonian even when the reference is not relaxed
    save_hbar: bool = False

    # Skip the three-body cumulant everywhere: the 3-RDM is never formed and lambda3
    # is never transformed into the semicanonical basis. For DSRG-MRPT3 lambda3 only
    # enters the energy, so the relaxed eigenvalues -- and hence any splitting -- are
    # unchanged; E_dsrg loses one term. For MRPT2 it also drops a term from the
    # virtual-virtual 1-RDM, so FNO natural orbitals (and anything downstream of them)
    # do change. See _warn_skip_3_cumulant.
    skip_3_cumulant: bool = False

    # Non-init attributes
    _warned_skip_3c: bool = field(init=False, default=False)
    converged: bool = field(init=False, default=False)
    _hbar0: float | None = field(init=False, default=None)
    _hbar1_canon: NDArray | None = field(init=False, default=None)
    _hbar2_canon: NDArray | None = field(init=False, default=None)
    # The DSRG energy before any incoming hbar_shift is applied
    _E_dsrg_bare: float | None = field(init=False, default=None)
    # Used for FNO corrections
    hbar_shift: dict | None = field(init=False, default=None)

    def _warn_skip_3_cumulant(self):
        """Say once, at run time, what dropping the three-body cumulant costs."""
        if not self.skip_3_cumulant or self._warned_skip_3c:
            return
        self._warned_skip_3c = True
        logger.log_warning(
            "  skip_3_cumulant is set: the three-body cumulant is not built. "
            "E_dsrg omits its lambda3 term, and any virtual natural orbitals built "
            "from the unrelaxed 1-RDM omit theirs. Relaxed eigenvalues of "
            "DSRG-MRPT3 are unaffected."
        )

    def __call__(self, parent_method):
        self._register_parent_method(parent_method)
        assert isinstance(self.parent_method, (ActiveSpaceDriver, DSRGBase)), (
            "Parent method must be a driver that owns an active-space solver "
            "(CI or MCOptimizer), or another DSRG method, got "
            f"{type(self.parent_method).__name__}."
        )
        return self

    @property
    def ci_solver(self):
        """
        The ci_solver of the parent method. A property (rather than a plain
        attribute cached in _startup) so it is available immediately at
        construction time, matching CIBase.ci_solver -- this lets a DSRGBase
        object itself serve as a valid parent for another DSRG method before
        it has necessarily been run.
        """
        return self.parent_method.ci_solver

    def __post_init__(self):
        self.requires = {"system", "mos", "mo_space"}
        self.provides = {"system", "mos", "mo_space"}
        self.requires_attrs.update({"ci_solver": None})

        # parse reference relaxation options
        if isinstance(self.relax_reference, bool):
            self.nrelax = self.relax_maxiter if self.relax_reference else 0
        elif isinstance(self.relax_reference, int):
            assert self.relax_reference >= 0, "relax_reference must be non-negative."
            self.nrelax = min(self.relax_reference, self.relax_maxiter)
        elif isinstance(self.relax_reference, str):
            assert self.relax_reference.lower() in [
                "once",
                "twice",
                "iterate",
            ], "relax_reference must be one of 'once', 'twice', or 'iterate'."
            if self.relax_reference.lower() == "once":
                self.nrelax = 1
            elif self.relax_reference.lower() == "twice":
                self.nrelax = 2
            else:
                self.nrelax = self.relax_maxiter
        else:
            logger.log_warning(
                "Reference relaxation options not recognized, no relaxation will be performed."
            )
            self.nrelax = 0

    def _startup(self):
        # [Edsrg(fixed_reference), Edsrg(relaxed_reference), Eref]
        self.relax_energies = np.zeros((self.nrelax + 1, 3))
        self.relax_eigvals_history = []

        if not self.parent_method.executed:
            self.parent_method.run()

        self.system = self.parent_method.system
        self.mos = self.parent_method.mos.copy()
        self.mo_space = self.parent_method.mo_space

        # update the MOSpace object if frozen orbitals are specified
        if (
            self.frozen_core_orbitals is not None
            or self.frozen_virtual_orbitals is not None
        ):
            self.mo_space = self.mo_space.update_frozen_orbitals(
                frozen_core_orbitals=self.frozen_core_orbitals,
                frozen_virtual_orbitals=self.frozen_virtual_orbitals,
            )

        self.ncorr = self.mo_space.corr.stop - self.mo_space.corr.start
        self.ncore = self.mo_space.core_corr.stop - self.mo_space.core_corr.start
        self.nact = self.mo_space.actv_corr.stop - self.mo_space.actv_corr.start
        self.nvirt = self.mo_space.virt_corr.stop - self.mo_space.virt_corr.start
        self.nhole = self.ncore + self.nact
        self.npart = self.nact + self.nvirt
        self.frozen_core = self.mo_space.frozen_core
        self.corr = self.mo_space.corr
        self.actv = self.mo_space.actv_corr
        self.core = self.mo_space.core_corr
        self.virt = self.mo_space.virt_corr
        self.hole = slice(0, self.nhole)
        self.part = slice(self.ncore, self.ncorr)
        self.ha = self.actv
        self.pa = slice(0, self.nact)
        self.hc = self.core
        self.pv = slice(self.nact, self.nact + self.nvirt)

        self.dsrg_helper = (
            _RelDSRGHelper(self) if self.two_component else _DSRGHelper(self)
        )
        perm = self.mo_space.orig_to_contig
        self._C = self.mos.C[0][:, perm].copy()

        if self.nrelax > 0:
            # The parent has had its say by now; every further run of this solver
            # is a relaxation cycle, so keep those quiet.
            self.ci_solver.log_level = logger.VERBOSITY_DEBUG

        self.E_core_orig = self.ci_solver.sub_solvers[0].ints.E
        self.H_orig = self.ci_solver.sub_solvers[0].ints.H.copy()
        self.V_orig = self.ci_solver.sub_solvers[0].ints.V.copy()

        self.semicanonicalizer = Semicanonicalizer(
            system=self.system,
            mo_space=self.mo_space,
            irrep_indices=np.array(self.mos.irrep_indices[0])[
                self.mo_space.orig_to_contig
            ],
            mix_active=False,
            # do not mix correlated core and frozen core orbitals after MCSCF
            mix_inactive=False,
        )

        self.fock_builder = self.system.fock_builder
        self.ints, self.cumulants = self.get_integrals()
        self._log_semicanonical_check()

    @property
    def _fock_actv_0th(self) -> NDArray:
        """Active-active block of the DSRG zeroth-order Hamiltonian.

        Diagonal for a single active space. With several GASes the coupling
        between them survives semicanonicalization and is kept here, i.e. treated
        as zeroth order, matching forte's spin-adapted MR-DSRG. Sole seam for that
        choice; see tests/dsrg/test_gas_dsrg_mrpt2.py.
        """
        return self.fock[self.actv, self.actv]

    def _build_fock_0th(self) -> NDArray:
        """The block-diagonal generalized Fock matrix, i.e. the DSRG H^(0)."""
        fock_0th = np.zeros_like(self.fock)
        fock_0th[self.core, self.core] = self.fock[self.core, self.core]
        fock_0th[self.actv, self.actv] = self._fock_actv_0th
        fock_0th[self.virt, self.virt] = self.fock[self.virt, self.virt]
        return fock_0th

    def _log_semicanonical_check(self):
        """Log per-block off-diagonal Fock norms and the coupling between GASes.

        The denominators use only the Fock diagonal, so an off-diagonal element
        inside a block is an error; coupling between GASes is expected.
        """
        gas = self.mo_space.gas_corr
        blocks = [("CORE", self.core)]
        if len(gas) == 1:
            blocks.append(("ACTIVE", gas[0]))
        else:
            blocks.extend((f"GAS{i + 1}", sl) for i, sl in enumerate(gas))
        blocks.append(("VIRTUAL", self.virt))

        width = 46
        logger.log_info1("\n" + "=" * width)
        logger.log_info1("DSRG Semicanonical Orbital Check".center(width))
        logger.log_info1("=" * width)
        logger.log_info1(f"{'Block':<10}{'#':>5}{'Max':>15}{'Mean':>16}")
        logger.log_info1("-" * width)
        worst = 0.0
        for name, sl in blocks:
            block = self.fock[sl, sl]
            nblock = block.shape[0]
            offdiag = np.abs(block - np.diag(np.diag(block)))
            fmax = offdiag.max() if nblock > 1 else 0.0
            fmean = offdiag.sum() / (nblock * (nblock - 1)) if nblock > 1 else 0.0
            worst = max(worst, fmax)
            logger.log_info1(f"{name:<10}{nblock:>5d}{fmax:>15.10f}{fmean:>16.10f}")
        logger.log_info1("=" * width)

        if worst > _SEMICANONICAL_TOL:
            logger.log_warning(
                f"  Fock matrix not diagonal within an orbital block (max "
                f"{worst:.3e}); DSRG denominators use only its diagonal."
            )

        if len(gas) < 2:
            return

        # gas_corr slices index the correlated space, as self.fock does.
        coupling = max(
            np.abs(self.fock[a, b]).max()
            for i, a in enumerate(gas)
            for b in gas[i + 1 :]
        )
        logger.log_info1(
            f"  Largest coupling between GASes: {coupling:.10f}\n"
            "  GAS-to-GAS excitations are excluded from the DSRG amplitudes."
        )

        if self.two_component:
            logger.log_warning(
                "  Two-component DSRG omits the GAS-GAS Fock coupling from T1 and "
                "F-tilde, which the spin-adapted solvers keep: expect ~1e-4 Eh "
                "disagreement between them."
            )

    def _release_integrals(self):
        """
        Release large per-run integral/cumulant state once it is no longer
        needed, freeing the underlying memory.
        """
        self.ints = None
        self.cumulants = None

    def _build_hbar(self):
        """
        Fold the raw hbar1/hbar2 left by solve_dsrg(form_hbar=True) into the
        CI-ready effective Hamiltonian, storing it as _hbar0/_hbar1_canon/
        _hbar2_canon (and applying any incoming hbar_shift).

        Implementations must not consume hbar1/hbar2 in place, so that calling
        this more than once per solve is harmless.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _build_hbar, so it "
            "cannot expose its effective Hamiltonian outside of reference "
            "relaxation (save_hbar is not supported)."
        )

    def _solve_dsrg_shifted(self, form_hbar):
        """
        Solve, keeping the unshifted energy in _E_dsrg_bare and returning it
        with any incoming hbar_shift's energy contribution applied.
        """
        e_dsrg = self.solve_dsrg(form_hbar)
        # Two-component solvers return a complex energy whose imaginary part should
        # be numerical noise. Check it before discarding it: casting first both
        # warns and makes the diagnostic unreachable, since a float's .imag is 0.
        if abs(np.imag(e_dsrg)) > 1e-12:
            logger.log_warning(
                f"DSRG energy has a significant imaginary component: {np.imag(e_dsrg)}"
            )
        self._E_dsrg_bare = float(np.real(e_dsrg))
        if self.save_hbar:
            self._build_hbar()
        shift = getattr(self.parent_method, "hbar_shift", None)
        if shift is None:
            return self._E_dsrg_bare
        return self._E_dsrg_bare + shift["e_dsrg"]

    def _post_process(self):
        """
        Hook run at the end of run(), after the energy (and, if requested, the
        effective Hamiltonian) is available. Subclasses override it to expose
        derived quantities -- e.g. RelDSRG_MRPT2 uses it to truncate its
        virtual space to frozen natural orbitals, and to publish an hbar_shift.
        """

    def run(self):
        self._startup()
        form_hbar = self.nrelax > 0 or self.save_hbar

        self.E_dsrg = self._solve_dsrg_shifted(form_hbar)

        self.relax_energies[0, 0] = self.E_dsrg.real
        # self.ints["E"] is <Psi_current| bare H |Psi_current>
        self.relax_energies[0, 2] = self.ints["E"].real
        self.E = self.E_dsrg

        width = 88
        for irelax in range(self.nrelax):
            if irelax == 0:
                logger.log_info1("\n DSRG reference relaxation glossary")
                logger.log_info1(" -E0      : <Psi_n     | bare H | Psi_n    >")
                logger.log_info1(" -Edsrg   : <Psi_n     | Hbar   | Psi_n    >")
                logger.log_info1(" -Erelaxed: <Psi_{n+1} | Hbar   | Psi_{n+1}>")
                logger.log_info1("=" * width)
                logger.log_info1("DSRG Reference Relaxation Summary".center(width))
                logger.log_info1("=" * width)
                logger.log_info1(
                    f"{'Iteration':>10} {'E0 (a.u.)':>25} {'Edsrg (a.u.)':>25} {'Erelaxed (a.u.)':>25}"
                )
                logger.log_info1("-" * width)
            self.relax_energies[irelax, 0] = self.E_dsrg.real
            self.relax_energies[irelax, 2] = self.ints["E"].real
            self.E_relaxed_ref = self.do_reference_relaxation()
            self.relax_eigvals_history.append(self.relax_eigvals)
            self.relax_energies[irelax, 1] = self.E_relaxed_ref.real

            logger.log_info1(
                f"{irelax:>10d} {self.relax_energies[irelax,2]:>25.12f} {self.relax_energies[irelax,0]:>25.12f} {self.relax_energies[irelax,1]:>25.12f}"
            )

            # "once": DSRG -> relax -> done
            if self.relax_reference == "once":
                self.converged = True
                break

            self.converged = self.test_relaxation_convergence(irelax)
            if self.converged or irelax == self.nrelax - 1:
                # leaving on the last iteration too: the work below only
                # prepares the next one, and there isn't one
                break

            # Drop the previous set first: get_integrals() does not read self.ints,
            # so holding both across the call would double the integral footprint on
            # every relaxation iteration. Subclasses extend this to the working
            # tensors they hang off self, which for MRPT3 is most of the footprint
            # and would otherwise accumulate across iterations.
            self._release_integrals()
            self.ints, self.cumulants = self.get_integrals()
            self.E_dsrg = self._solve_dsrg_shifted(form_hbar)
            self.E = self.E_dsrg

        if self.nrelax > 0 and not self.converged:
            logger.log_warning(
                f"DSRG reference relaxation did not converge in {self.nrelax} iterations."
            )
        if self.nrelax > 0:
            logger.log_info1("=" * width)
        logger.log_info1("\nFinal DSRG energies (a.u.):")
        logger.log_info1(f"  E0       : {self.ints['E'].real:.12f}")
        logger.log_info1(f"  Edsrg    : {self.E_dsrg.real:.12f}")
        if self.nrelax > 0:
            logger.log_info1(f"  Erelaxed : {self.E_relaxed_ref.real:.12f}")
            if len(self.relax_eigvals) > 1:
                logger.log_info1("")
                pretty_print_ci_summary(
                    self.ci_solver.sa_info, self.ci_solver.evals_per_solver
                )
        self.relax_eigvals_history = np.array(self.relax_eigvals_history)
        self._post_process()
        self.executed = True
        return self

    def test_relaxation_convergence(self, irelax):
        if irelax == 0:
            return False

        delta_fixed_ref = abs(
            self.relax_energies[irelax, 0] - self.relax_energies[irelax - 1, 0]
        )
        delta_relaxed_ref = abs(
            self.relax_energies[irelax, 1] - self.relax_energies[irelax - 1, 1]
        )
        delta = abs(self.relax_energies[irelax, 1] - self.relax_energies[irelax, 0])

        if all(e < self.relax_tol for e in [delta_fixed_ref, delta_relaxed_ref, delta]):
            return True
        else:
            return False

    @abstractmethod
    def solve_dsrg(self): ...

    @abstractmethod
    def do_reference_relaxation(self): ...

    @abstractmethod
    def get_integrals(self): ...

    @property
    def hbar0(self):
        if self._hbar0 is None:
            raise RuntimeError(
                "hbar0 is only available after reference relaxation or a run "
                "with save_hbar=True!"
            )
        return self._hbar0

    @property
    def hbar1_canon(self):
        if self._hbar1_canon is None:
            raise RuntimeError(
                "hbar1_canon is only available after reference relaxation or "
                "a run with save_hbar=True!"
            )
        return self._hbar1_canon

    @property
    def hbar2_canon(self):
        if self._hbar2_canon is None:
            raise RuntimeError(
                "hbar2_canon is only available after reference relaxation or "
                "a run with save_hbar=True!"
            )
        return self._hbar2_canon
