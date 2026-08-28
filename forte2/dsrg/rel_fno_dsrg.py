from dataclasses import dataclass, field

from forte2.base_classes import Method

from .rel_dsrg_mrpt2 import RelDSRG_MRPT2
from .rel_dsrg_mrpt3 import RelDSRG_MRPT3


@dataclass
class RelFNO_DSRG_MRPT3(Method):
    """
    Two-component relativistic DSRG-MRPT3 in a frozen natural orbital (FNO)
    truncated virtual space.

    Parameters
    ----------
    flow_param : float, optional, default=2.0
        The DSRG-MRPT3 flow parameter ("s2" in ref. [1]).
    fno_flow_param : float, optional, default=1.5
        The flow parameter used to build the natural orbitals and the
        truncation correction ("s1" in ref. [1]). Independent of flow_param,
        and typically smaller.
    fno_p_o : float, optional, default=None
        Retain the smallest set of leading virtual natural orbitals whose
        cumulative occupation is at least this fraction (0, 1] of the total.
        Mutually exclusive with fno_n_kappa; exactly one is required.
    fno_n_kappa : float, optional, default=None
        Retain all virtual natural orbitals with occupation number >=
        fno_n_kappa. Mutually exclusive with fno_p_o.
    fno_degeneracy_tol : float, optional, default=1e-2
        Push the truncation boundary outward while the occupation numbers
        straddling it differ by less than this fraction of the larger one, so
        near-degenerate natural orbitals (e.g. Kramers partners) are never
        split between the retained and discarded sets.
    relax_reference, relax_maxiter, relax_tol
        Reference relaxation options, applied to the DSRG-MRPT3 step. See
        RelDSRG_MRPT3.
    frozen_core_orbitals, frozen_virtual_orbitals : int | list[int], optional
        Orbitals frozen in the correlation treatment, applied to the
        full-space DSRG-MRPT2 step and inherited by the rest of the chain.

    Attributes
    ----------
    E : float
        The FNO-corrected DSRG-MRPT3 energy (relaxed, if relaxation was
        requested).
    E_dsrg : float
        The FNO-corrected DSRG-MRPT3 energy with a fixed reference.
    E_relaxed_ref : float
        The FNO-corrected DSRG-MRPT3 energy after reference relaxation. Only
        set if relaxation was requested.
    pt2_full, pt2_fno, pt3 : RelDSRG_MRPT2, RelDSRG_MRPT2, RelDSRG_MRPT3
        The composed solvers, for inspecting intermediates (e.g.
        ``pt2_fno.hbar_shift`` is the truncation correction, and
        ``pt2_full.mo_space.nvirt`` the retained virtual count).

    Note
    ----
    This class is a composition of three DSRG solvers: pt2_full -> pt2_fno -> pt3,
    where "pt2_full" runs DSRG-MRPT2 with the full virtual space at "fno_flow_param"
    and truncates its own virtual space to the leading natural orbitals of the
    unrelaxed virtual-virtual 1-RDM. "pt2_fno" repeats the same calculation in
    that truncated space and publishes the difference of the two effective
    Hamiltonians as an hbar_shift. "pt3" runs DSRG-MRPT3 in the truncated space
    at "flow_param" and folds that shift into its energy and, when relaxing,
    into the Hamiltonian handed to the CI solver.

    References
    ----------
    .. [1] C. Li, S. Mao, R. Huang, F. A. Evangelista, "Frozen Natural Orbitals for the State-Averaged Driven Similarity Renormalization Group",
           J. Chem. Theory Comput. 2024, 20, 4170-4181.
    """

    flow_param: float = 0.5
    fno_flow_param: float = 1.5

    fno_p_o: float | None = None
    fno_n_kappa: float | None = None
    fno_degeneracy_tol: float = 1e-2

    relax_reference: int | str | bool = False
    relax_maxiter: int = 10
    relax_tol: float = 1e-6

    frozen_core_orbitals: int | list[int] = None
    frozen_virtual_orbitals: int | list[int] = None

    pt2_full: RelDSRG_MRPT2 | None = field(init=False, default=None)
    pt2_fno: RelDSRG_MRPT2 | None = field(init=False, default=None)
    pt3: RelDSRG_MRPT3 | None = field(init=False, default=None)

    def __post_init__(self):
        self.requires = {"system", "mos", "mo_space"}
        self.provides = {"system", "mos", "mo_space"}
        self.requires_attrs.update({"ci_solver": None, "two_component": True})

        assert (self.fno_p_o is None) != (
            self.fno_n_kappa is None
        ), "Specify exactly one of fno_p_o or fno_n_kappa."

    def __call__(self, parent_method):
        self._register_parent_method(parent_method)
        return self

    def run(self):
        # Full virtual space: build the natural orbitals and truncate. Its
        # effective Hamiltonian is kept so pt2_fno can difference against it.
        self.pt2_full = RelDSRG_MRPT2(
            flow_param=self.fno_flow_param,
            frozen_core_orbitals=self.frozen_core_orbitals,
            frozen_virtual_orbitals=self.frozen_virtual_orbitals,
            fno_p_o=self.fno_p_o,
            fno_n_kappa=self.fno_n_kappa,
            fno_degeneracy_tol=self.fno_degeneracy_tol,
            save_hbar=True,
        )(self.parent_method)

        # Truncated space, same flow parameter: publishes the truncation
        # correction as an hbar_shift.
        self.pt2_fno = RelDSRG_MRPT2(
            flow_param=self.fno_flow_param,
            compute_hbar_shift=True,
        )(self.pt2_full)

        # High-level method, own flow parameter: picks the shift up.
        self.pt3 = RelDSRG_MRPT3(
            flow_param=self.flow_param,
            relax_reference=self.relax_reference,
            relax_maxiter=self.relax_maxiter,
            relax_tol=self.relax_tol,
        )(self.pt2_fno)
        self.pt3.run()

        self.system = self.pt3.system
        self.mos = self.pt3.mos
        self.mo_space = self.pt3.mo_space

        self.E_dsrg = self.pt3.E_dsrg
        self.converged = self.pt3.converged
        self.relax_energies = self.pt3.relax_energies
        self.relax_eigvals_history = self.pt3.relax_eigvals_history
        if self.pt3.nrelax > 0:
            self.relax_eigvals = self.pt3.relax_eigvals
            self.E_relaxed_ref = self.pt3.E_relaxed_ref
            self.E = self.E_relaxed_ref
        else:
            self.E = self.E_dsrg

        self.executed = True
        return self
