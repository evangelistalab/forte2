"""Sparse-reference MR-DSRG(n) prototype built on generalized normal ordering."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from forte2.helpers import DIIS
from forte2.lib import sparse_ops as _ft
from forte2.lib.det import Determinant

SCREEN = 1.0e-12


@dataclass(frozen=True)
class SparseMRDSRGExcitation:
    """A spin-orbital excitation operator and its DSRG denominator."""

    sqop: _ft.SQOperatorString
    denominator: float
    rank: int
    label: str = ""


@dataclass(frozen=True)
class SparseMRDSRGIteration:
    """One sparse MR-DSRG fixed-point iteration."""

    iteration: int
    energy: float
    delta_energy: float
    rms_update: float
    ncomm: int
    iter_s: float


@dataclass(frozen=True)
class SparseMRDSRGResult:
    """Sparse MR-DSRG result."""

    energy: float
    scalar_energy: float
    model_space_energies: np.ndarray | None
    converged: bool
    iterations: int
    max_rank: int
    max_cumulant: int
    gno_backend: str
    amplitudes: np.ndarray
    excitations: tuple[SparseMRDSRGExcitation, ...]
    history: tuple[SparseMRDSRGIteration, ...]
    hbar: _ft.GeneralizedNormalOrderedSparseOperator
    seconds: float


def regularized_denominator(denominator: float, flow_param: float) -> float:
    """Return the DSRG regularizer ``(1 - exp(-s Delta^2)) / Delta``."""
    if abs(denominator) < SCREEN:
        return flow_param * denominator
    return (1.0 - math.exp(-flow_param * denominator * denominator)) / denominator


def _mode_token(mode: tuple[int, str], creation: bool) -> str:
    orbital, spin = mode
    return f"{orbital}{spin}{'+' if creation else '-'}"


def canonical_operator_label(
    creation_modes: Sequence[tuple[int, str]],
    annihilation_modes: Sequence[tuple[int, str]],
) -> str:
    """Return Forte2's canonical sparse operator string for a spin-orbital operator."""
    alpha_cre = sorted(
        [mode for mode in creation_modes if mode[1] == "a"], key=lambda mode: mode[0]
    )
    beta_cre = sorted(
        [mode for mode in creation_modes if mode[1] == "b"], key=lambda mode: mode[0]
    )
    beta_ann = sorted(
        [mode for mode in annihilation_modes if mode[1] == "b"],
        key=lambda mode: mode[0],
        reverse=True,
    )
    alpha_ann = sorted(
        [mode for mode in annihilation_modes if mode[1] == "a"],
        key=lambda mode: mode[0],
        reverse=True,
    )

    tokens = [_mode_token(mode, True) for mode in alpha_cre]
    tokens += [_mode_token(mode, True) for mode in beta_cre]
    tokens += [_mode_token(mode, False) for mode in beta_ann]
    tokens += [_mode_token(mode, False) for mode in alpha_ann]
    return "[" + " ".join(tokens) + "]"


def _sqop_from_label(label: str) -> _ft.SQOperatorString:
    return next(iter(_ft.sparse_operator(label, 1.0)))[0]


def identity_sqop() -> _ft.SQOperatorString:
    """Return the identity sparse operator string."""
    return _sqop_from_label("[]")


def _number_operator_string(orbital: int, alpha: bool) -> _ft.SQOperatorString:
    spin = "a" if alpha else "b"
    return _sqop_from_label(f"[{orbital}{spin}+ {orbital}{spin}-]")


def _normal_ordered_diagonal_fock(
    h0: _ft.GeneralizedNormalOrderedSparseOperator, norb: int
) -> dict[tuple[int, bool], float]:
    """Return spin-orbital diagonal one-body coefficients of a GNO Hamiltonian."""
    diagonal = {}
    for orbital in range(norb):
        for alpha in (True, False):
            key = _number_operator_string(orbital, alpha)
            diagonal[(orbital, alpha)] = float(h0.coefficient(key).real)
    return diagonal


def _denominator_from_diagonal_fock(
    excitation: "SparseMRDSRGExcitation",
    diagonal_fock: dict[tuple[int, bool], float],
    norb: int,
) -> float:
    denominator = 0.0
    cre = excitation.sqop.cre()
    ann = excitation.sqop.ann()
    for orbital in range(norb):
        if cre.na(orbital):
            denominator -= diagonal_fock[(orbital, True)]
        if cre.nb(orbital):
            denominator -= diagonal_fock[(orbital, False)]
        if ann.na(orbital):
            denominator += diagonal_fock[(orbital, True)]
        if ann.nb(orbital):
            denominator += diagonal_fock[(orbital, False)]
    return denominator


def _with_normal_ordered_denominators(
    excitations: Sequence["SparseMRDSRGExcitation"],
    h0: _ft.GeneralizedNormalOrderedSparseOperator,
    norb: int,
) -> tuple["SparseMRDSRGExcitation", ...]:
    diagonal_fock = _normal_ordered_diagonal_fock(h0, norb)
    return tuple(
        SparseMRDSRGExcitation(
            sqop=excitation.sqop,
            denominator=_denominator_from_diagonal_fock(
                excitation, diagonal_fock, norb
            ),
            rank=excitation.rank,
            label=excitation.label,
        )
        for excitation in excitations
    )


def enumerate_mrdsrg_excitations(
    core_orbitals: Sequence[int],
    active_orbitals: Sequence[int],
    virtual_orbitals: Sequence[int],
    orbital_energies: Sequence[float],
    max_rank: int = 2,
) -> tuple[SparseMRDSRGExcitation, ...]:
    """Enumerate spin-conserving MR-DSRG external excitation operators.

    The excitation manifold follows the usual MR-DSRG hole/particle split:
    holes are core + active orbitals, particles are active + virtual orbitals,
    and pure active operators are excluded. The denominators built here are
    provisional; :class:`SparseMRDSRG` replaces them by default with diagonal
    one-body coefficients of the generalized normal-ordered Hamiltonian.
    """
    if max_rank < 1:
        return tuple()

    eps = np.asarray(orbital_energies)
    hole_modes = [(i, spin) for i in core_orbitals for spin in ("a", "b")]
    hole_modes += [(u, spin) for u in active_orbitals for spin in ("a", "b")]
    particle_modes = [(u, spin) for u in active_orbitals for spin in ("a", "b")]
    particle_modes += [(a, spin) for a in virtual_orbitals for spin in ("a", "b")]

    excitations = []
    highest_rank = min(max_rank, len(hole_modes), len(particle_modes))
    active_set = set(active_orbitals)

    for rank in range(1, highest_rank + 1):
        for ann_modes in itertools.combinations(hole_modes, rank):
            ann_spins = sorted(spin for _, spin in ann_modes)
            for cre_modes in itertools.combinations(particle_modes, rank):
                if ann_spins != sorted(spin for _, spin in cre_modes):
                    continue
                all_active = all(mode[0] in active_set for mode in ann_modes) and all(
                    mode[0] in active_set for mode in cre_modes
                )
                if all_active:
                    continue

                label = canonical_operator_label(cre_modes, ann_modes)
                denominator = sum(eps[i] for i, _ in ann_modes) - sum(
                    eps[a] for a, _ in cre_modes
                )
                excitations.append(
                    SparseMRDSRGExcitation(
                        sqop=_sqop_from_label(label),
                        denominator=float(denominator),
                        rank=rank,
                        label=label,
                    )
                )

    return tuple(excitations)


def _empty_gno(
    vacuum: _ft.SparseState, norb: int, max_cumulant: int
) -> _ft.GeneralizedNormalOrderedSparseOperator:
    return _ft.GeneralizedNormalOrderedSparseOperator(vacuum, norb, max_cumulant)


def _copy_gno(
    op: _ft.GeneralizedNormalOrderedSparseOperator,
) -> _ft.GeneralizedNormalOrderedSparseOperator:
    result = _empty_gno(op.vacuum(), op.norb(), op.max_cumulant())
    for term, coefficient in op:
        result.add(term, coefficient)
    return result


def _make_cluster_operator(
    vacuum: _ft.SparseState,
    norb: int,
    max_cumulant: int,
    excitations: Sequence[SparseMRDSRGExcitation],
    amplitudes: np.ndarray,
    screen_thresh: float,
) -> _ft.GeneralizedNormalOrderedSparseOperator:
    op = _empty_gno(vacuum, norb, max_cumulant)
    for excitation, amplitude in zip(excitations, amplitudes):
        if abs(amplitude) > screen_thresh:
            op.add(excitation.sqop, complex(amplitude))
    return op


def _make_antihermitian_cluster_operator(
    vacuum: _ft.SparseState,
    norb: int,
    max_cumulant: int,
    excitations: Sequence[SparseMRDSRGExcitation],
    amplitudes: np.ndarray,
    screen_thresh: float,
) -> _ft.GeneralizedNormalOrderedSparseOperator:
    op = _empty_gno(vacuum, norb, max_cumulant)
    for excitation, amplitude in zip(excitations, amplitudes):
        if abs(amplitude) <= screen_thresh:
            continue
        op.add(excitation.sqop, complex(amplitude))
        op.add(excitation.sqop.adjoint(), -complex(amplitude).conjugate())
    return op


def _gno_commutator(
    lhs: _ft.GeneralizedNormalOrderedSparseOperator,
    rhs: _ft.GeneralizedNormalOrderedSparseOperator,
    vacuum: _ft.SparseState,
    norb: int,
    max_cumulant: int,
    max_rank: int,
    screen_thresh: float,
    gno_backend: str = "sparse",
    sparse_product_engine: _ft.GeneralizedNormalOrderedProductComputer | None = None,
    cumulant_engine: _ft.CumulantWickEngine | None = None,
    validation_tol: float = 1.0e-9,
) -> _ft.GeneralizedNormalOrderedSparseOperator:
    sparse_result = None
    if gno_backend in {"sparse", "validate"}:
        if sparse_product_engine is None:
            raise RuntimeError(
                "the cumulant-truncated sparse backend was not initialized"
            )
        sparse_result = sparse_product_engine.commutator(lhs, rhs)
    if gno_backend == "sparse":
        return sparse_result
    if gno_backend == "rdm":
        return lhs.commutator(rhs, max_rank=max_rank, screen_thresh=screen_thresh)

    if cumulant_engine is None:
        raise RuntimeError("the cumulant Wick backend was not initialized")
    cumulant_result = cumulant_engine.commutator(lhs, rhs)
    if gno_backend == "cumulant":
        return cumulant_result

    sparse_terms = {term.str(): coefficient for term, coefficient in sparse_result}
    cumulant_terms = {term.str(): coefficient for term, coefficient in cumulant_result}
    for term in sparse_terms.keys() | cumulant_terms.keys():
        difference = abs(sparse_terms.get(term, 0.0) - cumulant_terms.get(term, 0.0))
        if difference > validation_tol:
            raise RuntimeError(
                "cumulant Wick backend mismatch for "
                f"{term}: coefficient difference {difference:.3e} exceeds "
                f"{validation_tol:.3e}"
            )
    return cumulant_result


def _bch_hbar(
    h0: _ft.GeneralizedNormalOrderedSparseOperator,
    a_op: _ft.GeneralizedNormalOrderedSparseOperator,
    vacuum: _ft.SparseState,
    norb: int,
    max_cumulant: int,
    max_rank: int,
    max_commutators: int,
    screen_thresh: float,
    commutator_threshold: float,
    gno_backend: str = "sparse",
    sparse_product_engine: _ft.GeneralizedNormalOrderedProductComputer | None = None,
    cumulant_engine: _ft.CumulantWickEngine | None = None,
    validation_tol: float = 1.0e-9,
) -> tuple[_ft.GeneralizedNormalOrderedSparseOperator, tuple[float, ...]]:
    hbar = _copy_gno(h0)
    nested = _copy_gno(h0)
    norms = []

    for ncomm in range(1, max_commutators + 1):
        nested = _gno_commutator(
            nested,
            a_op,
            vacuum,
            norb,
            max_cumulant,
            max_rank,
            screen_thresh,
            gno_backend,
            sparse_product_engine,
            cumulant_engine,
            validation_tol,
        )
        contribution = nested * (1.0 / math.factorial(ncomm))
        hbar += contribution
        norm = contribution.norm()
        norms.append(norm)
        if norm < commutator_threshold:
            break

    return hbar.truncate(max_rank, screen_thresh), tuple(norms)


def _effective_hamiltonian_matrix(
    hbar: _ft.GeneralizedNormalOrderedSparseOperator,
    model_space: Sequence[Determinant],
    screen_thresh: float,
) -> np.ndarray:
    """Return the transformed Hamiltonian matrix in the supplied model space."""
    hbar_sparse = hbar.to_sparse_operator(screen_thresh)
    h_eff = np.zeros((len(model_space), len(model_space)), dtype=complex)
    for col, ket in enumerate(model_space):
        hket = _ft.apply_op(
            hbar_sparse,
            _ft.SparseState({ket: 1.0}),
            screen_thresh=screen_thresh,
        )
        for row, bra in enumerate(model_space):
            h_eff[row, col] = hket[bra]
    return h_eff


def _effective_hamiltonian_energies(
    hbar: _ft.GeneralizedNormalOrderedSparseOperator,
    model_space: Sequence[Determinant] | None,
    screen_thresh: float,
) -> np.ndarray | None:
    if model_space is None:
        return None
    h_eff = _effective_hamiltonian_matrix(hbar, model_space, screen_thresh)
    h_eff = 0.5 * (h_eff + h_eff.conj().T)
    return np.linalg.eigvalsh(h_eff).real


def _select_reported_energy(
    scalar_energy: float,
    model_space_energies: np.ndarray | None,
) -> float:
    if model_space_energies is None:
        return scalar_energy
    return float(model_space_energies[0])


@dataclass
class SparseMRDSRG:
    """Sparse-reference MR-DSRG(n) fixed-point solver.

    The ``sparse`` backend converts through bare operator strings while
    reconstructing density moments with unavailable cumulants set to zero. The
    ``cumulant`` backend evaluates the same generalized Wick expansion directly.
    ``validate`` compares these independent implementations term by term. The
    historical density-moment rank truncation remains available as ``rdm``.
    Cumulants through rank 3 are used by default. Molecular inputs must be
    semicanonical within their core, active, and virtual orbital subspaces.
    """

    hamiltonian: _ft.SparseOperator
    vacuum: _ft.SparseState
    norb: int
    excitations: Sequence[SparseMRDSRGExcitation]
    flow_param: float = 0.5
    max_cumulant: int = 3
    max_rank: int = 2
    gno_backend: str = "sparse"
    gno_validation_tol: float = 1.0e-9
    max_commutators: int = 20
    maxiter: int = 50
    e_tol: float = 1.0e-10
    r_tol: float = 1.0e-8
    screen_thresh: float = SCREEN
    commutator_threshold: float = SCREEN
    diis_start: int = 3
    diis_nvec: int = 8
    do_diis: bool = False
    damping: float = 1.0
    use_hamiltonian_denominators: bool = True
    initial_amplitudes: np.ndarray | None = None
    model_space: Sequence[Determinant] | None = None
    history: list[SparseMRDSRGIteration] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.max_rank < 1:
            raise ValueError("max_rank must be positive")
        if self.max_cumulant < -1:
            raise ValueError("max_cumulant must be non-negative or -1")
        if self.gno_backend not in {"sparse", "cumulant", "validate", "rdm"}:
            raise ValueError(
                "gno_backend must be one of 'sparse', 'cumulant', 'validate', or 'rdm'"
            )
        if self.gno_backend in {"sparse", "cumulant", "validate"} and self.max_rank > 4:
            raise ValueError(
                "the cumulant-truncated backends currently support max_rank <= 4"
            )
        if self.gno_backend in {"sparse", "cumulant", "validate"} and not (
            1 <= self.max_cumulant <= 4
        ):
            raise ValueError(
                "the cumulant-truncated backends currently support max_cumulant from 1 to 4"
            )
        if self.gno_validation_tol < 0.0:
            raise ValueError("gno_validation_tol must be non-negative")
        if self.flow_param < 0.0:
            raise ValueError("flow_param must be non-negative")
        if self.screen_thresh < 0.0:
            raise ValueError("screen_thresh must be non-negative")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must be in the interval (0, 1]")
        self.excitations = tuple(
            excitation
            for excitation in self.excitations
            if excitation.rank <= self.max_rank
        )
        if self.model_space is not None:
            self.model_space = tuple(self.model_space)
            if len(self.model_space) == 0:
                raise ValueError("model_space must contain at least one determinant")

    def _initial_amplitudes(
        self,
        h0: _ft.GeneralizedNormalOrderedSparseOperator,
        excitations: Sequence[SparseMRDSRGExcitation],
    ) -> np.ndarray:
        if self.initial_amplitudes is not None:
            amplitudes = np.asarray(self.initial_amplitudes, dtype=complex)
            if amplitudes.shape != (len(excitations),):
                raise ValueError("initial_amplitudes has the wrong shape")
            return amplitudes.copy()

        amplitudes = np.zeros(len(excitations), dtype=complex)
        for idx, excitation in enumerate(excitations):
            residual = h0.coefficient(excitation.sqop)
            amplitudes[idx] = residual * regularized_denominator(
                excitation.denominator, self.flow_param
            )
        return amplitudes

    def run(self) -> SparseMRDSRGResult:
        start_s = time.perf_counter()
        identity = identity_sqop()
        h0 = _ft.generalized_normal_order(
            self.hamiltonian,
            self.vacuum,
            self.norb,
            max_cumulant=self.max_cumulant,
            screen_thresh=self.screen_thresh,
            max_rank=self.max_rank,
        )

        if len(self.excitations) == 0:
            scalar_energy = float(h0.coefficient(identity).real)
            model_space_energies = _effective_hamiltonian_energies(
                h0, self.model_space, self.screen_thresh
            )
            energy = _select_reported_energy(scalar_energy, model_space_energies)
            return SparseMRDSRGResult(
                energy=energy,
                scalar_energy=scalar_energy,
                model_space_energies=model_space_energies,
                converged=True,
                iterations=0,
                max_rank=self.max_rank,
                max_cumulant=self.max_cumulant,
                gno_backend=self.gno_backend,
                amplitudes=np.zeros(0, dtype=complex),
                excitations=tuple(),
                history=tuple(),
                hbar=h0,
                seconds=time.perf_counter() - start_s,
            )

        excitations = (
            _with_normal_ordered_denominators(self.excitations, h0, self.norb)
            if self.use_hamiltonian_denominators
            else self.excitations
        )
        amplitudes = self._initial_amplitudes(h0, excitations)
        sparse_product_engine = None
        cumulant_engine = None
        if self.gno_backend in {"sparse", "cumulant", "validate"}:
            cumulant_reference = _ft.CumulantReference(
                self.vacuum,
                self.norb,
                max_cumulant=self.max_cumulant,
                screen_thresh=self.screen_thresh,
            )
        if self.gno_backend in {"sparse", "validate"}:
            sparse_product_engine = _ft.GeneralizedNormalOrderedProductComputer(
                cumulant_reference,
                self.max_rank,
                screen_thresh=self.screen_thresh,
            )
        if self.gno_backend in {"cumulant", "validate"}:
            cumulant_engine = _ft.CumulantWickEngine(
                cumulant_reference,
                self.max_rank,
                screen_thresh=self.screen_thresh,
            )
        diis = DIIS(
            diis_start=self.diis_start,
            diis_nvec=self.diis_nvec,
            diis_min=min(self.diis_start, self.diis_nvec),
            do_diis=self.do_diis,
        )
        previous_energy = None
        hbar = h0
        converged = False

        for iteration in range(self.maxiter + 1):
            iter_s = time.perf_counter()
            a_op = _make_antihermitian_cluster_operator(
                self.vacuum,
                self.norb,
                self.max_cumulant,
                excitations,
                amplitudes,
                self.screen_thresh,
            )
            hbar, commutator_norms = _bch_hbar(
                h0,
                a_op,
                self.vacuum,
                self.norb,
                self.max_cumulant,
                self.max_rank,
                self.max_commutators,
                self.screen_thresh,
                self.commutator_threshold,
                self.gno_backend,
                sparse_product_engine,
                cumulant_engine,
                self.gno_validation_tol,
            )
            scalar_energy = float(hbar.coefficient(identity).real)
            model_space_energies = _effective_hamiltonian_energies(
                hbar, self.model_space, self.screen_thresh
            )
            energy = _select_reported_energy(scalar_energy, model_space_energies)
            hbar_offdiag = np.array(
                [hbar.coefficient(excitation.sqop) for excitation in excitations],
                dtype=complex,
            )
            fixed_point = np.array(
                [
                    (hbar_offdiag[idx] + excitation.denominator * amplitudes[idx])
                    * regularized_denominator(excitation.denominator, self.flow_param)
                    for idx, excitation in enumerate(excitations)
                ],
                dtype=complex,
            )
            damped_fixed_point = amplitudes + self.damping * (fixed_point - amplitudes)
            update = damped_fixed_point - amplitudes
            rms_update = float(np.linalg.norm(update) / math.sqrt(len(update)))
            delta_energy = 0.0 if previous_energy is None else energy - previous_energy
            self.history.append(
                SparseMRDSRGIteration(
                    iteration=iteration,
                    energy=energy,
                    delta_energy=delta_energy,
                    rms_update=rms_update,
                    ncomm=len(commutator_norms),
                    iter_s=time.perf_counter() - iter_s,
                )
            )

            if (
                previous_energy is not None
                and abs(delta_energy) < self.e_tol
                and rms_update < self.r_tol
            ):
                converged = True
                break

            amplitudes = (
                diis.update(damped_fixed_point, update)
                if self.do_diis
                else damped_fixed_point
            )
            previous_energy = energy

        return SparseMRDSRGResult(
            energy=energy,
            scalar_energy=scalar_energy,
            model_space_energies=model_space_energies,
            converged=converged,
            iterations=len(self.history),
            max_rank=self.max_rank,
            max_cumulant=self.max_cumulant,
            gno_backend=self.gno_backend,
            amplitudes=amplitudes,
            excitations=tuple(excitations),
            history=tuple(self.history),
            hbar=hbar,
            seconds=time.perf_counter() - start_s,
        )


SparseMRDSRG2 = SparseMRDSRG


def solve_sparse_mrdsrg(
    hamiltonian: _ft.SparseOperator,
    vacuum: _ft.SparseState,
    norb: int,
    excitations: Sequence[SparseMRDSRGExcitation],
    max_rank: int = 2,
    **kwargs,
) -> SparseMRDSRGResult:
    """Run sparse-reference MR-DSRG(n), truncated to ``max_rank``."""
    return SparseMRDSRG(
        hamiltonian=hamiltonian,
        vacuum=vacuum,
        norb=norb,
        excitations=excitations,
        max_rank=max_rank,
        **kwargs,
    ).run()


def _solve_sparse_mrdsrg_fixed_rank(
    rank: int,
    hamiltonian: _ft.SparseOperator,
    vacuum: _ft.SparseState,
    norb: int,
    excitations: Sequence[SparseMRDSRGExcitation],
    **kwargs,
) -> SparseMRDSRGResult:
    requested_rank = kwargs.pop("max_rank", rank)
    if requested_rank != rank:
        raise ValueError(f"solve_sparse_mrdsrg{rank} requires max_rank={rank}")
    return solve_sparse_mrdsrg(
        hamiltonian,
        vacuum,
        norb,
        excitations,
        max_rank=rank,
        **kwargs,
    )


def solve_sparse_mrdsrg2(
    hamiltonian: _ft.SparseOperator,
    vacuum: _ft.SparseState,
    norb: int,
    excitations: Sequence[SparseMRDSRGExcitation],
    **kwargs,
) -> SparseMRDSRGResult:
    """Run sparse-reference MR-DSRG(2)."""
    return _solve_sparse_mrdsrg_fixed_rank(
        2,
        hamiltonian=hamiltonian,
        vacuum=vacuum,
        norb=norb,
        excitations=excitations,
        **kwargs,
    )


def solve_sparse_mrdsrg3(
    hamiltonian: _ft.SparseOperator,
    vacuum: _ft.SparseState,
    norb: int,
    excitations: Sequence[SparseMRDSRGExcitation],
    **kwargs,
) -> SparseMRDSRGResult:
    """Run sparse-reference MR-DSRG(3)."""
    return _solve_sparse_mrdsrg_fixed_rank(
        3,
        hamiltonian=hamiltonian,
        vacuum=vacuum,
        norb=norb,
        excitations=excitations,
        **kwargs,
    )


def solve_sparse_mrdsrg4(
    hamiltonian: _ft.SparseOperator,
    vacuum: _ft.SparseState,
    norb: int,
    excitations: Sequence[SparseMRDSRGExcitation],
    **kwargs,
) -> SparseMRDSRGResult:
    """Run sparse-reference MR-DSRG(4)."""
    return _solve_sparse_mrdsrg_fixed_rank(
        4,
        hamiltonian=hamiltonian,
        vacuum=vacuum,
        norb=norb,
        excitations=excitations,
        **kwargs,
    )
