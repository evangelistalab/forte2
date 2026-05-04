import contextlib
import io
import itertools
import math

import numpy as np
import pytest

import forte2
from forte2 import Determinant, RHF, System
from forte2 import normal_order, sparse_operator, sparse_operator_hamiltonian
from forte2.helpers import DIIS, logger

SCREEN = 1.0e-12
FLOW_PARAM = 5.0
E_TOL = 1.0e-10
R_TOL = 1.0e-5
MAX_ITER = 80


EXPECTED_DSRG_ENERGIES = {
    2: {
        2: -1.137831710258920,
        3: -1.137283834651968,
        4: -1.137283834651968,
    },
    4: {
        2: -2.140264303595860,
        3: -2.138895129858825,
        4: -2.138889864332239,
    },
    6: {
        2: -3.144610080816127,
        3: -3.142386244444168,
        4: -3.142365107876118,
    },
}


def build_linear_h_sparse_hamiltonian(natoms, spacing=0.74):
    xyz = "\n".join(f"H 0.0 0.0 {i * spacing:.12f}" for i in range(natoms))
    with contextlib.redirect_stdout(io.StringIO()):
        system = System(
            xyz=xyz,
            basis_set="sto-3g",
            minao_basis_set=None,
            cholesky_tei=True,
            cholesky_tol=1.0e-12,
        )
        rhf = RHF(charge=0, e_tol=1.0e-10, d_tol=1.0e-8)(system)
        rhf.run()

    coeff = rhf.C[0]
    hcore_mo = np.einsum(
        "pq,pi,qj->ij", system.ints_hcore(), coeff, coeff, optimize=True
    )
    eri_mo = system.fock_builder.two_electron_integrals_block(coeff)
    ham = sparse_operator_hamiltonian(system.nuclear_repulsion, hcore_mo, eri_mo)
    return rhf, ham, np.array(rhf.eps[0])


def normal_key_and_phase(spop, reference):
    no_op = normal_order(spop, reference, SCREEN)
    items = [(term, coeff) for term, coeff in no_op if abs(coeff) > 1.0e-10]
    if len(items) != 1:
        raise RuntimeError(
            "Expected one normal-ordered term, got "
            + str([(term.str(reference), coeff) for term, coeff in items])
        )
    return items[0]


def physical_coeff(no_op, key, phase):
    return no_op.coefficient(key) / phase


def canonical_excitation_string(cre_modes, ann_modes):
    def token(mode, creation):
        orbital, spin = mode
        return f"{orbital}{spin}{'+' if creation else '-'}"

    alpha_cre = sorted(
        [mode for mode in cre_modes if mode[1] == "a"], key=lambda mode: mode[0]
    )
    beta_cre = sorted(
        [mode for mode in cre_modes if mode[1] == "b"], key=lambda mode: mode[0]
    )
    beta_ann = sorted(
        [mode for mode in ann_modes if mode[1] == "b"],
        key=lambda mode: mode[0],
        reverse=True,
    )
    alpha_ann = sorted(
        [mode for mode in ann_modes if mode[1] == "a"],
        key=lambda mode: mode[0],
        reverse=True,
    )

    tokens = [token(mode, True) for mode in alpha_cre]
    tokens += [token(mode, True) for mode in beta_cre]
    tokens += [token(mode, False) for mode in beta_ann]
    tokens += [token(mode, False) for mode in alpha_ann]
    return "[" + " ".join(tokens) + "]"


def enumerate_spin_conserving_excitations(
    nspatial, nocc, eps, reference, max_excitation_rank
):
    occ = [(i, spin) for i in range(nocc) for spin in ("a", "b")]
    virt = [(a, spin) for a in range(nocc, nspatial) for spin in ("a", "b")]
    highest_rank = min(max_excitation_rank, len(occ), len(virt))
    excitations = []

    for rank in range(1, highest_rank + 1):
        for ann in itertools.combinations(occ, rank):
            ann_spins = sorted(spin for _, spin in ann)
            for cre in itertools.combinations(virt, rank):
                if ann_spins != sorted(spin for _, spin in cre):
                    continue
                label = canonical_excitation_string(cre, ann)
                key, phase = normal_key_and_phase(
                    sparse_operator(label, 1.0), reference
                )
                denom = sum(eps[i] for i, _ in ann) - sum(eps[a] for a, _ in cre)
                excitations.append(
                    {
                        "label": label,
                        "key": key,
                        "phase": phase,
                        "denom": denom,
                    }
                )
    return excitations


def make_normal_ordered_cluster_operator(excitations, amplitudes, reference):
    t_no = forte2.NormalOrderedSparseOperator(reference)
    for excitation, amplitude in zip(excitations, amplitudes):
        if abs(amplitude) > SCREEN:
            t_no.add(excitation["key"], complex(amplitude) * excitation["phase"])
    return t_no


def bch_hbar_dsrg(
    ham, a_no, reference, truncation_rank, max_comm=20, comm_thresh=1.0e-12
):
    hbar = normal_order(ham, reference, SCREEN, max_rank=truncation_rank)
    nested = hbar
    commutator_norms = []

    for ncomm in range(1, max_comm + 1):
        nested = nested.commutator(a_no, truncation_rank, SCREEN)
        contribution = nested * (1.0 / math.factorial(ncomm))
        hbar += contribution

        norm = contribution.norm()
        commutator_norms.append(norm)
        if norm < comm_thresh:
            break

    return hbar.truncate(truncation_rank, SCREEN), commutator_norms


def regularized_denominator(denom, flow_param):
    return (1.0 - math.exp(-flow_param * denom * denom)) / denom


def solve_sparse_dsrg(ham, reference, excitations, truncation_rank):
    ham_no = normal_order(ham, reference, SCREEN, max_rank=truncation_rank)
    h0 = np.array(
        [
            physical_coeff(ham_no, excitation["key"], excitation["phase"])
            for excitation in excitations
        ],
        dtype=complex,
    )
    amplitudes = np.array(
        [
            h0[k] * regularized_denominator(excitation["denom"], FLOW_PARAM)
            for k, excitation in enumerate(excitations)
        ],
        dtype=complex,
    )

    diis = DIIS(diis_start=3, diis_nvec=8, diis_min=3, do_diis=True)
    identity_key, identity_phase = normal_key_and_phase(
        sparse_operator("[]", 1.0), reference
    )
    previous_energy = None

    for iteration in range(MAX_ITER + 1):
        t_no = make_normal_ordered_cluster_operator(excitations, amplitudes, reference)
        a_no = t_no - t_no.adjoint(SCREEN)
        hbar_no, _ = bch_hbar_dsrg(ham, a_no, reference, truncation_rank)
        energy = physical_coeff(hbar_no, identity_key, identity_phase).real
        hbar_offdiag = np.array(
            [
                physical_coeff(hbar_no, excitation["key"], excitation["phase"])
                for excitation in excitations
            ],
            dtype=complex,
        )
        fixed_point = np.array(
            [
                (hbar_offdiag[k] + excitation["denom"] * amplitudes[k])
                * regularized_denominator(excitation["denom"], FLOW_PARAM)
                for k, excitation in enumerate(excitations)
            ],
            dtype=complex,
        )
        update = fixed_point - amplitudes
        rms_update = float(np.linalg.norm(update))
        delta_energy = 0.0 if previous_energy is None else energy - previous_energy

        if (
            previous_energy is not None
            and abs(delta_energy) < E_TOL
            and rms_update < R_TOL
        ):
            return energy, iteration + 1

        amplitudes = diis.update(fixed_point, update)
        previous_energy = energy

    raise RuntimeError(
        f"DSRG({truncation_rank}) did not converge in {MAX_ITER} iterations"
    )


@pytest.mark.slow
@pytest.mark.parametrize("natoms", [2, 4, 6])
def test_linear_h_chain_sparse_dsrg_energies(natoms):
    logger.set_verbosity_level(0)
    rhf, ham, eps = build_linear_h_sparse_hamiltonian(natoms)
    reference = Determinant("2" * rhf.na + "0" * (rhf.nmo - rhf.na))

    for rank, expected_energy in EXPECTED_DSRG_ENERGIES[natoms].items():
        excitations = enumerate_spin_conserving_excitations(
            rhf.nmo, rhf.na, eps, reference, max_excitation_rank=rank
        )
        energy, niter = solve_sparse_dsrg(ham, reference, excitations, rank)

        assert niter <= MAX_ITER
        assert energy == pytest.approx(expected_energy, abs=5.0e-9)
