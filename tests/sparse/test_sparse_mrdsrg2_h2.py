import contextlib
import io

import numpy as np
import pytest

import forte2
from forte2.lib import sparse_ops
from forte2.lib.det import Determinant


def cas22_reference(hamiltonian, norb):
    model_space = tuple(
        Determinant(label + "0" * (norb - 2)) for label in ("20", "ab", "ba", "02")
    )
    matrix = hamiltonian.matrix(model_space)
    energies, vectors = np.linalg.eigh(matrix)
    vacuum = sparse_ops.SparseState(
        {
            determinant: coefficient
            for determinant, coefficient in zip(model_space, vectors[:, 0])
            if abs(coefficient) > 1.0e-14
        }
    )
    return vacuum, float(energies[0].real)


@pytest.mark.slow
def test_mrldsrg2_h2_ccpvdz_matches_wickd_and_legacy_forte():
    with contextlib.redirect_stdout(io.StringIO()):
        system = forte2.System(
            xyz="H 0 0 0\nH 0 0 0.75",
            basis_set="cc-pVDZ",
            unit="angstrom",
            cholesky_tei=True,
            cholesky_tol=1.0e-12,
        )
        rhf = forte2.RHF(charge=0, e_tol=1.0e-12)(system)
        rhf.run()

    coefficients = rhf.C[0]
    norb = rhf.nmo
    hcore = np.einsum(
        "pq,pi,qj->ij", system.ints_hcore(), coefficients, coefficients, optimize=True
    )
    eri = system.fock_builder.two_electron_integrals_block(coefficients)
    hamiltonian = sparse_ops.sparse_operator_hamiltonian(
        system.nuclear_repulsion, hcore, eri
    )
    vacuum, reference_energy = cas22_reference(hamiltonian, norb)

    cumulants = sparse_ops.CumulantReference(vacuum, norb, max_cumulant=3)
    gamma = np.array(
        [
            [
                sum(cumulants.gamma(p, alpha, q, alpha).real for alpha in (True, False))
                for q in range(norb)
            ]
            for p in range(norb)
        ]
    )
    fock = hcore + np.einsum("rs,prqs->pq", gamma, eri, optimize=True)
    fock -= 0.5 * np.einsum("rs,prsq->pq", gamma, eri, optimize=True)

    rotation = np.zeros_like(fock)
    for orbital_slice in (slice(0, 2), slice(2, norb)):
        _, vectors = np.linalg.eigh(fock[orbital_slice, orbital_slice])
        rotation[orbital_slice, orbital_slice] = vectors
    hcore = rotation.T @ hcore @ rotation
    eri = np.einsum(
        "pi,qj,pqrs,rk,sl->ijkl",
        rotation,
        rotation,
        eri,
        rotation,
        rotation,
        optimize=True,
    )
    hamiltonian = sparse_ops.sparse_operator_hamiltonian(
        system.nuclear_repulsion, hcore, eri
    )
    vacuum, reference_energy = cas22_reference(hamiltonian, norb)
    excitations = forte2.enumerate_mrdsrg_excitations(
        core_orbitals=[],
        active_orbitals=[0, 1],
        virtual_orbitals=range(2, norb),
        orbital_energies=np.diag(rotation.T @ fock @ rotation),
        max_rank=2,
    )

    result = forte2.solve_sparse_mrdsrg2(
        hamiltonian,
        vacuum,
        norb,
        excitations,
        flow_param=1.0,
        max_cumulant=3,
        gno_backend="validate",
        max_commutators=4,
        commutator_threshold=0.0,
        maxiter=60,
        e_tol=1.0e-10,
        r_tol=1.0e-8,
        do_diis=True,
        diis_start=2,
        diis_nvec=8,
        initial_amplitudes=np.zeros(len(excitations)),
    )

    assert reference_energy == pytest.approx(-1.131600248802019, abs=3.0e-12)
    assert result.history[0].energy - reference_energy == pytest.approx(
        0.0, abs=3.0e-12
    )
    assert result.history[1].energy - reference_energy == pytest.approx(
        -0.030204971583883370, abs=3.0e-10
    )
    assert result.converged
    assert result.energy == pytest.approx(-1.1636262944980262, abs=2.0e-11)
