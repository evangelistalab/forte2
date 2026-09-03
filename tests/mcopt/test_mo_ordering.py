from types import SimpleNamespace

import numpy as np

from forte2 import CISolver, MCOptimizer, MOSpace, RHF, State, System
from forte2.helpers.comparisons import approx
from forte2.mcopt import mc_optimizer as mc_optimizer_module
from forte2.mcopt.mc_optimizer import MCOptimizerBase


def test_gasscf_symmetry_invariant_to_noncontiguous_gas_order():
    system = System(
        xyz="H 0.0 0.0 0.0\nH 0.0 0.0 1.4",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
        unit="bohr",
        symmetry=True,
    )
    rhf = RHF(charge=0, e_tol=1e-12)(system)
    state = State(
        nel=2,
        multiplicity=1,
        ms=0.0,
        gas_min=[0, 0],
        gas_max=[2, 2],
    )

    def run(active_orbitals):
        mc = MCOptimizer(
            CISolver(state, active_orbitals=active_orbitals),
            freeze_inter_gas_rots=True,
            final_orbitals="original",
            maxiter=100,
        )(rhf)
        mc.run()
        return mc

    mc_contiguous = run([[0], [1]])
    mc_noncontiguous = run([[1], [0]])

    assert mc_noncontiguous.E == approx(mc_contiguous.E)


def test_mcoptimizer_ao_composition_uses_original_mo_indices(monkeypatch):
    recorded_indices = []

    class RecordingBasisInfo:
        def __init__(self, system, basis):
            pass

        def print_ao_composition(self, C, indices):
            recorded_indices.append(indices)

    monkeypatch.setattr(mc_optimizer_module, "BasisInfo", RecordingBasisInfo)

    mo_space = MOSpace(
        nmo=5,
        frozen_core_orbitals=[2],
        core_orbitals=[0],
        active_orbitals=[[4], [1]],
    )
    mc = object.__new__(MCOptimizerBase)
    mc.system = SimpleNamespace(basis=None)
    mc.mos = SimpleNamespace(C=[np.eye(mo_space.nmo)])
    mc.mo_space = mo_space
    mc.core = mo_space.docc
    mc.actv = mo_space.actv

    mc._print_ao_composition()

    assert recorded_indices == [
        mo_space.docc_indices,
        mo_space.active_indices,
    ]
