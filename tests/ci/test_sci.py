from dataclasses import dataclass, field

from forte2 import sparse_operator_hamiltonian, Determinant, SparseState, System
from forte2.scf import RHF
from forte2.jkbuilder import RestrictedMOIntegrals
from forte2.state.state import State
from forte2.base_classes.mixins import MOsMixin, SystemMixin


def parse_state(state: dict) -> tuple:
    if "nel" not in state:
        raise KeyError(
            "The state dictionary must contain the number of electrons ('nel')"
        )
    nel = state["nel"]

    if "multiplicity" in state:
        multiplicity = state["multiplicity"]
    elif "multp" in state:
        multiplicity = state["multp"]
    else:
        raise KeyError(
            "The state dictionary must contain either 'multiplicity' or 'multp'"
        )

    if "ms" not in state:
        raise KeyError("The state dictionary must contain 'ms'")
    ms = state["ms"]
    twoms = round(ms * 2)

    # nel = na + nb
    # 2 ms = na - nb
    na = (nel + twoms) // 2
    nb = nel - na

    return (na, nb, multiplicity, twoms)


@dataclass
class SelectedCI(MOsMixin, SystemMixin):
    orbitals: list[int] | list[list[int]]
    norb: int = field(init=False)
    state: State
    nroot: int
    core_orbitals: list[int] = field(default_factory=list)
    max_iter: int = 2

    def __post_init__(self):
        self.norb = len(self.orbitals)

    def __call__(self, method):
        SystemMixin.copy_from_upstream(self, method)
        MOsMixin.copy_from_upstream(self, method)

        return self

    def run(self):
        print("\nSelected configuration interaction")

        # dets = forte2.hilbert_space(self.norb, self.state.na, self.state.nb)
        ints = RestrictedMOIntegrals(
            self.system, self.C[0], self.orbitals, self.core_orbitals
        )

        H = sparse_operator_hamiltonian(ints.E, ints.H, ints.V)
        ndocc = min(self.state.na, self.state.nb)
        nsocc = max(self.state.na, self.state.nb) - ndocc
        aufbau = Determinant("2" * ndocc + "1" * nsocc)
        self.P = SparseState(aufbau, 1.0)
        # Hmat = H.matrix(dets)
        # eig = np.linalg.eigh(Hmat)[0]
        # print("Eigenvalues:", eig)

        # print(eig[0] + 1.096071975854)

        # pre_iter_preparation()

        for cycle in range(self.max_iter):
            print(f"\nCycle {cycle + 1}")
            # Step 1. Diagonalize the Hamiltonian in the P space
            self._diagonalize_P_space()

            # # Step 2. Find determinants in the Q space
            # find_q_space()

            # # Step 3. Diagonalize the Hamiltonian in the P + Q space
            # diagonalize_PQ_space()

            # # Step 4. Check convergence and break if needed
            # if check_convergence():
            #     break

            # # Step 5. Prune the P + Q space to get an updated P space
            # prune_PQ_to_P()

    # if one_cycle_:
    #     diagonalize_PQ_space()

    # # Post-iter process
    # post_iter_process()

    def _diagonalize_P_space(self):
        """Diagonalize the Hamiltonian in the P space."""
        print(str(self.P))


def test_sci1():
    xyz = f"""
    H 0.0 0.0 0.0
    H 0.0 0.0 {0.529177210903 * 2}
    """

    system = System(xyz=xyz, basis_set="sto-6g", auxiliary_basis_set="cc-pVTZ-JKFIT")

    scf = RHF(charge=0)(system)
    scf.econv = 1e-12
    scf.run()

    sci = SelectedCI(
        orbitals=[0, 1],
        state=State(nel=2, multiplicity=1, ms=0.0),
        nroot=1,
    )(scf)

    sci.run()
