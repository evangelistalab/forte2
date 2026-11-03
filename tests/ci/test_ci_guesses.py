import pytest

from forte2 import (
    CI,
    CISolver,
    RHF,
    RelCISolver,
    SpinorUpcaster,
    State,
    System,
)
from forte2.base_classes.params import DavidsonLiuParams
from forte2.helpers.comparisons import approx
from forte2.lib.det import Determinant


def _hf_rhf():
    xyz = """
    H 0.0 0.0 0.0
    F 0.0 0.0 2.0
    """
    system = System(
        xyz=xyz, basis_set="cc-pVDZ", auxiliary_basis_set="cc-pVTZ-JKFIT", unit="bohr"
    )
    return RHF(charge=0, e_tol=1e-12)(system)


def test_ci_guesses():
    """CISolver started from CI vectors, indexed by absolute root, or from guess determinants."""
    rhf = _hf_rhf()
    singlet = State(nel=10, multiplicity=1, ms=0.0)
    triplet = State(nel=10, multiplicity=3, ms=0.0)
    core = [0]
    active = [1, 2, 3, 4, 5, 6]

    # roots 0-2 are the singlet ground state and pi -> sigma* pair, roots 3-4 the triplet pair
    ref_solver = CISolver(
        states=[singlet, triplet],
        nroots=[3, 2],
        core_orbitals=core,
        active_orbitals=active,
    )
    ref = CI(ref_solver)(rhf)
    ref.run()
    assert ref.E_ci[0] == approx(-100.019788438077)

    # get_top_determinants returns guesses in the same format, so it restarts a calculation
    ndet = max(sub_solver.ndet for sub_solver in ref_solver.sub_solvers)
    restart_solver = CISolver(
        states=[singlet, triplet],
        nroots=[3, 2],
        core_orbitals=core,
        active_orbitals=active,
        ci_vector_guess=ref_solver.get_top_determinants(n=ndet),
    )
    restart = CI(restart_solver)(rhf)
    restart.run()
    assert restart.E_ci == approx(ref.E_ci)
    for sub_solver in restart_solver.sub_solvers:
        assert sub_solver.eigensolver.iter <= 2

    # truncated guesses are not normalized, and root 0's lacks the spin partner of 2a220b
    truncated_solver = CISolver(
        states=[singlet, triplet],
        nroots=[3, 2],
        core_orbitals=core,
        active_orbitals=active,
        ci_vector_guess=ref_solver.get_top_determinants(n=10),
    )
    truncated = CI(truncated_solver)(rhf)
    truncated.run()
    assert truncated.E_ci == approx(ref.E_ci)

    # each guess is a list of (determinant, coefficient) pairs, one per root
    guesses = [
        [(Determinant("222200"), 0.99), (Determinant("202220"), -0.06)],
        [
            (Determinant("22a2b0"), 1.0)
        ],  # pi_x -> sigma*, projected onto the singlet CSFs
        [(Determinant("222ab0"), 1.0)],  # pi_y -> sigma*
        [(Determinant("22a2b0"), 1.0)],  # projected onto the triplet CSFs
        [(Determinant("222ab0"), 1.0)],
    ]
    mixed_solver = CISolver(
        states=[singlet, triplet],
        nroots=[3, 2],
        core_orbitals=core,
        active_orbitals=active,
        ci_vector_guess=guesses,
    )
    mixed = CI(mixed_solver)(rhf)
    mixed.run()
    assert mixed.E_ci == approx(ref.E_ci)

    # guess determinants per state, None for the automatic guess
    per_state_solver = CISolver(
        states=[singlet, triplet],
        nroots=[3, 2],
        core_orbitals=core,
        active_orbitals=active,
        guess_dets=[None, [Determinant("22a2b0"), Determinant("222ab0")]],
    )
    per_state = CI(per_state_solver)(rhf)
    per_state.run()
    assert per_state.E_ci == approx(ref.E_ci)

    # one guess vector per root, from one candidate configuration
    single_guess = DavidsonLiuParams(guess_per_root=1, ndets_per_guess=1)
    # the RHF determinant alone gives the RHF energy
    basis_solver = CISolver(
        states=singlet,
        core_orbitals=core,
        active_orbitals=active,
        guess_dets=[Determinant("222200")],
    )
    CI(basis_solver)(rhf).run()
    assert basis_solver.sub_solvers[0].guess_energies[0] == approx(rhf.E)
    # the pi_y^2 -> sigma*^2 determinant is higher in energy, so narrowing drops it
    narrowed_solver = CISolver(
        states=singlet,
        core_orbitals=core,
        active_orbitals=active,
        davidson_liu_params=single_guess,
        guess_dets=[Determinant("222200"), Determinant("222020")],
    )
    CI(narrowed_solver)(rhf).run()
    assert narrowed_solver.sub_solvers[0].guess_energies[0] == approx(rhf.E)
    # pinned, it stays, and its coupling lowers the guess energy
    pinned_solver = CISolver(
        states=singlet,
        core_orbitals=core,
        active_orbitals=active,
        davidson_liu_params=single_guess,
        guess_dets=[Determinant("222200")],
        pinned_guess_dets=[Determinant("222020")],
    )
    pinned = CI(pinned_solver)(rhf)
    pinned.run()
    assert pinned_solver.sub_solvers[0].guess_energies[0] < rhf.E - 1e-5
    assert pinned.E_ci[0] == approx(ref.E_ci[0])


def test_ci_guess_inputs():
    """Invalid guess options are rejected at construction, or against the CI space at run time."""
    rhf = _hf_rhf()
    singlet = State(nel=10, multiplicity=1, ms=0.0)
    triplet = State(nel=10, multiplicity=3, ms=0.0)
    core = [0]
    active = [1, 2, 3, 4, 5, 6]

    # errors that need no CI space are raised at construction
    with pytest.raises(ValueError, match="one guess per root"):
        CISolver(
            states=[singlet, triplet],
            nroots=[3, 2],
            core_orbitals=core,
            active_orbitals=active,
            ci_vector_guess=[[(Determinant("222200"), 1.0)]] * 4,
        )
    with pytest.raises(ValueError, match="every root"):
        CISolver(
            states=singlet,
            nroots=2,
            core_orbitals=core,
            active_orbitals=active,
            ci_vector_guess=[[(Determinant("222200"), 1.0)], None],
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        CISolver(
            states=singlet,
            core_orbitals=core,
            active_orbitals=active,
            ci_vector_guess=[[(Determinant("222200"), 1.0)]],
            pinned_guess_dets=[Determinant("222020")],
        )
    with pytest.raises(ValueError, match="one entry per state"):
        CISolver(
            states=[singlet, triplet],
            nroots=[3, 2],
            core_orbitals=core,
            active_orbitals=active,
            guess_dets=[Determinant("222200")],
        )
    with pytest.raises(TypeError, match=r"\(Determinant, coefficient\) pairs"):
        CISolver(
            states=singlet,
            core_orbitals=core,
            active_orbitals=active,
            ci_vector_guess=[Determinant("222200")],
        )
    with pytest.raises(ValueError, match="duplicate"):
        CISolver(
            states=singlet,
            core_orbitals=core,
            active_orbitals=active,
            ci_vector_guess=[
                [(Determinant("222200"), 0.6), (Determinant("222200"), 0.8)]
            ],
        )
    with pytest.raises(ValueError, match="imaginary"):
        CISolver(
            states=singlet,
            core_orbitals=core,
            active_orbitals=active,
            ci_vector_guess=[[(Determinant("222200"), 1.0j)]],
        )

    # errors against the CI space are raised when the solver runs
    # 7 active electrons instead of 8
    outside_solver = CISolver(
        states=singlet,
        core_orbitals=core,
        active_orbitals=active,
        ci_vector_guess=[[(Determinant("222a00"), 1.0)]],
    )
    outside = CI(outside_solver)(rhf)
    with pytest.raises(ValueError, match="not in the CI space"):
        outside.run()

    # in forte2's phase convention this combination is the triplet, so it has no singlet part
    triplet_guess_solver = CISolver(
        states=singlet,
        core_orbitals=core,
        active_orbitals=active,
        ci_vector_guess=[[(Determinant("22a2b0"), 1.0), (Determinant("22b2a0"), -1.0)]],
    )
    triplet_guess = CI(triplet_guess_solver)(rhf)
    with pytest.raises(ValueError, match="multiplicity"):
        triplet_guess.run()

    # both determinants project onto the same singlet CSF
    dependent_solver = CISolver(
        states=singlet,
        nroots=2,
        core_orbitals=core,
        active_orbitals=active,
        ci_vector_guess=[
            [(Determinant("22a2b0"), 1.0)],
            [(Determinant("22b2a0"), 1.0)],
        ],
    )
    dependent = CI(dependent_solver)(rhf)
    with pytest.raises(ValueError, match="linearly dependent"):
        dependent.run()

    outside_basis_solver = CISolver(
        states=singlet,
        core_orbitals=core,
        active_orbitals=active,
        guess_dets=[Determinant("222a00")],
    )
    outside_basis = CI(outside_basis_solver)(rhf)
    with pytest.raises(ValueError, match="not in the CI space"):
        outside_basis.run()

    # one configuration has one singlet CSF, too few for three roots
    small_basis_solver = CISolver(
        states=singlet,
        nroots=3,
        core_orbitals=core,
        active_orbitals=active,
        guess_dets=[Determinant("222200")],
    )
    small_basis = CI(small_basis_solver)(rhf)
    with pytest.raises(ValueError, match="fewer than the 3 roots"):
        small_basis.run()


def test_rel_ci_guesses():
    """User guesses for RelCISolver, with complex coefficients."""
    rhf = _hf_rhf()
    # one upcaster for every solver, so that they all see the same spinor phases
    spinors = SpinorUpcaster(apply_random_phase=True)(rhf)

    # the ground state and the six components of the pi -> sigma* triplet pair
    ref_solver = RelCISolver(
        nel=10,
        nroots=7,
        core_orbitals=2,
        active_orbitals=12,
    )
    ref = CI(ref_solver)(spinors)
    ref.run()
    assert ref.E_ci[0] == approx(-100.019788438077)

    ndet = ref_solver.sub_solvers[0].ndet
    restart_solver = RelCISolver(
        nel=10,
        nroots=7,
        core_orbitals=2,
        active_orbitals=12,
        ci_vector_guess=ref_solver.get_top_determinants(n=ndet),
    )
    restart = CI(restart_solver)(spinors)
    restart.run()
    assert restart.E_ci == approx(ref.E_ci)
    assert restart_solver.sub_solvers[0].eigensolver.iter <= 2

    # truncated guesses are not normalized
    truncated_solver = RelCISolver(
        nel=10,
        nroots=7,
        core_orbitals=2,
        active_orbitals=12,
        ci_vector_guess=ref_solver.get_top_determinants(n=10),
    )
    truncated = CI(truncated_solver)(spinors)
    truncated.run()
    assert truncated.E_ci == approx(ref.E_ci)

    # determinants are spinor occupations, and coefficients may be complex
    guesses = ref_solver.get_top_determinants(n=10)
    guesses[0] = [
        (Determinant("aaaaaaaa0000"), 1.0),
        (Determinant("aa00aaaaaa00"), 0.05j),
    ]
    mixed_solver = RelCISolver(
        nel=10,
        nroots=7,
        core_orbitals=2,
        active_orbitals=12,
        ci_vector_guess=guesses,
    )
    mixed = CI(mixed_solver)(spinors)
    mixed.run()
    assert mixed.E_ci == approx(ref.E_ci)
