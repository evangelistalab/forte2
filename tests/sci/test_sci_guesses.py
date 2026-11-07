from forte2 import CI, RHF, SelectedCISolver, SpinorUpcaster, State, System
from forte2.sci import RelSelectedCISolver
from forte2.base_classes.params import DavidsonLiuParams, SelectedCIParams
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


def test_sci_guesses():
    """Selected CI started from CI vectors, whose determinants make up the initial space,
    or from guess determinants."""
    rhf = _hf_rhf()
    singlet = State(nel=10, multiplicity=1, ms=0.0)
    core = [0]
    active = [1, 2, 3, 4, 5, 6]
    sci_params = SelectedCIParams(var_threshold=1e-12, pt2_threshold=0.0)

    # the singlet ground state and pi -> sigma* pair, at the CASCI limit
    ref_solver = SelectedCISolver(
        states=singlet,
        nroots=3,
        core_orbitals=core,
        active_orbitals=active,
        sci_params=sci_params,
    )
    ref = CI(ref_solver)(rhf)
    ref.run()
    assert ref.E_ci == approx([-100.019788438077, -99.687583941414, -99.687583941414])

    ndet = ref_solver.sub_solvers[0].ndet
    restart_solver = SelectedCISolver(
        states=singlet,
        nroots=3,
        core_orbitals=core,
        active_orbitals=active,
        sci_params=sci_params,
        ci_vector_guess=ref_solver.get_top_determinants(n=ndet),
    )
    restart = CI(restart_solver)(rhf)
    restart.run()
    assert restart_solver.sub_solvers[0].guess_energies == approx(ref.E_ci)
    assert restart.E_ci == approx(ref.E_ci)

    # truncated guesses are not normalized, and seed only part of the final space
    truncated_solver = SelectedCISolver(
        states=singlet,
        nroots=3,
        core_orbitals=core,
        active_orbitals=active,
        sci_params=sci_params,
        ci_vector_guess=ref_solver.get_top_determinants(n=10),
    )
    truncated = CI(truncated_solver)(rhf)
    truncated.run()
    assert truncated.E_ci == approx(ref.E_ci)

    # one guess vector per root, from one candidate configuration
    single_guess = DavidsonLiuParams(guess_per_root=1, ndets_per_guess=1)
    # the RHF determinant alone gives the RHF energy
    basis_solver = SelectedCISolver(
        states=singlet,
        core_orbitals=core,
        active_orbitals=active,
        sci_params=sci_params,
        guess_dets=[Determinant("222200")],
    )
    CI(basis_solver)(rhf).run()
    assert basis_solver.sub_solvers[0].guess_energies[0] == approx(rhf.E)
    # the pi_y^2 -> sigma*^2 determinant is higher in energy, so narrowing drops it
    narrowed_solver = SelectedCISolver(
        states=singlet,
        core_orbitals=core,
        active_orbitals=active,
        sci_params=sci_params,
        davidson_liu_params=single_guess,
        guess_dets=[Determinant("222200"), Determinant("222020")],
    )
    CI(narrowed_solver)(rhf).run()
    assert narrowed_solver.sub_solvers[0].guess_energies[0] == approx(rhf.E)
    # pinned, it stays, and its coupling lowers the guess energy
    pinned_solver = SelectedCISolver(
        states=singlet,
        core_orbitals=core,
        active_orbitals=active,
        sci_params=sci_params,
        davidson_liu_params=single_guess,
        guess_dets=[Determinant("222200")],
        pinned_guess_dets=[Determinant("222020")],
    )
    pinned = CI(pinned_solver)(rhf)
    pinned.run()
    assert pinned_solver.sub_solvers[0].guess_energies[0] < rhf.E - 1e-5
    assert pinned.E_ci[0] == approx(ref.E_ci[0])


def test_rel_sci_guesses():
    """Relativistic selected CI restarted from guesses with complex coefficients."""
    rhf = _hf_rhf()
    # one upcaster for every solver, so that they all see the same spinor phases
    spinors = SpinorUpcaster(apply_random_phase=True)(rhf)

    # the default guess windows give fewer guess determinants than roots
    wide_sci_params = SelectedCIParams(
        var_threshold=1e-12,
        pt2_threshold=0.0,
        guess_occ_window=4,
        guess_vir_window=4,
    )
    # the ground state and the six components of the pi -> sigma* triplet pair
    ref_solver = RelSelectedCISolver(
        nel=10,
        nroots=7,
        core_orbitals=2,
        active_orbitals=12,
        sci_params=wide_sci_params,
    )
    ref = CI(ref_solver)(spinors)
    ref.run()
    assert ref.E_ci == approx([-100.019788438077] + [-99.705264582881] * 6)

    # the initial space is the guess determinants alone, so the windows don't matter
    ndet = ref_solver.sub_solvers[0].ndet
    sci_params = SelectedCIParams(var_threshold=1e-12, pt2_threshold=0.0)
    restart_solver = RelSelectedCISolver(
        nel=10,
        nroots=7,
        core_orbitals=2,
        active_orbitals=12,
        sci_params=sci_params,
        ci_vector_guess=ref_solver.get_top_determinants(n=ndet),
    )
    restart = CI(restart_solver)(spinors)
    restart.run()
    assert restart_solver.sub_solvers[0].guess_energies == approx(ref.E_ci)
    assert restart.E_ci == approx(ref.E_ci)

    # truncated guesses are not normalized, and seed only part of the final space
    truncated_solver = RelSelectedCISolver(
        nel=10,
        nroots=7,
        core_orbitals=2,
        active_orbitals=12,
        sci_params=sci_params,
        ci_vector_guess=ref_solver.get_top_determinants(n=10),
    )
    truncated = CI(truncated_solver)(spinors)
    truncated.run()
    assert truncated.E_ci == approx(ref.E_ci)
