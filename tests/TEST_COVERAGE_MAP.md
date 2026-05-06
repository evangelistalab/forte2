# Forte2 Test Coverage Map and Expansion Needs

This document summarizes what is currently tested in Forte2 and what should be added next to improve coverage. It is based on the current `main` branch test suite under `tests/`.

## How to read this

- `Yes`: explicit tests exist for this functionality.
- `Partial`: covered indirectly, conditionally, or only in one variant.
- `No`: no clear direct test currently exists.

## 1. Current Coverage by Subsystem

| Subsystem | What is currently tested | Representative tests |
|---|---|---|
| `scf` | RHF/ROHF/UHF/CUHF/GHF energies, spin states, symmetry labels, Hubbard-model HF, level-shift usage, linear-dependence handling, `sf`/`so` X2C workflows | `tests/scf/test_rhf.py`, `tests/scf/test_uhf.py`, `tests/scf/test_rohf.py`, `tests/scf/test_cuhf.py`, `tests/scf/test_ghf.py`, `tests/scf/test_x2c1e.py`, `tests/scf/test_lindep.py` |
| `mcopt` | CASSCF/GASSCF, state-averaging, AVAS-driven spaces, frozen core/virtual, active-frozen GAS orbitals, noncontiguous spaces, cholesky path, relativistic CASSCF/GASSCF | `tests/mcopt/test_casscf_simple.py`, `tests/mcopt/test_gasscf_simple.py`, `tests/mcopt/test_sa_casscf_n2.py`, `tests/mcopt/test_rel_casscf.py`, `tests/mcopt/test_rel_gasscf.py` |
| `ci` | CI solvers (RHF/ROHF/relativistic), GASCI/Rel-GASCI, RDMs/cumulants, transition dipoles, determinant algebra, exact/HZ/KH solver paths | `tests/ci/test_ci_rhf.py`, `tests/ci/test_ci_rohf.py`, `tests/ci/test_rel_ci.py`, `tests/ci/test_ci_exact_diagonalization.py`, `tests/ci/test_rel_gasci.py` |
| `dsrg` | Nonrelativistic and relativistic DSRG-MRPT2 workflows | `tests/dsrg/test_dsrg_mrpt2.py`, `tests/dsrg/test_rel_dsrg_mrpt2.py` |
| `integrals` | One-/two-/three-center integrals, DF paths, scalar one-electron pieces, basis/shell behavior, libcint-backed routes (conditional) | `tests/integrals/test_one_electron.py`, `tests/integrals/test_two_electron.py`, `tests/integrals/test_df.py`, `tests/integrals/test_libcint.py` |
| `jkbuilder` | AO and MO JK builds, restricted and spinorbital MO integrals, complex variants | `tests/jkbuilder/test_jkbuilder.py`, `tests/jkbuilder/test_mointegrals.py` |
| `orbitals` | AVAS/ASET/IAO/IBO, semicanonicalization, cube generation, GHF-equivalent orbital workflows | `tests/orbitals/test_avas.py`, `tests/orbitals/test_aset.py`, `tests/orbitals/test_iao.py`, `tests/orbitals/test_ibo.py`, `tests/orbitals/test_semican.py`, `tests/orbitals/test_cube.py` |
| `sparse` | Sparse operators/states, commutators/products, exponential actions and derivatives | `tests/sparse/test_sparse_operator.py`, `tests/sparse/test_sparse_exp.py`, `tests/sparse/test_sq_operator.py` |
| `state` | State and state-average validation | `tests/state/test_state.py`, `tests/state/test_state_average_info.py` |
| `symmetry` | Point-group detection, MO irrep assignment checks, spherical-harmonic transforms, large Otterbein database checks | `tests/symmetry/test_pg_detect.py`, `tests/symmetry/test_sph_harm_utils.py`, `tests/symmetry/test_otterbein_sym_db.py` |
| `system` | Geometry parsing, basis assignment, units, ghost atoms, custom basis maps, core AO integral wrappers | `tests/system/test_system.py`, `tests/system/test_system_bse.py`, `tests/system/test_basis_utils.py` |
| `props` | Dipoles/quadrupoles/mulliken and mutual-correlation properties | `tests/props/test_props.py`, `tests/props/test_mutcorr.py` |
| `helpers` | Orthogonalization/invsqrt, Davidson-Liu, Cholesky helper, L-BFGS | `tests/helpers/test_ortho.py`, `tests/helpers/test_davidsonliu.py`, `tests/helpers/test_cholesky.py`, `tests/helpers/test_lbfgs.py` |
| `mp2` | Basic MP2 energy path | `tests/mp2/test_mp2.py` |

## 2. Hartree-Fock Coverage Grid

### 2.1 Nonrelativistic HF methods

| Functionality | RHF | ROHF | UHF | CUHF | GHF | Evidence |
|---|---|---|---|---|---|---|
| Basic energy regression | Yes | Yes | Yes | Yes | Yes | `tests/scf/test_rhf.py`, `tests/scf/test_rohf.py`, `tests/scf/test_uhf.py`, `tests/scf/test_cuhf.py`, `tests/scf/test_ghf.py` |
| Open-shell spin handling (`ms`, `<S^2>`) | No | Yes | Yes | Yes | Yes (via `ms_guess`) | `tests/scf/test_rohf.py`, `tests/scf/test_uhf.py`, `tests/scf/test_cuhf.py`, `tests/scf/test_ghf.py` |
| Parameter validation (incompatible charge/spin) | Partial | Yes | Yes | Yes | Partial | `tests/scf/test_rohf.py`, `tests/scf/test_uhf.py`, `tests/scf/test_cuhf.py` |
| Symmetry-resolved MO labels | Yes | No | No | No | No | `tests/scf/test_hf_sym.py` |
| Cholesky TEI route | Yes | No | No | No | No | `tests/scf/test_rhf.py::test_rhf_cholesky` |
| DIIS-related path exercised | Partial | Partial | Partial | Partial | Partial | many SCF tests run default DIIS; explicit `do_diis=False` in selected tests |
| Level shift feature | Yes | No | No | No | No | `tests/scf/test_rhf.py::test_rhf_level_shift` |
| Initial guess controls (`hcore`, `guess_mix`, etc.) | Partial | Partial | Yes | Partial | Yes | Hubbard tests (`guess_type='hcore'`), `test_coulson_fischer`, `test_equivalence_to_uhf` |
| Restart from prior MO coefficients (`C`) | Yes | No | No | No | Partial | `tests/scf/test_read_wfn.py`; `test_so_from_sf_water` sets initial `C` for GHF |
| Linear-dependence handling | Yes | No | No | No | Yes | `tests/scf/test_lindep.py` |
| Model-system (Hubbard) support | Yes | Yes | Yes | No | No | `tests/scf/test_hubbard_model.py` |

### 2.2 Relativistic HF (`X2C`) coverage

| Functionality | RHF (`sf-X2C`) | UHF (`sf-X2C`) | GHF (`so-X2C`) | Evidence |
|---|---|---|---|---|
| Scalar-relativistic X2C energies | Yes | Partial | No | `tests/scf/test_x2c1e.py::test_sfx2c1e` |
| Gaussian nuclear-charge variant in X2C | Yes | No | No | `tests/scf/test_x2c1e.py::test_sfx2c1e_with_gaussian_charges` |
| SO-X2C energies/spinor SCF | No | No | Yes | `tests/scf/test_x2c1e.py::test_sox2c1e_water` |
| SNSO scaling modes | No | No | Partial | `dcb` and `row-dependent` covered in `test_x2c1e.py` |
| j-adapted spinor basis path | No | No | Yes | `tests/scf/test_ghf.py::test_j_adapted_ghf` |
| SF-to-SO coefficient transfer workflow | No | Yes | Yes | `tests/scf/test_x2c1e.py::test_so_from_sf_water` |
| X2C + severe linear dependence | Partial (currently skipped) | No | No | `test_lindep_sfx2c1e` is marked `skip` |

## 3. CASSCF/GASSCF Coverage Grid

| Functionality | CASSCF (NR) | GASSCF (NR) | SA-CASSCF (NR) | Rel-CASSCF (2c) | Rel-GASSCF (2c) | Evidence |
|---|---|---|---|---|---|---|
| Basic optimization/energy regression | Yes | Yes | Yes | Yes | Yes | `tests/mcopt/test_casscf_simple.py`, `test_gasscf_simple.py`, `test_sa_casscf_*.py`, `test_rel_casscf.py`, `test_rel_gasscf.py` |
| AVAS-driven active spaces | Yes | Partial | Yes | Yes | No | `test_casscf_avas_cyclopropene.py`, `test_sa_casscf_*`, `test_rel_casscf.py` |
| Frozen core/virtual orbitals | Yes | Partial (active-frozen only) | No | Yes | Partial (active-frozen) | `test_casscf_frozen_co.py`, `test_rel_casscf.py`, `test_gasscf_active_frozen.py`, `test_rel_gasscf.py` |
| Noncontiguous spaces | Yes | Yes | No | No | No | `test_casscf_noncontiguous_spaces_n2.py`, selected GAS tests |
| Active-frozen orbitals | No | Yes | No | No | Yes | `test_gasscf_active_frozen.py`, `test_rel_gasscf.py` |
| Cholesky + symmetry path | Yes | No | No | No | No | `test_casscf_cholesky_n2.py` |
| State averaging, same multiplicity | Partial | Partial | Yes | Partial | Partial | `test_sa_casscf_n2.py`, selected GAS tests |
| State averaging, mixed multiplicity/weights | No | Yes | Yes | No | No | `test_sa_casscf_n2.py`, `test_gasscf_h2o_core` |
| Transition dipole moments | No | No | Yes | No | No | `test_sa_casscf_hf.py`, `test_sa_casscf_transition_dipole_c2.py` |
| Relativistic `so-X2C` support | No | No | No | Yes | Yes | `test_rel_casscf.py`, `test_rel_gasscf.py` |
| Heavy-element relativistic cases | No | No | No | Yes | Partial | `test_rel_casscf_na_ghf`, `test_rel_ci_br` |
| Equivalence checks (nonrel vs rel, CAS vs GAS) | Partial | Yes | Partial | Yes | Yes | `test_gasscf_equiv_to_casscf.py`, `test_rel_casscf_hf_equivalence_to_nonrel`, `test_rel_gasscf_equivalence_to_nonrel` |

## 4. Coverage Gaps and Recommended Expansion

### 4.1 High-priority HF gaps

- Add explicit tests for `SCFBase` validation branches:
  - negative `level_shift`
  - tuple `level_shift` misuse for non-UHF
  - tuple length errors for UHF
  - `SO-X2C` rejection for non-GHF methods
- Add non-convergence behavior tests:
  - `die_if_not_converged=True` raises
  - `die_if_not_converged=False` warning path
- Add explicit invalid `guess_type` tests (currently only valid `minao`/`hcore` paths are exercised).
- Add UHF/CUHF/GHF level-shift behavior tests (only RHF level shift is directly regression-tested).
- Add ROHF/CUHF/GHF symmetry-path tests (currently symmetry assignment is heavily RHF-centric).
- Add UHF/ROHF/CUHF restart tests from user-provided orbitals (`C`) analogous to RHF restart.
- Add deterministic test coverage for all SNSO options (`boettger`, `dc`, `dcb`, `row-dependent`) and invalid `snso_type`.
- Replace or stabilize currently skipped X2C linear-dependence test (`test_lindep_sfx2c1e`) to keep this edge case active in CI.

### 4.2 High-priority CASSCF/GASSCF gaps

- Add `ActiveSpaceSolver` argument-validation tests:
  - conflicting `mo_space` + orbital-list inputs
  - missing MO-space source when parent cannot provide it
- Add `MCOptimizer` validation tests:
  - unsorted `active_frozen_orbitals`
  - out-of-active-space `active_frozen_orbitals`
  - invalid `final_orbital` / `ci_algorithm` combinations
- Add `MCOptimizer` branch tests for:
  - `die_if_not_converged` true/false paths
  - explicit DIIS parameter behavior (`diis_start`, `diis_nvec`, `diis_min`)
  - `final_orbital="original"` in MC optimization (not only CI/semi tests)
- Expand relativistic MC tests:
  - transition dipole in 2c state-averaged runs
  - multi-GAS with and without `freeze_inter_gas_rots`
  - heavier-element 2c GAS workflows beyond current small set

### 4.3 Broader suite gaps

- `tests/block2`, `tests/orbopt`, and `tests/thc` currently have no active `.py` tests in-tree (only `__pycache__`/tmp artifacts); either restore tests or remove stale directories/artifacts.
- Add direct unit tests for utility/support modules that are currently mostly indirectly covered:
  - `forte2/scf/scf_utils.py` (mixing/perturbation helpers)
  - `forte2/x2c/x2c.py` internal helper branches (currently mostly end-to-end tested)
  - `forte2/utils/mutual_correlation_plot.py`
  - `forte2/mods_manager.py`

## 5. Practical Notes

- Some coverage is conditional (`BSE_AVAILABLE`, `LIBCINT_AVAILABLE`) or marked `slow`; CI matrix decisions can hide regressions in those branches.
- At least one important edge test is currently skipped (`test_lindep_sfx2c1e`), which leaves a known fragile X2C+lindep region without active regression protection.

## 6. Test Implementation Checklist

This section is an implementation-oriented queue for expanding coverage.

### 6.1 Quick Wins (high value, low effort)

| ID | Suggested test file | What to add | Expected assertion |
|---|---|---|---|
| QW-SCF-01 | `tests/scf/test_scf_base_validation.py` | Negative `level_shift` for RHF/UHF | `ValueError` from `SCFBase.__call__` |
| QW-SCF-02 | `tests/scf/test_scf_base_validation.py` | Tuple `level_shift` on RHF/ROHF/GHF | `ValueError` ("Tuple level_shift is only valid for UHF.") |
| QW-SCF-03 | `tests/scf/test_scf_base_validation.py` | Bad tuple length for UHF level shift | `ValueError` ("length 2") |
| QW-SCF-04 | `tests/scf/test_scf_base_validation.py` | `x2c_type="so"` with RHF/UHF/ROHF/CUHF | `ValueError` ("SO-X2C is only available for GHF") |
| QW-SCF-05 | `tests/scf/test_guess_validation.py` | Invalid SCF `guess_type` (e.g., `"foobar"`) | `RuntimeError` from `_initial_guess` |
| QW-SCF-06 | `tests/scf/test_level_shift_non_rhf.py` | UHF/GHF/CUHF level-shift usage | Energy converges and differs from no-shift early iterations |
| QW-MC-01 | `tests/mcopt/test_mcopt_validation.py` | `active_frozen_orbitals` unsorted | `AssertionError` ("must be sorted") |
| QW-MC-02 | `tests/mcopt/test_mcopt_validation.py` | `active_frozen_orbitals` outside active space | `ValueError` with missing indices |
| QW-MC-03 | `tests/mcopt/test_active_space_solver_validation.py` | Conflicting `mo_space` and orbital-list args | `ValueError` from `_make_mo_space` |
| QW-MC-04 | `tests/mcopt/test_active_space_solver_validation.py` | Missing MO-space source from parent method | `ValueError` about MO space provisioning |
| QW-X2C-01 | `tests/scf/test_x2c1e_snso_modes.py` | Deterministic coverage of SNSO `"boettger"` and `"dc"` | Finite converged energies and expected ordering/tolerance checks |
| QW-X2C-02 | `tests/scf/test_x2c1e_snso_modes.py` | Invalid SNSO keyword | `ValueError` |

### 6.2 Medium Tasks (branch coverage and robustness)

| ID | Suggested test file | What to add | Expected assertion |
|---|---|---|---|
| MD-SCF-01 | `tests/scf/test_scf_nonconvergence.py` | Force SCF non-convergence with tiny `maxiter` and test both `die_if_not_converged` branches | Raises `RuntimeError` when `True`; returns with warning path when `False` |
| MD-SCF-02 | `tests/scf/test_restart_wfn.py` | Restart tests for UHF/ROHF/CUHF/GHF from previous `C` | Restart reaches same final energy as fresh run |
| MD-SCF-03 | `tests/scf/test_hf_sym_open_shell.py` | Symmetry-label checks for UHF/ROHF/CUHF/GHF | Irrep labels are populated and stable under rotation/orientation cases |
| MD-X2C-01 | `tests/scf/test_x2c1e_lindep.py` | Replace skipped `test_lindep_sfx2c1e` with stabilized geometry/basis/tolerance | Deterministic energy and expected `nmo` reduction |
| MD-MC-01 | `tests/mcopt/test_mcopt_diis_controls.py` | Explicit DIIS controls (`diis_start`, `diis_nvec`, `diis_min`) | Converged energy invariant within tolerance across settings |
| MD-MC-02 | `tests/mcopt/test_mcopt_final_orbital.py` | `final_orbital="original"` in MCSCF path | Distinct final orbitals from semicanonical path, CI energy consistency |
| MD-MC-03 | `tests/mcopt/test_rel_sa_transition_dipole.py` | 2c state-averaged transition dipole | Nonzero/expected transition moments and oscillator strengths |
| MD-MC-04 | `tests/mcopt/test_rel_gasscf_intergas.py` | `freeze_inter_gas_rots` true/false in 2c GAS | Both branches converge and produce distinct orbital-rotation behavior |
| MD-UTIL-01 | `tests/scf/test_scf_utils.py` | Unit tests for `guess_mix`, `guess_mix_ghf`, `alpha_beta_mix`, `break_complex_conjugation_symmetry` | Orthogonality and expected structure changes preserved |

### 6.3 Heavy Tasks (performance/infra-sensitive)

| ID | Suggested test file | What to add | Expected assertion |
|---|---|---|---|
| HV-X2C-01 | `tests/scf/test_x2c_regression_heavy.py` | Heavy-element `so-X2C` regression set across SNSO modes | Stable splittings/energies within tight windows |
| HV-MC-01 | `tests/mcopt/test_rel_gasscf_heavy.py` | Larger heavy-element 2c GASSCF workflows | Convergence and reproducible state energies |
| HV-THC-01 | `tests/thc/test_*.py` restoration | Reintroduce THC test modules (currently only pycache remnants) | Active tests run and validate THC factorization paths |
| HV-BLOCK2-01 | `tests/block2/test_*.py` restoration | Reintroduce block2 integration tests | Active tests run with expected fallback/skip behavior |

## 7. Suggested Execution Order

1. Complete all `QW-*` tasks first (they mostly validate existing branches).
2. Add `MD-SCF-01`, `MD-MC-01`, and `MD-X2C-01` next to protect unstable control-flow branches.
3. Expand relativistic MCSCF with `MD-MC-03` and `MD-MC-04`.
4. Schedule `HV-*` work only after CI-runtime budgeting and dependency matrix decisions.

## 8. Definition of Done for Coverage Expansion

- Every new branch-focused test includes at least one `raises` or behavior-differentiating assertion.
- New relativistic tests run in a dedicated CI lane or are clearly marked (e.g., `slow`, dependency-gated).
- At least one previously skipped critical path (`X2C + lindep`) is converted to an active regression test.
- `tests/block2`, `tests/orbopt`, and `tests/thc` either contain active tests or are intentionally removed to avoid false expectations.
