# Forte2 Option Naming Migration Plan

## Goal

Standardize user-facing option names to reduce ambiguity, improve discoverability, and keep behavior explicit across SCF, CI/MCSCF, DSRG, system setup, and orbital tooling.

This plan is backward-compatible: existing names remain usable during a deprecation window.

## Scope

- Constructor/dataclass options in:
  - `forte2/system/system.py`
  - `forte2/scf/*.py`
  - `forte2/base_classes/active_space_solver.py`
  - `forte2/ci/ci.py`
  - `forte2/mcopt/mc_optimizer.py`
  - `forte2/dsrg/*.py`
  - `forte2/orbitals/avas.py`
  - `forte2/orbitals/aset.py`

## Naming Conventions (Canonical)

1. Tolerances:
   - Use explicit units/meaning in name.
   - Prefer `*_rtol` and `*_atol` when relative/absolute is relevant.
   - Prefer `energy_tol`, `density_tol`, `gradient_tol`, `residual_tol` over abbreviated `econv/dconv/gconv/rconv`.

2. Booleans:
   - Prefer `use_*` or `enable_*` for feature toggles.
   - Prefer `raise_on_*` for exception behavior.
   - Avoid mixed prefixes like `do_*`, `die_if_*`, and bare adjectives in the same API surface.

3. Modes/enums:
   - Prefer descriptive values.
   - Keep short aliases accepted for compatibility (`sf`, `so`, `hz`, `kh`) but document descriptive forms as canonical.

4. Orbitals / active-space options:
   - Use consistent adjective order:
     - `frozen_core_orbitals`
     - `frozen_virtual_orbitals`
     - `frozen_active_orbitals` (not `active_frozen_orbitals`)

## Proposed Rename Map

## System / Orthogonalization

| Current | Proposed canonical | Notes |
|---|---|---|
| `linear_dep_trigger` | `lindep_rtol` | Trigger for deciding lindep behavior from overlap spectrum. |
| `ortho_thresh` | `orth_rtol` | Relative orthogonalization threshold. |
| `cholesky_tei` | `use_cholesky_tei` | Boolean naming consistency. |
| `x2c_type` values `sf`, `so` | `x2c_mode` values `scalar`, `spin_orbit` | Keep `sf`/`so` accepted as aliases. |
| `snso_type` | `snso_mode` | Enum-like mode naming consistency. |

## SCFBase and SCF methods

| Current | Proposed canonical | Notes |
|---|---|---|
| `do_diis` | `use_diis` | Boolean consistency. |
| `econv` | `energy_tol` | Explicit quantity. |
| `dconv` | `density_tol` | Explicit quantity. |
| `guess_type` | `initial_guess` | More explicit API (`sap`, `hcore`). |
| `level_shift_thresh` | `level_shift_disable_energy_tol` | Clarifies branch behavior in SCF loop. |
| `die_if_not_converged` | `raise_on_nonconvergence` | Explicit exception semantics. |
| `ms` | `spin_projection` | Clearer for external users; keep `ms` alias. |
| `ms_guess` (GHF) | `spin_projection_guess` | Align with `spin_projection`. |
| `guess_mix` (UHF/GHF) | `break_spin_symmetry_guess` | Describes purpose, not mechanism. |
| `alpha_beta_mix` (GHF) | `mix_alpha_beta_guess` | Imperative style and clarity. |
| `break_complex_symmetry` (GHF) | `break_time_reversal_guess` | Physics-consistent wording. |
| `j_adapt` | `use_j_adapted_basis` | Boolean and explicit target. |

## ActiveSpace / CI / MCSCF

| Current | Proposed canonical | Notes |
|---|---|---|
| `final_orbital` | `final_orbital_basis` | Values: `original`, `semicanonical`. |
| `ci_algorithm` values `hz`, `kh` | `ci_solver` values `harrison_zarrabian`, `knowles_handy`, `exact`, `sparse` | Keep short aliases accepted. |
| `active_frozen_orbitals` | `frozen_active_orbitals` | Make naming parallel with core/virtual variants. |
| `do_transition_dipole` | `compute_transition_dipole` | Prefer verb that implies action. |
| `ci_maxiter` | `ci_max_iterations` | Optional readability upgrade. |
| `ci_econv` | `ci_energy_tol` | Consistent tolerance naming. |
| `ci_rconv` | `ci_residual_tol` | Explicit residual meaning. |
| `ci_guess_per_root` | `ci_guesses_per_root` | Plural noun consistency. |
| `ci_ndets_per_guess` | `ci_dets_per_guess` | Shorter and clearer. |
| `ci_collapse_per_root` | `ci_vectors_kept_per_root` | Describe effect. |
| `ci_basis_per_root` | `ci_subspace_per_root` | More standard Davidson terminology. |
| `ci_energy_shift` | `ci_target_shift` | Clarifies role as root-targeting shift. |

## DSRG

| Current | Proposed canonical | Notes |
|---|---|---|
| `flow_param` | `flow_parameter` | Eliminate abbreviation. |
| `relax_reference` (`bool|int|str`) | `reference_relaxation_mode` + `reference_relaxation_iterations` | Split overloaded option into explicit mode and count. |
| `relax_maxiter` | `reference_relaxation_max_iterations` | Full semantic clarity. |
| `relax_tol` | `reference_relaxation_energy_tol` | Explicit tolerance type. |

## AVAS / ASET

| Current | Proposed canonical | Notes |
|---|---|---|
| `selection_method` (AVAS) | `selection_mode` | Mode terminology consistency. |
| `sigma` (AVAS) | `cumulative_overlap_target` | Avoid overloaded Greek-symbol naming. |
| `cutoff` (AVAS) | `overlap_eigenvalue_cutoff` | Clarifies what is cut off. |
| `evals_threshold` (AVAS) | `eigenvalue_zero_tol` | Explicit use. |
| `cutoff_method` (ASET) | `selection_mode` | Align with AVAS naming. |
| `num_A_occ`, `num_A_vir` (ASET) | `num_fragment_occ`, `num_fragment_vir` | Remove single-letter fragment ambiguity. |
| `semicanonicalize_active` | `use_semicanonical_active` | Boolean naming consistency. |
| `semicanonicalize_frozen` | `use_semicanonical_frozen` | Boolean naming consistency. |

## Compatibility and Deprecation Strategy

## Phase 0 (immediately, same release)

- Implement canonical names and keep old names as aliases.
- If both old and new forms are provided:
  - Raise `ValueError` if values disagree.
  - Prefer canonical name if equal.
- Emit `DeprecationWarning` on old-name usage.
- Add a one-line migration hint in warning text.

## Phase 1 (next minor release)

- Old names still accepted but warnings become more visible (e.g., `FutureWarning`).
- Docs and examples use only canonical names.
- Add migration guide section to docs and release notes.

## Phase 2 (next major release)

- Remove old aliases.
- Keep a dedicated error message for one cycle if old keys are detected via kwargs:
  - `"Option 'econv' was removed; use 'energy_tol'."`

## Implementation Plan

1. Add alias parsing helpers:
   - Create shared utility in `forte2/utils/`:
     - normalize constructor kwargs
     - detect conflicting alias usage
     - emit deprecation warnings

2. Apply migration module-by-module:
   - `System` and SCF first (largest user surface).
   - ActiveSpace/CI/MC next.
   - DSRG and orbitals tooling last.

3. Update docstrings and defaults:
   - Canonical names in signatures/docs.
   - Mention aliases only in a short compatibility note.

4. Update tests:
   - Add canonical-name tests.
   - Add alias-compatibility tests.
   - Add conflicting-alias tests.

5. Update examples/notebooks:
   - Replace old names with canonical names.
   - Keep one migration example in docs.

## Test Requirements for This Migration

- For each renamed option:
  - canonical name accepted
  - old alias accepted with warning
  - both provided and equal -> allowed
  - both provided and conflicting -> error

- Add centralized migration tests by subsystem:
  - `tests/system/test_option_aliases.py`
  - `tests/scf/test_option_aliases.py`
  - `tests/mcopt/test_option_aliases.py`
  - `tests/ci/test_option_aliases.py`
  - `tests/dsrg/test_option_aliases.py`

## Risks and Mitigations

- Risk: silent behavior changes if alias precedence is unclear.
  - Mitigation: strict conflict detection and explicit precedence rules.

- Risk: docs drift during transition.
  - Mitigation: canonical-only docs policy after Phase 0.

- Risk: external scripts rely on old options.
  - Mitigation: deprecation window with clear warnings and migration table.

## Suggested Adoption Order (Practical)

1. Tolerance naming (`econv/dconv/...`) and boolean style (`do_/die_if_`) in SCF + MCOptimizer.
2. `x2c_mode` / `snso_mode` descriptive enum aliases.
3. Active-space naming consistency (`frozen_active_orbitals`, `final_orbital_basis`).
4. DSRG relaxation option split (`reference_relaxation_mode` + iterations).
5. AVAS/ASET cleanup (`sigma`, `cutoff`, `num_A_*`).

## Out of Scope

- Internal variable names that are not user-facing.
- Changing physics defaults or numerical behavior as part of renaming.
- Removing all abbreviations in low-level/internal helper APIs in this pass.
