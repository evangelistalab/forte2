# AGENTS.md

## Purpose
This file gives coding agents repo-specific guidance for working in `forte2`.

## Project At A Glance
- `forte2` is a hybrid Python/C++ quantum chemistry codebase.
- Python package code lives in `forte2/`.
- Performance-critical C++ is exposed via nanobind as `forte2.lib`.
- Tests are organized by subsystem in `tests/` (for example: `tests/scf`, `tests/ci`, `tests/integrals`).
- Docs source is in `docs/source/` and built with Sphinx.

## Architecture

### The composition pattern (the core mental model)
Nearly every method is a `@dataclass` built in three stages: **construct → `__call__(upstream)` → `run()`**.

```python
rhf  = forte2.RHF(charge=0)(system)          # __call__ takes a System
avas = forte2.AVAS(subspace=["N(2p)"])(rhf)  # __call__ takes the SCF (the "parent_method")
mc   = forte2.MCOptimizer(ci_solver)(avas)   # __call__ takes AVAS
pt   = forte2.DSRG_MRPT2(s=0.5)(mc)
pt.run()                                       # lazily runs the whole chain
```

- **Constructor args** are all method options, validated in `__post_init__`.
- **`__call__(upstream)`** stores the upstream as `self.parent_method` (or `self.system` for SCF),
  validates its type, copies shared state, and returns `self`. It does not compute.
- **`run()`** is lazy and recursive: it first runs `self.parent_method.run()` if `not
  parent_method.executed`, then does the work. Calling `run()` on the last object executes the entire
  pipeline; the `executed` flag prevents recomputation.

Every method derives from `Method` (`forte2/base_classes/method.py`), which declares the data-flow
contract between links in the chain as three attributes set in `__post_init__`:
- `requires` — names this method needs the parent to supply (checked against the parent's `provides`).
- `provides` — names this method supplies downstream (e.g. `{"system", "mos", "mo_space"}`).
- `requires_attrs` — a **dict** of parent attributes, `{name: required_value}`, with `None` meaning
  "must exist" (e.g. `{"two_component": True}`). Because it is keyed by name, subclasses compose with
  `|=` and override an inherited requirement by re-stating its key. Never assign it with `=`: that
  silently drops requirements set by a base class.

`_register_parent_method` enforces all three at bind time (`__call__`), so incompatible pipelines fail
at construction rather than mid-run. MO coefficients travel as an `MO` value object (`base_classes/mo.py`),
reached as `self.mos.C[0]` / `self.mos.irrep_indices[0]`.

#### Solvers and drivers

Active-space methods come in pairs, and the distinction is load-bearing.

A **solver** (`CISolver`, `RelCISolver`, `SelectedCISolver`, `RelSelectedCISolver`, all on
`base_classes/ci_base.py::CIBase`) answers one question: solve in the *current* orbital basis with
the *current* integrals. Its `run()` is idempotent and safe to call in a loop, and it never touches
the orbitals. `CIBase` owns everything representation-agnostic — the `_startup`/`run` skeletons,
sub-solver fan-out, state-averaged RDMs and cumulants, transition properties, and the machinery
behind the `final_orbitals` rotation. Concrete solvers supply two class-level hooks:
`_integrals_cls` (`RestrictedMOIntegrals` vs `SpinorbitalIntegrals`) and `_ss_solver_cls` (the
per-state single-state solver).

A **driver** (`CI` and `MCOptimizer`, both on
`base_classes/active_space_driver.py::ActiveSpaceDriver`) is handed a solver and finishes a
calculation with it: solve, rotate to the requested final orbitals once, report. Its `run()` is the
chain entry point and is not loop-safe. The solver is the first argument --
`CI(CISolver(State(...), active_orbitals=[...]), final_orbitals="semicanonical")` -- so the
active-space options live on the solver, and the driver carries only what belongs to finishing a
one-shot calculation: `final_orbitals`, `do_transition_dipole` and `die_if_not_converged`, the last
two of which it pushes onto the solver at bind time alongside `log_level`.

`CI` is **solver agnostic**: one driver class takes any `CIBase`/`RelCIBase`, so there is no
`RelCI`/`SelectedCI`/`RelSelectedCI`. Anything a particular method needs to print differently is a
hook on the *solver*, since that is what knows its own quantities: `_print_energy_summary` (selected
CI prints variational and PT2-corrected tables, keyed off the `_energy_summary_label` class
attribute) and `_transition_property_energies`.

Keeping these apart is what lets a downstream method re-invoke `parent.ci_solver.run()` in a loop —
MCSCF macro-iterations, DSRG reference relaxation — without re-running the finishing logic. A
downstream method takes a *driver* as its parent (DSRG asserts this) and reaches the solver through
`.ci_solver`. **A driver is not a solver**: it forwards the four RDM accessors (`make_rdm`,
`make_cumulant`, `make_average_rdm`, `make_average_cumulant`) and copies the energies (`E`, `E_ci`,
`E_avg`) plus `mo_space`, `mos` and `dtype`. Everything else the solver exposes is reached through
`.ci_solver`, so reach for that rather than adding a forwarder.

Two chain-specific behaviors to watch:
- **MCSCF re-binds its `ci_solver`.** `MCOptimizer(ci_solver)(parent)` re-invokes the solver against
  `parent` in `_startup`, then alternates orbital optimization (L-BFGS) with `ci_solver.run()`.
- **DSRG consumes a reference, not a basis.** `DSRGBase` takes the active-space integral triple
  `E` (frozen-core energy), `H` (one-electron), `V` (antisymmetrized two-electron) plus cumulants
  from `parent.ci_solver`, and semicanonicalizes internally in `get_integrals()`, so it imposes no
  `final_orbitals` requirement on its parent.

Relativistic (two-component) variants are subclasses that set `dtype = complex` and
`two_component = True` (e.g. `RelActiveSpaceSolver`, `RelCISolver`); non-rel uses `float`.
`RelCIBase(RelActiveSpaceSolver, CIBase)` inherits the shared orchestration rather than re-binding
methods by hand. **Base order matters**: `dataclasses` collects fields in reverse-MRO order and
`CIBase` still carries an undefaulted `states` from `ActiveSpaceSolver`, so listing `CIBase` first
would clobber the `states = None` default that `nel`-based construction relies on.

The `final_orbitals` option (`"original"` / `"semicanonical"` / `"natural"`) is a driver option,
declared per driver because the default legitimately differs (`"original"` for the CI drivers,
`"semicanonical"` for `MCOptimizer`) and validated through `forte2/orbitals/final_orbitals.py`.

### Rebuilding a chain at a new geometry
`forte2/base_classes/rebuild.py` reconstructs or rebinds an entire method chain against a displaced
`System`, and is what `GeometryOptimizer` and `FDGradient` (`forte2/gradients/fd_gradient.py`) are
built on:
- `rebuild_method_chain(method, new_system)` walks root-to-leaf and calls `type(stage)(**kwargs)` per
  stage, with `kwargs` read straight from `dataclasses.fields()`. That means **every init field must
  survive being fed back into the constructor unchanged** — a field that a method overwrites with a
  derived value in `__post_init__`/`_startup` (e.g. an `MOSpace` built from `active_orbitals`) breaks
  this silently. The fix is to give the derived value its own `init=False` field rather than mutating
  the input field, as `ActiveSpaceSolver` does: the raw constructor arg is `mo_space_override`, and the
  resolved space consumed downstream is the separate, non-init `mo_space`.
- `rebind_method_chain(method, new_system)` reuses the same objects instead: `stage.reset()` then
  re-`__call__` per stage. The base `Method.reset()` only flips `executed`/`converged`; any state a
  method mutates *in place* across repeated `run()` calls (a list `.append()`ed to then swapped for an
  array, a cache dict, ...) has to be rebuilt in that method's own `_startup()`, or a second `run()` on
  the same object corrupts silently or crashes outright — see `DSRGBase._startup()` resetting
  `relax_eigvals_history` for exactly this reason.
- `project_scf_guess(source_method, method)` and `forte2/orbitals/orbital_overlap.py`
  (`mo_overlap`/`project_orbitals`/`project_occupied_orbitals`) project occupied orbitals from a
  converged method onto a rebuilt chain's SCF root as an initial guess, including across two-component
  (GHF) chains. When checking whether source and target agree on representation, compare
  `source_method.mos.spinorbital` (frozen when its `MO` was built) against the target's own
  `two_component` attribute — never `System.two_component` directly: `RHF`/`UHF`/`GHF.__call__` all
  mutate that flag in place on the (possibly shared) `System` object, so it can read stale for a method
  that finished running earlier against the same `System`.

### C++ / Python boundary
- `forte2.lib` is the compiled nanobind module. C++ sources live **inline inside the package**
  (e.g. `forte2/integrals/*.cc`, `forte2/ci/*.cc`, `forte2/sci/*.cc`, `forte2/sparse/*.cc`), not in a
  separate `src/` tree. `forte2/CMakeLists.txt` lists every `.cc` compiled into `lib`.
- `forte2/api/*_api.cc` are the nanobind binding layers (one per subsystem); `api/forte2_api.cc` is the
  `NB_MODULE` root that calls each `export_*`. Stubs live in `forte2/lib/*.pyi`.
- **C++** holds performance kernels: integral evaluation, CI string/sigma builds and RDMs, selected-CI
  (HCI), determinant/Slater-rules machinery, sparse operators/states. **Python** holds all orchestration:
  SCF loop, MCSCF optimizer, DSRG, AVAS, gradients, geometry optimization, properties, J/K building.

### Threading model
All parallel C++ code determines its thread count through `forte2/helpers/parallel.h::get_num_threads()`
— never accept a thread-count constructor/setter argument on a class or function; call
`get_num_threads()` directly, or use the `parallel_for*` helpers in the same header, which already do.
Precedence: `FORTE_NUM_THREADS_OVERRIDE` if set, otherwise the smallest of
`std::thread::hardware_concurrency()` (logical CPUs, including SMT/hyperthreads — *not* physical
cores), the process's CPU affinity mask, `OMP_NUM_THREADS`, `OMP_THREAD_LIMIT`, and
`SLURM_CPUS_PER_TASK` (whichever are set). The effective count is printed once at `import forte2`

### Integrals
All two-electron integrals are **density-fitted or Cholesky-decomposed** — there is no conventional
4-index path, which is why a `System` requires an `auxiliary_basis_set`. Two backends sit behind one
interface: **libint2** (always) and **libcint** (`USE_LIBCINT=ON` by default). `coulomb_3c` returns
`(P|mn)` row-major regardless of backend (`P` = auxiliary; `m,n` = orbital basis). See
`forte2/integrals/integrals.py` (backend selection), `forte2/jkbuilder/jkbuilder.py` (J/K from the
3-center B tensor), and `forte2/jkbuilder/mointegrals.py` (MO-space `E`/`H`/`V` for CI/DSRG).

### Subsystem map (beyond the obvious)
- `ci` — full / spin-adapted / GAS CI (`CISolver`→`CI`, `RelCISolver`→`RelCI`).
- `sci` — selected CI / heat-bath CI (`SelectedCISolver`), usable as an active-space solver in MCSCF.
- `determinant` / `sparse` — determinant & bit-string representations, Slater rules, `SparseOperator`/`SparseState`.
- `mcopt` — MCSCF/CASSCF/GASSCF optimizer, exposed as `forte2.MCOptimizer` (not `MCSCF`).
- `dsrg` — DSRG-MRPT2 and its relativistic variant.
- `x2c` — exact two-component relativistic transform (`sf` scalar, `so` spin-orbit); gated by `System.x2c_type`.
- `orbitals` — AVAS, ASET embedding, IAO/IBO, semicanonicalizer, cube generation, `SpinorUpcaster`.
- `props`, `gradients`, `optimize` — 1e-properties/populations; DF-based analytic gradients plus
  `FDGradient`, which differentiates *any* rebuildable method's energy by finite differences (see
  "Rebuilding a chain at a new geometry" above); `GeometryOptimizer` drives either.
- `state`, `symmetry` — `State`/`RelState`/`MOSpace`/state-averaging; point-group MO symmetry detection.

## Environment And Build
- Preferred environment setup:
  - `conda env create -f environment.yml`
  - `conda activate forte2` (CI uses environment name `forte`)
- If conda is not yet installed, install Miniconda first. On a fresh conda you must accept the
  default-channel Terms of Service before `conda env create` will run, even though
  `environment.yml` only uses conda-forge:
  - `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`
  - `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r`
- Editable developer install:
  - `pip install --no-build-isolation -ve .`
- Editable installs do NOT auto-rebuild the C++ extension. After editing any `.cc`/`.h`, re-run the
  editable install to recompile (or add `-Ceditable.rebuild=true` to rebuild on import). Editing only
  Python requires no rebuild.
- Build system details:
  - `scikit-build-core` + `CMake` + `nanobind`
  - C++ standard is C++20
  - `Libint2`, `Eigen3`, BLAS/LAPACK are required by CMake
- `USE_LIBCINT` is enabled by default through `pyproject.toml`; override if needed:
  - `pip install . --config-settings=cmake.define.USE_LIBCINT=OFF`

## Test Commands
- Fast local run:
  - `pytest -m "not slow"`
- Full run:
  - `pytest -v --cov --cov-branch --cov-report=xml`
- Subsystem-focused run examples:
  - `pytest tests/scf -q`
  - `pytest tests/ci -q`
  - `pytest tests/integrals -q`
- Some tests are conditionally skipped based on optional dependencies (`BSE_AVAILABLE`, `LIBCINT_AVAILABLE`).

## Documentation Commands
- Install docs dependencies:
  - `pip install -r docs/requirements.txt`
- Build docs:
  - `make -C docs html`

## Coding Conventions
- Follow the existing functional-composition pattern for methods:
  - instantiate -> `__call__(upstream/system)` -> `run()`
- Keep argument validation in initialization (`__post_init__`) or method entry points.
- Python style:
  - Black formatting
  - NumPy-style docstrings
- C++ style:
  - `.clang-format` settings (4-space indent, 100-column limit)
- Avoid editing generated artifacts:
  - `build/`
  - `docs/build/`

## C++/Binding Change Checklist
When adding or changing bound C++ functionality:
1. Update C++ implementation/header files in `forte2/`.
2. Add/update nanobind exposure in `forte2/api/*_api.cc`.
3. Update `forte2/CMakeLists.txt` if adding new source files.
4. Regenerate Python stubs:
   - `python -m nanobind.stubgen -m forte2.lib -O forte2 -r`
5. Commit updated `.pyi` files in `forte2/lib/` with the code changes.

## Test Expectations For Changes
- Add or update tests in the matching subsystem folder under `tests/`.
- Prefer deterministic numerical assertions and existing comparison helpers (for example `forte2.helpers.comparisons.approx`).
- Mark expensive tests with `@pytest.mark.slow`.
- Gate optional-dependency tests with `pytest.mark.skipif(...)`.

## Practical Workflow
1. Make minimal, focused code changes.
2. Run targeted subsystem tests first.
3. Run `pytest -m "not slow"` before finishing.
4. If public API or behavior changes, update docs in `docs/source/`.

## Repo-Specific Notes
- CI runs on Ubuntu and macOS with Python 3.12.
- Importing `forte2` triggers `load_mods()` from `forte2/mods_manager.py`; keep tests independent of user-local mods.
