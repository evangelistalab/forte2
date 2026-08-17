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

State flows down the chain via mixins in `forte2/base_classes/mixins.py`, each with a
`copy_from_upstream` classmethod: `SystemMixin` (`system`), `MOsMixin` (MO coefficients `C`, irrep
info), `MOSpaceMixin` (`mo_space` = core/active/virtual partition). `base_classes/active_space_solver.py`
and `ci_base.py` are the bases for CI-type solvers (active-space resolution, state-averaged RDMs/cumulants).

Two chain-specific behaviors to watch:
- **MCSCF re-binds its `ci_solver`.** `MCOptimizer(ci_solver)(parent)` re-invokes the solver against
  `parent` in `_startup`, then alternates orbital optimization (L-BFGS) with `ci_solver.run()`.
- **DSRG requires semicanonical orbitals.** `DSRG_MRPT2.__call__` asserts
  `parent.final_orbitals == "semicanonical"` and consumes the active-space integral triple `E` (frozen-core
  energy), `H` (one-electron), `V` (antisymmetrized two-electron) plus cumulants from the upstream solver.
  Use `forte2/orbitals/semicanonicalizer.py` if a reference isn't already semicanonical.

Relativistic (two-component) variants are subclasses that set `dtype = complex` and
`two_component = True` (e.g. `RelActiveSpaceSolver`, `RelCISolver`); non-rel uses `float`.

### C++ / Python boundary
- `forte2.lib` is the compiled nanobind module. C++ sources live **inline inside the package**
  (e.g. `forte2/integrals/*.cc`, `forte2/ci/*.cc`, `forte2/sci/*.cc`, `forte2/sparse/*.cc`), not in a
  separate `src/` tree. `forte2/CMakeLists.txt` lists every `.cc` compiled into `lib`.
- `forte2/api/*_api.cc` are the nanobind binding layers (one per subsystem); `api/forte2_api.cc` is the
  `NB_MODULE` root that calls each `export_*`. Stubs live in `forte2/lib/*.pyi`.
- **C++** holds performance kernels: integral evaluation, CI string/sigma builds and RDMs, selected-CI
  (HCI), determinant/Slater-rules machinery, sparse operators/states. **Python** holds all orchestration:
  SCF loop, MCSCF optimizer, DSRG, AVAS, gradients, geometry optimization, properties, J/K building.

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
- `props`, `gradients`, `optimize` — 1e-properties/populations, DF-based analytic gradients, `GeometryOptimizer`.
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
- Parallel run (`pytest-xdist`):
  - `pytest -m "not slow" -n auto` spreads tests across all available CPU cores.
  - On a shared dev box (e.g. multiple agent worktrees/conda envs active at once), prefer a
    bounded worker count like `-n 4` instead of `-n auto` — `auto` will happily spawn one
    worker per core and can starve other concurrent sessions.
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
