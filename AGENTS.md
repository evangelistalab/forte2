# AGENTS.md

## Purpose
This file gives coding agents repo-specific guidance for working in `forte2`.

## Project At A Glance
- `forte2` is a hybrid Python/C++ quantum chemistry codebase.
- Python package code lives in `forte2/`.
- Performance-critical C++ is exposed via nanobind as `forte2._forte2`.
- Tests are organized by subsystem in `tests/` (for example: `tests/scf`, `tests/ci`, `tests/integrals`).
- Docs source is in `docs/source/` and built with Sphinx.

## Environment And Build
- Preferred environment setup:
  - `conda env create -f environment.yml`
  - `conda activate forte2` (CI uses environment name `forte`)
- Editable developer install:
  - `pip install --no-build-isolation -ve .`
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
   - `python -m nanobind.stubgen -m forte2._forte2 -O forte2 -r`
5. Commit updated `.pyi` files in `forte2/_forte2/` with the code changes.

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
