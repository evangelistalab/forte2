# CASSCF wavefunction overlap: handoff

## Overview

This note describes the state of the `ci_ovlp` branch, which adds the
machinery to compute the overlap between two complete active space
self-consistent field (CASSCF) wavefunctions built from independent,
non-orthogonal orbital sets — for example, two states from separate CASSCF
optimizations, or the same state at two different geometries.

The implementation follows the nonunitary orbital biorthogonalization scheme
of Malmqvist [*Int. J. Quantum Chem.* **30**, 479 (1986)], specialized to
CASSCF wavefunctions. Given two orbital sets that aren't orthonormal to each
other, it constructs a pair of biorthonormal bases, transforms both
configuration interaction (CI) vectors into that basis, and takes a plain dot
product.

## Branch structure

Two branches carry this work:

- `sigma-reuse` (based on `origin/main`, independently useful on its own):
  adds reusable sigma-build primitives to `CISigmaBuilder` and
  `RelCISigmaBuilder`. Cherry-picked into `ci_ovlp` as commit `f15b885`.
- `ci_ovlp` (based on `findiff`): the CASSCF overlap feature itself, built on
  top of the cherry-picked `sigma-reuse` commit.

## Architecture

### C++ sigma-build reuse primitives

Commit `f15b885`.

`forte2/ci/ci_sigma_builder.{h,cc}` and `forte2/ci/rel_ci_sigma_builder.{h,cc}`
gain two public methods, both thin wrappers splitting the existing private
`H1_*`/`H2_*` sigma contributions:

- `sigma_one_electron(basis, sigma)`: applies the scalar and one-electron part
  of the Hamiltonian only.
- `sigma_two_electron(basis, sigma)`: applies the two-electron part only.

`Hamiltonian(basis, sigma)` — the existing fused method — is unchanged, and
`sigma_one_electron(basis, s1) + sigma_two_electron(basis, s2)` always equals
it, for `s1 + s2`. The existing `set_Hamiltonian(E, H, V)` method is now bound
to Python for both classes; it already reused its scratch buffers internally
(`std::vector::resize` doesn't reallocate when the number of orbitals is
unchanged), so binding it lets a caller swap in a different Hamiltonian
without constructing a new builder. Neither method assumes `H` is symmetric —
both algorithms (Knowles-Handy and Harrison-Zarrabian) loop over the full
range of orbital index pairs. This matters because the module described below
feeds in a one-body orbital-rotation generator, not a physical Hamiltonian,
and that generator generally isn't symmetric.

Tests: `tests/ci/test_sigma_builder_reuse.py` (10 tests, covering both
algorithms, symmetric and non-symmetric `H`, and the relativistic path).

### Python overlap module

`forte2/orbitals/wavefunction_overlap.py` is organized into four parts.

**Orbital-space construction.**
`biorthogonalize_casscf_orbitals(S, ndocc, nactv) -> (C_XA, C_YB)` implements
Malmqvist's Appendix "pseudo-corresponding orbitals" recipe via two singular
value decompositions (SVDs). Given the mixed inactive-plus-active molecular
orbital (MO) overlap `S` between orbital sets X and Y, it returns
transformation matrices such that the new orbital sets are biorthonormal:
`(C_XA)^† S C_YB = 1`. `C_XA` is block upper-triangular in the
(inactive, active) ordering, so new inactive orbitals never pick up active
character — the structural requirement for the transformation to stay closed
within the CASSCF CI expansion.

**Robust real-logarithm helpers.** A one-body orbital-rotation generator `t`
must satisfy `exp(t) = M` for the relevant active-active block `M`. Computing
`t` with a plain `scipy.linalg.logm(M)` is unreliable for the real-valued
("direct") backend — see [Numerical pitfall](#numerical-pitfall-real-logarithm-of-an-orthogonal-matrix)
below. `_real_orthogonal_logm`, `_real_symmetric_logm`,
`_robust_orthogonal_steps`, `_robust_real_logm`, and
`_apply_orbital0_reflection` implement the fix.

**Ground-truth backend.** `transform_ci_vector_sparse_ops(ci_strings, C,
t_actv, docc_scale, maxk, screen_thresh)` applies `exp(-t_actv)` to a CI
vector using `forte2.lib.sparse_ops.SparseExp`'s Taylor-series operator
exponential — the generic, hash-map-based sparse-operator infrastructure.
Correct but not optimized for large active spaces; kept as a correctness
reference.

**Efficient backend (default).** `transform_ci_vector_direct(ci_strings, C,
t_actv, docc_scale, tol, max_taylor_order, max_squarings, scale_threshold)`
does the same computation using `CISigmaBuilder.sigma_one_electron` — the
same machinery every CASSCF or CI run already uses for its own sigma-vector
builds — via scaling-and-squaring around a Taylor series. `_apply_generator_steps`
wraps this to consume the step sequence `_robust_real_logm` returns.

**Dispatcher.** `casscf_wavefunction_overlap(ci_strings_1, C1,
C_docc_actv_1, system_1, ci_strings_2, C2, C_docc_actv_2, system_2, ndocc,
nactv, backend="direct"|"sparse_ops", **backend_kwargs) -> complex` runs the
full pipeline: mixed MO overlap, biorthogonalization, per-side CI-vector
transform, final dot product.

### Numerical pitfall: real logarithm of an orthogonal matrix

`C_XA`'s active-active block is always exactly an SVD orthogonal factor.
`scipy.linalg.logm` computes eigenvalue-wise logarithms using the principal
branch of the complex logarithm, which is discontinuous on the negative real
axis. A proper rotation with an eigenvalue near -1 (a rotation by close to
π) comes back with a spurious, large imaginary part, even though a real
antisymmetric logarithm always exists for a proper rotation. Worse, an
improper rotation (determinant -1, a reflection composed with a rotation) has
*no* real logarithm at all, since `det(exp(A)) = exp(tr A) > 0` for any real
matrix `A`.

The complex-valued ground-truth backend never hits this, because
`SparseOperator`/`SparseExp` handle a complex generator natively. The
real-valued direct backend can't hide it, and testing showed it isn't a rare
edge case: two independently converged CASSCF wavefunctions comparing
themselves (or two states related by a plain orbital rotation) reliably
produce these matrices, since the underlying `MCOptimizer` orbital
optimization is float-order-sensitive enough that repeated runs occasionally
land in a different numerical regime.

`_robust_real_logm(M)` decomposes `M` into a list of steps —
`("reflect",)` and/or `("generator", t)` — applied to the CI vector in order:

1. Try `scipy.linalg.logm(M)` directly. If the result is already
   negligibly complex, use it (the common, fast path).
2. If `M` is orthogonal, use `_real_orthogonal_logm`: a real Schur
   decomposition, reading each 2x2 rotation block's angle off directly with
   `arctan2` (branch-free over the full range), pairing up any unpaired -1
   eigenvalues into synthetic π-rotation blocks. For an improper `M`, first
   factor `M = F @ R` (`F` flips the sign of active orbital 0 — an exact,
   closed-form CI-vector operation needing no exponential at all) and log the
   proper part `R`.
3. Otherwise, use the polar decomposition `M = Q @ P` (`Q` orthogonal, `P`
   symmetric positive-definite), handle `Q` as in step 2, and log `P` by
   eigendecomposition (always real for a positive-definite matrix, so this
   step never hits a branch cut).

**Composition order matters and is easy to get backward.** Orbitals transform
by right multiplication, `φ_new = φ_old @ M`. For a product `M = M1 @ M2`,
the corresponding CI-vector operators apply in the *reverse* order:
`ρ(M) = ρ(M2) · ρ(M1)` — an anti-homomorphism, not the naive
`ρ(M1) · ρ(M2)`. Getting this backward doesn't crash; it silently produces a
plausible-looking but wrong overlap value (0.908 instead of 1.0 was the
symptom that caught it during development). If you extend this decomposition
further, rederive the order from this relation rather than guessing by
pattern-matching.

## Validation

- `tests/orbitals/test_wavefunction_overlap.py` (11 tests):
  biorthogonalization math on synthetic random systems; a real LiH CASSCF
  wavefunction against itself; the same physical state under a random active
  orbital rotation (verified against an independently-built reference using
  `SparseExp.apply_op` directly); two fully independent `MCOptimizer` runs at
  different bond lengths, cross-checked against a from-scratch brute-force
  Löwdin determinant-overlap sum; the `direct` and `sparse_ops` backends
  cross-checked against each other, including a deliberately large-angle
  case exercising the scaling-and-squaring path.
- Every wavefunction-level test is parametrized over both backends.
- A physical sanity check (not part of the automated suite, but worth
  rerunning if you touch this code): scanning `⟨Ψ(Re)|Ψ(R)⟩` for LiH from its
  equilibrium bond length out to 6 bohr reproduces a smooth decay from 1,
  followed by a sign flip and partial recovery near R = 5 bohr — the expected
  signature of LiH's known ionic-covalent avoided crossing. The `direct` and
  `sparse_ops` backends agree to six decimal places on this curve (they can
  differ in overall sign very close to the crossing, where the wavefunction's
  phase is inherently ambiguous).
- Full `pytest -m "not slow"`: 812 passed, 3 skipped, no regressions, run
  five consecutive times to build confidence against the MCSCF non-determinism
  described above.

## Known limitations and explicitly out-of-scope items

- No relativistic or complex-CASSCF support yet. `RelCISigmaBuilder` has the
  same `sigma_one_electron`/`set_Hamiltonian` primitives (see the C++ section
  above), so the same approach should extend cleanly, but the Python module
  only handles real CI vectors and orbitals.
- `_robust_real_logm`'s general (polar-decomposition) branch handles every
  case observed in testing, but no case has come up requiring a *non-normal*
  fallback beyond that. If `_robust_real_logm` ever raises in practice, that's
  the next thing to generalize.
- The new C++ primitives (`sigma_one_electron`, `sigma_two_electron`,
  `set_Hamiltonian`) are general-purpose infrastructure, not wired up
  anywhere else yet — for example, computing a full pairwise overlap matrix
  across many CASSCF states by reusing one builder across states is a natural
  follow-up, not yet implemented.
- Skipping the O(active-orbital-count⁴) two-electron-integral tensor setup
  inside `set_Hamiltonian` when only one-electron sigma builds are needed was
  considered and deferred — it would require splitting the Knowles-Handy
  correction term's coupling to `V` out of the one-electron setup, adding
  state-tracking complexity for a cost that's microseconds at the active-space
  sizes (fewer than about 30 orbitals) this module targets.

## Running the tests

```
pytest tests/ci/test_sigma_builder_reuse.py -v
pytest tests/orbitals/test_wavefunction_overlap.py -v
pytest -m "not slow"
```

The C++ changes require a rebuild before the Python tests can pick them up:

```
pip install --no-build-isolation -ve .
python -m nanobind.stubgen -m forte2.lib -O forte2 -r
```
