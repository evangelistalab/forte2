# SA-CASSCF Gradient Refactoring Notes

These are possible follow-up refactors identified while reviewing the
SA-CASSCF gradient branch. None of them have been applied.

## Higher priority

1. **Share a response workspace across gradient stages.**

   `solve_state_specific_response`, `compute_omega`, and the relaxed-density
   construction repeatedly build target and averaged RDMs, CI-response RDMs,
   Fock matrices, and DF transforms. A small internal data carrier could build
   these once and pass them between stages. It should remain a plain data
   object used by free functions rather than introducing another mixin or base
   class.

2. **Cache determinant-space reference vectors during GMRES.**

   `_compute_ci_response_rdms` repeatedly converts the fixed reference CSF
   vectors to determinant vectors for each matrix-vector product. Preparing
   these vectors once in the response workspace should reduce iterative-solver
   overhead.

3. **Narrow the public-looking response API.**

   Several non-underscored Hessian-vector and b-vector functions in
   `mc_optimizer_response.py` are used only by tests. Possible approaches are
   to make them explicitly private, define `__all__`, or collapse each
   wrapper/worker pair using optional precomputed intermediates.

4. **Consolidate response-intermediate construction.**

   The orbital, coupled, CI-orbital, and orbital-CI builders independently
   form the core Fock matrix, its MO transformation, and the active-space DF
   tensor. A lightweight common builder could remove duplication while still
   allowing callers to request only the arrays they need.

5. **Split the relaxed-density-weight builder into mathematical components.**

   `_build_sa_casscf_relaxed_density_weights` currently handles RDM
   construction, orbital relaxation, metric inversion, cumulant weights, and
   AO back-transformation. A few focused private kernels would make the
   equations, memory lifetimes, and test boundaries clearer.

6. **Investigate sharing DF metric infrastructure.**

   `build_metric_inverted_three_center` rebuilds the three-center integrals and
   Coulomb metric. The integral or Fock layer could expose a reusable
   inverse-metric operation or cached factor if profiling shows this cost is
   material. The existing \(M^{-1/2}J\) tensor cannot directly replace the
   required \(M^{-1}J\) tensor.

## Structural cleanup

7. **Centralize root validation and selection.**

   `_resolve_casscf_gradient_root` and the response module perform overlapping
   type and range checks. One shared private helper could define root semantics
   consistently.

8. **Make geometry-optimization root selection method-agnostic.**

   `GeometryOptimizer` currently knows about `E_ci` and
   `gradient(root=...)`. A generic energy/gradient accessor or a root-selecting
   method adapter would avoid coupling the optimizer to `MCOptimizer`
   internals and would align with the finite-difference accessor design.

9. **Remove duplicated root dispatch in `_GeometryObjective.ensure`.**

   Its cached and newly evaluated branches contain essentially the same
   gradient-selection logic. Bound energy and gradient accessors would
   simplify both paths.

10. **Reorganize the large SA-CASSCF test module.**

    The response-kernel tests and end-to-end gradient tests could live in
    separate files. Related system builders and finite-difference cases could
    also be parameterized. Shared `MCOptimizer` fixtures may reduce runtime,
    provided mutable optimizer state can be reset safely and tests remain
    independent.

11. **Decouple the timing benchmark from response internals.**

    The benchmark imports `_build_coupled_response_intermediates` to estimate
    memory. A stable diagnostic helper or an explicit analytical memory
    estimate would make it less sensitive to internal refactoring. Adding a
    public API solely for the benchmark may not be worthwhile.

## PR-size cleanup

12. **Drop unrelated diff noise.**

    Formatting-only changes in `mc_optimizer.py` and the one-line ASET
    docstring correction are unrelated to gradients. Restoring those lines to
    match `main` would make the PR smaller and easier to review.
