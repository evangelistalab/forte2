Nuclear gradients
=================

Forte2 computes nuclear gradients in two ways: analytically, where an
implementation exists, and by finite differences of the energy, which works for
any method that can be rebuilt at a displaced geometry.

Both expose the same interface. A method's ``gradient()`` returns an array of
shape ``(natoms, 3)`` in Hartree/Bohr, ordered like ``system.atomic_positions``.

Analytic gradients
------------------

Density-fitted analytic gradients are available for RHF, UHF, GHF, and
state-specific CASSCF/GASSCF::

    rhf = forte2.RHF(charge=0)(system).run()
    g = rhf.gradient()

They require an auxiliary basis (there is no conventional four-index path) and
raise ``NotImplementedError`` for combinations that are not supported, rather
than silently returning something approximate.

Finite-difference gradients
---------------------------

:class:`forte2.FDGradient` attaches to any
upstream method and differentiates its energy::

    mc = forte2.MCOptimizer(ci_solver)(rhf)
    fd = forte2.FDGradient(step=1.0e-3)(mc)
    fd.run()
    g = fd.gradient()

Each displacement rebuilds the whole upstream chain at the displaced geometry
and reruns it, so the cost is ``npoints * 3 * natoms`` evaluations of the
upstream method -- 36 SCF calculations for a three-atom molecule with the
default four-point stencil. Use ``npoints=2`` to halve that, at the cost of
accuracy.

Because it provides ``gradient()``, it is interchangeable with an analytic
implementation, including as the driver of a geometry optimization::

    forte2.GeometryOptimizer(g_tol=1.0e-5)(
        forte2.FDGradient()(mc)
    ).run()

Choosing a step
~~~~~~~~~~~~~~~

The step :math:`h` (``step``, in Bohr) balances two errors:

- **Truncation error** grows with the step. A central stencil with :math:`n`
  points (``npoints``) is exact for polynomials up to degree :math:`n`, so it
  leaves an error of order :math:`h^n`.
- **Noise** grows as the step shrinks. Each energy carries an error
  :math:`\epsilon`, and the stencil divides differences of energies by
  :math:`h`, so the noise adds an error of order :math:`\epsilon / h`.

The total error is about :math:`C h^n + \epsilon / h`, where :math:`C` depends on
the :math:`(n+1)`-th derivative of the energy. It is smallest at
:math:`h \sim (\epsilon / C)^{1/(n+1)}`, where it is of order
:math:`\epsilon^{n/(n+1)}`. The noisier the energies, the larger the best step,
and a stencil with more points tolerates a larger step.

The energy error :math:`\epsilon` has two sources:

- **Rounding**, about :math:`10^{-15}` times the total energy, so it is larger
  for heavy elements.
- **Incomplete convergence.** If the energy is variational in all of its
  parameters, as for SCF and CASSCF, the convergence error enters the energy
  quadratically, and default thresholds keep it near the rounding level. If it
  isn't, as for a CI on fixed SCF orbitals or a perturbation theory, the SCF
  convergence error enters linearly and usually dominates.

The following table shows both regimes for the RHF gradient of
H\ :sub:`2`\ O/STO-3G, converged to ``e_tol=1e-12`` and ``d_tol=1e-10``:

.. list-table:: Maximum error (Eh/Bohr) against the analytic gradient
   :header-rows: 1

   * - ``npoints``
     - ``step=1e-5``
     - ``1e-4``
     - ``1e-3``
     - ``1e-2``
     - ``3e-2``
   * - 2
     - 5e-9
     - 6e-9
     - 6e-7
     - 6e-5
     - 6e-4
   * - 4
     - 7e-9
     - 7e-10
     - 6e-11
     - 2e-8
     - 1e-6
   * - 6
     - 9e-9
     - 8e-10
     - 7e-11
     - 9e-12
     - 2e-9

At the smallest step, all three stencils give the same error, and it grows as
the step shrinks: that's noise. At the largest steps, the error falls steeply
as ``npoints`` increases: that's truncation error. For a CI on fixed RHF orbitals
of the same molecule, with ``npoints=4`` and ``step=1e-3``, the error is
``2e-10`` with ``d_tol=1e-10`` but ``3e-7`` with the default ``d_tol=1e-6``.

Follow these recommendations:

- Start from the defaults, ``step=1e-3`` and ``npoints=4``. With well-converged
  energies, they give gradients accurate to about ``1e-10`` Eh/Bohr.
- If the energy isn't variational in the SCF orbitals, converge the SCF to
  ``d_tol=1e-10``. If you can't, increase the step to ``1e-2``.
- Keep the step at ``1e-4`` or above. Below that, noise dominates even for
  well-converged energies.
- To halve the cost, use ``npoints=2`` with ``step=1e-3``, which is accurate to
  about ``1e-6``. That is enough to drive a geometry optimization to the default
  ``g_tol=1e-4``.
- For the highest accuracy, use ``npoints=6`` with ``step=1e-2``.

To measure the actual error of a result, see `Checking the result`_.

Checking the result
~~~~~~~~~~~~~~~~~~~

An exact gradient of a translationally and rotationally invariant energy has
zero net force and zero net torque, so whatever remains measures the numerical
error directly. Both are reported and are available afterwards::

    fd.net_force     # sum of the gradient rows, shape (3,)
    fd.net_torque    # sum of r_A x g_A, shape (3,)

A residual above ``residual_tol`` (default ``1e-6`` Eh/Bohr) is reported as a
warning. This measurement is a far better guide than any estimate based on the
convergence threshold alone.

A separate warning fires when a displaced energy differs from the reference by
much more than the gradient implies, which usually means that displacement
converged to a different SCF solution or a different CI root. The difference
quotient then straddles a discontinuity and the result is meaningless rather
than merely noisy.

Multiple roots
~~~~~~~~~~~~~~

Methods that report several energies require ``root`` to select the one to
differentiate::

    fd = forte2.FDGradient(root=1)(ci_solver)

Omitting it raises rather than silently differentiating the lowest root.

Limitations
~~~~~~~~~~~

Displaced geometries are built with
:meth:`forte2.System.with_geometry`, so the system must be rebuildable:
``symmetry=False`` (symmetry detection reorients the molecule, which would
invalidate Cartesian displacements), a defined ``basis_set``, and not a
``ModelSystem``.

Numerical differentiation on its own
------------------------------------

The finite-difference machinery is independent of the chemistry and can be used
directly on any callable::

    from forte2.gradients import finite_difference

    finite_difference(f, x, step=1.0e-3, npoints=4)

``x`` may be a scalar or an array of any shape, and ``f`` may return a scalar or
an array; the derivative preserves the output shape. Pass ``components`` to
differentiate only selected entries of ``x``.
