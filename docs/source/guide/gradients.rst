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

The truncation error of an ``n``-point central stencil falls as ``step**n``,
while noise in the energy is amplified by ``1 / step``. The default
``step = 1e-3`` Bohr with a four-point stencil is a good starting point.

The noise term is much smaller than it first appears. Every displacement is
seeded with orbitals projected from the *reference* geometry, so the residual
convergence error is nearly identical at ``+h`` and ``-h`` and largely cancels
in the difference. Measured on H\ :sub:`2`\ O/STO-3G RHF, an upstream ``e_tol``
of ``1e-5`` still yields a gradient accurate to ``1.5e-8`` Eh/Bohr.

That cancellation depends on the initial guess being a function of the displaced
geometry alone, not of the order in which displacements are evaluated. Seeding
each displacement from the *previous* one instead costs about a factor of 700 in
accuracy at the same threshold, which is why the reference geometry is used even
though a neighbouring displacement would be a marginally better guess.

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

Orbital projection also applies to two-component (relativistic) chains, provided
the source and target share the same representation -- for example, a GHF root
projects cleanly onto a rebuilt GHF root at the displaced geometry. It falls back
to the default guess only when source and target disagree (e.g. a one-component
source projected onto a two-component target, or vice versa); the gradients
remain correct in that case, but each displacement takes more iterations.

Numerical differentiation on its own
------------------------------------

The finite-difference machinery is independent of the chemistry and can be used
directly on any callable::

    from forte2.gradients import finite_difference

    finite_difference(f, x, step=1.0e-3, npoints=4)

``x`` may be a scalar or an array of any shape, and ``f`` may return a scalar or
an array; the derivative preserves the output shape. Pass ``components`` to
differentiate only selected entries of ``x``.
