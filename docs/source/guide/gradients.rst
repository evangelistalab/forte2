Nuclear gradients and nonadiabatic couplings
============================================

Forte2 computes nuclear gradients in two ways: analytically, where an
implementation exists, and by finite differences of the energy, which works for
any method that can be rebuilt at a displaced geometry. Finite differences also
give the nonadiabatic couplings between the roots of a CI or MCSCF
wavefunction.

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

:class:`forte2.FiniteDifference` attaches to any upstream method and
differentiates its energy::

    mc = forte2.MCOptimizer(ci_solver)(rhf)
    fd = forte2.FiniteDifference(step=1.0e-3)(mc)
    g = fd.gradient()

Each displacement rebuilds the whole upstream chain at the displaced geometry
and reruns it, so the cost is ``npoints * 3 * natoms`` evaluations of the
upstream method -- 36 SCF calculations for a three-atom molecule with the
default four-point stencil. Use ``npoints=2`` to halve that, at the cost of
accuracy. To differentiate only some Cartesian components, pass them as
``(atom, xyz)`` pairs, and the other entries of the result are NaN::

    fd = forte2.FiniteDifference(components=[(0, 2), (1, 2)])(mc)

Because it provides ``gradient()``, it is interchangeable with an analytic
implementation, including as the driver of a geometry optimization::

    forte2.GeometryOptimizer(g_tol=1.0e-5)(
        forte2.FiniteDifference()(mc)
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
error directly. If every component was differentiated, both are reported for
each gradient and are available afterward::

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

To differentiate the energy of one root, pass ``root`` to ``gradient()``. One
sweep of displacements serves every root, so the gradients of all roots cost
the same as the gradient of one::

    fd = forte2.FiniteDifference()(mc)
    g_0 = fd.gradient(root=0)
    g_1 = fd.gradient(root=1)  # no new displacements

For ``CI`` and ``MCOptimizer``, ``root`` indexes ``E_ci``, and ``gradient()``
without a root differentiates the state-averaged energy ``E``. For a method that
reports one energy per root in ``E``, such as a ``CISolver``, ``root`` indexes
``E``, and omitting it raises an error rather than silently differentiating the
lowest root.

A root-resolved geometry optimization uses the same interface::

    forte2.GeometryOptimizer(root=1)(forte2.FiniteDifference()(mc)).run()

To differentiate an energy that is neither ``E`` nor a root energy, pass a
function that extracts it as ``energy_accessor``, for example
``lambda method: method.E_relaxed_ref`` for a relaxed DSRG reference.

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

Nonadiabatic couplings
----------------------

For a ``CI`` or ``MCOptimizer`` with a ``CISolver`` or ``RelCISolver`` and more
than one root, :class:`forte2.FiniteDifference` also computes the nonadiabatic
coupling :math:`\langle \Psi_\mathrm{bra} | \nabla_R \Psi_\mathrm{ket} \rangle`
in inverse Bohr::

    fd = forte2.FiniteDifference(compute_nac=True)(mc)
    d = fd.nonadiabatic_coupling(ket=1, bra=0)
    g_0 = fd.gradient(root=0)
    g_1 = fd.gradient(root=1)

The coupling is the derivative of the overlap between the reference bra and
the displaced ket, from :func:`forte2.orbitals.ci_overlap_matrix`. With
``compute_nac=True``, every sweep collects these overlaps along with the
energies, so the coupling and both gradients come from one sweep, and
``FiniteDifference`` checks that the upstream method supports couplings as soon
as you attach it. Without it, the first call to ``nonadiabatic_coupling()``
after a ``gradient()`` call runs a second sweep. To get the coupling multiplied
by :math:`E_\mathrm{ket} - E_\mathrm{bra}`, pass ``energy_gap_weighted=True``.

Phases and degenerate roots
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each calculation returns its roots with arbitrary phases: arbitrary signs in a
nonrelativistic calculation, and arbitrary complex phases in a two-component
one. Before differencing, ``FiniteDifference`` aligns every displaced root with
its reference root by choosing its phase so that the two overlap positively.
Roots closer in energy than ``degeneracy_tol`` (default ``1e-6`` Eh), such as
the two roots of a Kramers pair, form a degenerate manifold, and each displaced
manifold is rotated as a whole to match its reference manifold.

After alignment, the only arbitrary factor left is the phase of the reference
roots. It's the same for every Cartesian component, so the coupling vector is
determined up to one overall phase, and its magnitude is unique. Within a
degenerate manifold, the coupling isn't defined, and requesting it raises an
error. Between two manifolds, the individual couplings depend on how the
reference roots were chosen within each manifold, but the singular values of the
block of couplings don't.

Moving basis functions
~~~~~~~~~~~~~~~~~~~~~~

The basis functions move with the atoms, so the coupling includes their
contribution, known as the configuration state function (CSF) term, and isn't
translationally invariant: displacing either atom of a bond gives a different
coupling. It corresponds to analytic couplings without electron-translation
factors, such as PySCF's with ``use_etfs=False``. Finite differences of overlaps
can't give the translationally invariant coupling that electron-translation
factors produce.

Accuracy
~~~~~~~~

An overlap is first order in the convergence error of the wavefunction, while an
energy is second order, so couplings need tighter convergence than gradients.
Converge MCSCF orbitals to ``g_tol=1e-8`` or tighter; at that threshold, the
couplings of LiH at 4 Å reproduce analytic values to a few ``1e-6`` inverse
Bohr.

Each sweep that collects overlaps reports two diagnostics, which are also
available afterward:

* ``fd.anti_hermiticity_residual`` is the largest :math:`\lVert D + D^\dagger
  \rVert` over the differentiated components, where :math:`D` is the matrix of
  couplings between all roots. It vanishes for exact couplings.
* ``fd.min_overlap_singular_value`` is the smallest singular value of the
  overlap between a reference root or manifold and its displaced counterpart.
  It stays close to 1, and a warning fires below 0.9, which usually means a
  displaced root changed character or order.

Numerical differentiation on its own
------------------------------------

The finite-difference machinery is independent of the chemistry and can be used
directly on any callable::

    from forte2.gradients import finite_difference

    finite_difference(f, x, step=1.0e-3, npoints=4)

``x`` may be a scalar or an array of any shape, and ``f`` may return a real or
complex scalar or array; the derivative preserves the output shape and type. Pass ``components`` to
differentiate only selected entries of ``x``.
