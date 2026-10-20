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
CASSCF/GASSCF::

    rhf = forte2.RHF(charge=0)(system).run()
    g = rhf.gradient()

They require an auxiliary basis (there is no conventional four-index path) and
raise ``NotImplementedError`` for combinations that are not supported, rather
than silently returning something approximate.

For a state-averaged CASSCF wavefunction, ``gradient(root=...)`` returns the
gradient of one root. This needs a one-component wavefunction and
``final_orbitals="original"`` on the ``MCOptimizer``.
``forte2.GeometryOptimizer(root=...)`` optimizes the geometry of that root.

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
``symmetry=False`` (symmetry detection reorients the molecule and moves atoms to
make it exactly symmetric, which would invalidate Cartesian displacements), a
defined ``basis_set``, and not a ``ModelSystem``.

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
