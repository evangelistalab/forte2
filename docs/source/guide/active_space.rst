Active-space solvers
====================

Forte2's active-space solvers treat a set of active orbitals with a multiconfigurational
wavefunction, through configuration interaction (CI) or multiconfigurational self-consistent field
(MCSCF) calculations. Here, we cover how to run them, choose their states and active spaces,
tune the solvers, and control their initial guesses.

Solvers and drivers
-------------------

An active-space calculation combines two objects. A *solver* holds the states and the active space
and solves the CI problem in the current orbitals:

``CISolver``
    Complete active space CI (CASCI) and generalized active space CI (GASCI), in a spin-adapted (CSF)
    basis.

``SelectedCISolver``
    Selected CI (heat-bath CI variant) which grows a determinant space iteratively.

``RelCISolver`` and ``RelSelectedCISolver``
    The two-component counterparts, for relativistic references.

A *driver* takes a solver as its first argument and uses it to finish a calculation:

``CI``
    Solves the CI problem once, using the orbitals of the upstream method.

``MCOptimizer``
    Optimizes the orbitals together with the CI wavefunction, for CASSCF or GASSCF.

Both drivers accept any solver.

Run a CASCI calculation
-----------------------

To solve the CI problem in the Hartree-Fock orbitals, pass a solver to ``CI`` and attach it to the
Hartree-Fock calculation::

    import forte2

    system = forte2.System(
        xyz="H 0 0 0; F 0 0 2.0",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        unit="bohr",
    )
    rhf = forte2.RHF(charge=0)(system)

    ci_solver = forte2.CISolver(
        states=forte2.State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0],  # or 1
        active_orbitals=[1, 2, 3, 4, 5, 6],  # or 6
    )
    ci = forte2.CI(ci_solver)(rhf)
    ci.run()
    print(ci.E_ci)

Run a CASSCF calculation
------------------------

To optimize the orbitals as well, pass the solver to ``MCOptimizer`` instead::

    ci_solver = forte2.CISolver(
        states=forte2.State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
    )
    mc = forte2.MCOptimizer(ci_solver)(rhf)
    mc.run()
    print(mc.E)

``MCOptimizer`` alternates solving the CI problem and optimizing the orbitals until the energy and the orbital gradient converge
The same call runs GASSCF if the solver has a GAS active space, as well as state-averaged CAS/GASSCF,
which optimizes the orbitals for the weighted average energy, if the solver has several states or
roots.

``MCOptimizer`` takes these options:

``maxiter``
    The maximum number of macro-iterations, each a CI solve and an orbital step. The default is 50.

``e_tol`` and ``g_tol``
    The convergence thresholds for the energy and the orbital gradient. The defaults are 1e-8 and
    1e-7.

``micro_maxiter`` and ``max_rotation``
    The number of L-BFGS iterations per orbital step, 6 by default, and the largest orbital
    rotation in a step, 0.2 by default.

``active_frozen_orbitals``
    Active orbitals to keep fixed during the optimization.

``freeze_inter_gas_rots``
    If True, rotations between orbitals of different GASes are excluded. The default is False.

``die_if_not_converged``
    If True, the default, an unconverged optimization raises an error. If False, it logs a
    warning.

``MCOptimizer`` rotates the orbitals with unitary steps, so the orbitals it starts from must be
orthonormal in the current basis set. To start from orbitals of another geometry or basis set, map
them with ``forte2.orbitals.transfer_orbitals`` first. For analytic CASSCF gradients, see
:doc:`gradients`.

States and roots
----------------

A ``State`` sets the number of electrons, the spin, the symmetry, and the GAS occupations of the
wavefunction:

``nel``
    The total number of electrons, including those in core orbitals.

``multiplicity``
    The spin multiplicity, 2S + 1.

``ms``
    The spin projection. Non-relativistic CI works in a spin-adapted basis of configuration state
    functions (CSFs), so any ``ms`` allowed by the multiplicity gives the same energies.

``symmetry``
    The irrep of the wavefunction, as its index in Cotton order. The default, 0, is the totally
    symmetric irrep. If the ``System`` uses symmetry, the solver keeps only determinants of this
    irrep; see :doc:`symmetry`. Without symmetry, every orbital is in irrep 0, so leave the default.
    ``symmetry_label`` sets the name printed for the irrep.

``gas_min`` and ``gas_max``
    The minimum and maximum numbers of electrons in each GAS, in GAS order. A list shorter than
    the number of GASes restricts only the first GASes, and the default, an empty list, restricts
    none. See `Active spaces`_.

``nroots`` sets how many roots to compute. To average over several states, pass a list of states
and one ``nroots`` entry per state, and optionally ``weights``::

    singlet = forte2.State(nel=10, multiplicity=1, ms=0.0)
    triplet = forte2.State(nel=10, multiplicity=3, ms=0.0)
    ci_solver = forte2.CISolver(
        states=[singlet, triplet],
        nroots=[3, 2],
        weights=[[1, 1, 1], [1, 1]],
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
    )

Roots are numbered in *absolute root order*: the roots of the first state, then those of the
second, and so on. Here, roots 0-2 are singlets and roots 3-4 are triplets. The driver's ``E_ci``
lists the energies in that order, and ``E_avg`` holds their weighted average, which is what
``MCOptimizer`` minimizes.

Active spaces
-------------

The solver splits the orbitals into frozen core, core, active, and frozen virtual orbitals with
the ``frozen_core_orbitals``, ``core_orbitals``, ``active_orbitals``, and
``frozen_virtual_orbitals`` options. Orbitals you don't list are virtual. If the method before the
driver already provides an active space, for example ``AVAS``, you can leave these options out.

For a GAS calculation, pass ``active_orbitals`` as one list per GAS, and set occupation limits on
the state with ``gas_min`` and ``gas_max``. For example, to allow at most one hole in the first
orbital::

    ci_solver = forte2.CISolver(
        states=forte2.State(nel=10, multiplicity=1, ms=0.0, gas_min=[1]),
        core_orbitals=[0],
        active_orbitals=[[1], [2, 3, 4, 5, 6]],
    )

Pass this solver to ``CI`` for GASCI, or to ``MCOptimizer`` for GASSCF.

Solver options
--------------

Two parameter classes from ``forte2.base_classes`` tune the solver. ``CIParams`` sets how the CI
problem is solved:

``ci_algorithm``
    ``"hz"`` (Harrison-Zarrabian, the default) or ``"kh"`` (Knowles-Handy) for the iterative
    sigma build, or ``"exact"`` to build and diagonalize the full Hamiltonian, which suits only
    small spaces.

``ci_builder_memory``
    The memory limit of the sigma build, in MB.

``energy_shift``
    A total energy, in hartree. If you set it, the solver converges the roots nearest this energy
    instead of the lowest roots: the automatic guess picks candidates near it, and the
    Davidson-Liu solver keeps the roots nearest it.

``DavidsonLiuParams`` controls the Davidson-Liu solver: the convergence thresholds ``e_tol`` and
``r_tol``, the iteration limit ``maxiter``, the subspace sizes ``basis_per_root`` and
``collapse_per_root``, and the automatic guess (see `Initial guesses`_)::

    from forte2.base_classes import CIParams, DavidsonLiuParams

    ci_solver = forte2.CISolver(
        states=forte2.State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
        ci_params=CIParams(ci_algorithm="kh"),
        davidson_liu_params=DavidsonLiuParams(r_tol=1e-8, maxiter=200),
    )

By default, the ``CI`` driver raises an error if the solver doesn't converge. To get a warning
instead, pass ``die_if_not_converged=False`` to the driver.

Determinant strings
-------------------

The guess options in the next section take determinants, which you build from strings::

    from forte2.lib.det import Determinant

    hf = Determinant("222200")

Each character is one active orbital, from left to right, in the order the solver numbers them:
GAS by GAS, which is also the order ``get_top_determinants`` prints. Core and frozen orbitals
don't appear. The characters are:

``2``
    Doubly occupied.

``a``
    Singly occupied by an alpha electron.

``b``
    Singly occupied by a beta electron.

``0``
    Empty.

In the two-component solvers, each character is a spinor, so a determinant contains only ``a`` and
``0``. For example, ``Determinant("aaaaaaaa0000")`` puts eight electrons in the first eight of
twelve spinors.

Initial guesses
---------------

The Davidson-Liu solver needs a starting vector for each root. By default, the solver builds these
automatically. You can instead give it the determinants to build them from, with ``guess_dets``
and ``pinned_guess_dets``, or the vectors themselves, with ``ci_vector_guess``.

The guess options apply to the first CI solve. When a solver runs again, for example in the
macro-iterations of ``MCOptimizer``, it restarts from its previous vectors.

Automatic guess
~~~~~~~~~~~~~~~

By default, the solver starts from a small diagonalization:

1. It ranks the candidate basis functions, every CSF (or every determinant in two-component CI),
   by diagonal energy, or by distance from ``energy_shift`` if you set it.
2. It keeps at most ``ndets_per_guess * guess_per_root * nroot`` of them. The defaults,
   ``ndets_per_guess=10`` and ``guess_per_root=2``, keep 20 basis functions per root.
3. It diagonalizes the Hamiltonian among the kept basis functions and starts the Davidson-Liu
   solver from the lowest ``guess_per_root * nroot`` eigenvectors.

The extra guess vectors beyond one per root make it more likely that the starting subspace overlaps
every target state, which can save iterations.

Guess determinants
~~~~~~~~~~~~~~~~~~

To choose the candidates yourself, pass determinants as ``guess_dets``. They replace the automatic
candidates, and the solver ranks and narrows them as in steps 1 to 3. In non-relativistic CI, each
determinant brings in every CSF of its configuration, so a single open-shell determinant is enough
to describe a spin-coupled state.

To keep basis functions that the narrowing would drop, pass them as ``pinned_guess_dets``. Pinned
determinants always join the guess space, on top of the narrowed candidates, and you can pin
determinants with or without a ``guess_dets``::

    ci_solver = forte2.CISolver(
        states=forte2.State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
        guess_dets=[Determinant("222200"), Determinant("22a2b0"), Determinant("222ab0")],
        pinned_guess_dets=[Determinant("222020")],
    )

Both options are per state. With one state, pass a list of determinants. With several, pass one
entry per state: a list of determinants, or None to use the automatic candidates for that state::

    ci_solver = forte2.CISolver(
        states=[singlet, triplet],
        nroots=[3, 2],
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
        guess_dets=[None, [Determinant("22a2b0"), Determinant("222ab0")]],
    )

If the guess determinants yield fewer vectors than roots, the solver raises an error.

CI vector guesses
~~~~~~~~~~~~~~~~~

To start from vectors you already have, pass ``ci_vector_guess``, a list with exactly one guess per
root in absolute root order. Each guess is a list of ``(Determinant, coefficient)`` pairs. The
solver uses the guesses as given, without diagonalizing anything, and the Davidson-Liu solver
starts from exactly these vectors, so ``guess_per_root`` and ``ndets_per_guess`` don't apply. You
can't combine ``ci_vector_guess`` with either guess determinant option.

``get_top_determinants`` returns its results in the same format, so you can restart from a previous
calculation in the same active space::

    previous = ci.ci_solver.get_top_determinants(n=50)
    ci_solver = forte2.CISolver(
        states=forte2.State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
        ci_vector_guess=previous,
    )

You don't need to normalize the guesses. In non-relativistic CI, each guess is projected onto the
CSFs of the target multiplicity, so a spin-impure guess, such as a single open-shell determinant,
keeps only its component of that multiplicity. A guess with no such component raises an error, as
do two guesses that project onto the same vector.

.. note::
    For two open-shell electrons, ``[(Determinant("22a2b0"), 1.0), (Determinant("22b2a0"), 1.0)]``
    is the singlet, and the combination with a minus sign is the triplet with Ms = 0.

.. caution::
    A vector guess sets where the Davidson-Liu solver starts, not which roots it converges to: it
    still converges the lowest roots, or the ones nearest ``energy_shift``. A guess that lacks the
    character of a target state slows convergence. If symmetry or spin separates that state from
    every guess, the solver can't reach it at all.

Selected CI
-----------

``SelectedCISolver`` builds a determinant space iteratively: each cycle adds the determinants that
contribute more than ``var_threshold`` to the wavefunction, and a second-order perturbative
correction accounts for the rest. Its parameters are in ``SelectedCIParams``::

    from forte2.base_classes import SelectedCIParams

    sci_solver = forte2.SelectedCISolver(
        states=forte2.State(nel=10, multiplicity=1, ms=0.0),
        core_orbitals=[0],
        active_orbitals=[1, 2, 3, 4, 5, 6],
        sci_params=SelectedCIParams(var_threshold=1e-5, pt2_threshold=1e-8),
    )
    sci = forte2.CI(sci_solver)(rhf)
    sci.run()

To optimize orbitals with selected CI as the solver, pass it to ``MCOptimizer`` instead of ``CI``.

The guess options work as in CI, with two differences:

- The guess space is also the initial determinant space. Without a ``guess_dets``, the candidates
  are every determinant of a small complete active space made of the highest
  ``guess_occ_window`` occupied and the lowest ``guess_vir_window`` virtual orbitals. In
  non-relativistic selected CI, the narrowed candidates are completed with every spin arrangement
  of their configurations.
- The solver starts from ``nroot`` vectors, not ``guess_per_root * nroot``. With
  ``ci_vector_guess``, the determinants of the guesses make up the initial space.

With a GAS active space, the selection and the perturbative correction include only the
determinants within the occupation limits of the state, so selected CI converges to GASCI. The
guess determinants must satisfy the limits, and the automatic guess keeps only the candidates that
do. For example, for the lowest core-excited triplet of Ne, with at most one electron in 1s::

    sci_solver = forte2.SelectedCISolver(
        states=forte2.State(nel=10, multiplicity=3, ms=1.0, gas_max=[1]),
        active_orbitals=[[0], list(range(1, 14))],
        guess_dets=[Determinant("a2222a")],
        sci_params=SelectedCIParams(var_threshold=1e-5),
    )

Relativistic solvers
--------------------

With a two-component reference, use ``RelCISolver`` or ``RelSelectedCISolver``, with either
driver. Instead of a ``State``, you can pass the number of electrons with ``nel``, and the orbital
options count spinors::

    rel_solver = forte2.RelCISolver(nel=10, core_orbitals=2, active_orbitals=12)

The two-component solvers work in the determinant basis, so guesses aren't spin-projected, and
``ci_vector_guess`` coefficients can be complex. To set up a two-component reference, see
:doc:`relativistic`.

Results
-------

After ``run``, both drivers hold the energies of the roots in ``E_ci`` and their weighted average in
``E_avg`` and ``E``. ``MCOptimizer`` also holds the optimized orbitals, which the methods after it
use. The solver provides the wavefunctions:

- ``ci_solver.get_top_determinants(n)`` returns the ``n`` largest determinants of each root, as
  ``(Determinant, coefficient)`` pairs.
- The drivers' ``make_rdm`` and ``make_cumulant`` methods return reduced density matrices and
  cumulants. See :doc:`rdms`.

The ``final_orbitals`` driver option sets the orbitals at the end of the calculation:

``"original"``
    Keeps the orbitals as they are. This is the default for ``CI``.

``"semicanonical"``
    Diagonalizes the averaged Fock matrix within the core, active, and virtual orbitals. This is
    the default for ``MCOptimizer``.

``"natural"``
    Semicanonical core and virtual orbitals, and active orbitals that diagonalize the averaged
    one-particle density matrix.

If the orbitals change, the driver solves the CI problem again in the new basis.
