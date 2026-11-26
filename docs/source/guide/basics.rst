Basic usage
===========

Forte2 is written in C++ and Python.
Most of its functionality is implemented in Python, with only the most demanding parts implemented in C++.
The C++ code is exposed to Python through nanobind.

Forte2 uses the `functional composition <https://en.wikipedia.org/wiki/Function_composition_(computer_science)>`_ style (like the `TensorFlow functional API <https://www.tensorflow.org/guide/keras/functional_api>`_) for most quantum chemical methods, with the following programmatic flow::

    rhf = forte2.RHF(charge=0)(system)
    ci = forte2.CI(forte2.CISolver(states=state, active_orbitals=[...]))(rhf)
    ci.run()

This lets you chain methods together flexibly.
Each method checks its arguments, and whether it can follow the method before it, when you build the chain, so errors surface before any time-consuming calculation runs.
A single ``run`` call then executes the whole chain.

.. admonition:: Experimental
   :class: tip

   To build an input without writing code, try the `Forte2 input builder <https://brianz98.github.io/forte2-builder/>`_,
   where you can browse template inputs or build your own by interactively connecting methods into a
   chain. It will be validated, and a Python script will be generated for you.

To set up a molecular system, create a ``System``::

    system = forte2.System(
        xyz="C 0 0 0; N 0 0 1.4",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
    )

You can then attach a Hartree-Fock calculation to the system::

    rhf = forte2.RHF(charge=-1)(system)

or a restricted open-shell Hartree-Fock calculation::

    rohf = forte2.ROHF(charge=0, ms=0.5)(system)

You might then want to perform an atomic valence active space (AVAS) calculation to prepare for a CASSCF calculation::

    avas = forte2.AVAS(subspace=["C(2s)", "C(2p)", "N(2p)"])(rohf)

A state-averaged CASSCF calculation on the AVAS orbitals takes a CI solver, which holds the states, and an ``MCOptimizer``, which optimizes the orbitals around it::

    doublet = forte2.State(nel=13, multiplicity=2, ms=0.5)
    singlet = forte2.State(nel=14, multiplicity=1, ms=0.0)
    triplet = forte2.State(nel=14, multiplicity=3, ms=0.0)
    ci_solver = forte2.CISolver(
        states=[doublet, singlet, triplet],
        nroots=[2, 3, 1],
        weights=[[2, 1], [1, 1, 1], [0.5]],
    )
    casscf = forte2.MCOptimizer(ci_solver)(avas)

At this point the methods are bound to each other and have checked their inputs, but nothing has been computed, including AVAS.
To run the whole chain, call ``run`` on its last method::

    casscf.run()

To use a relativistic Hamiltonian, see :doc:`relativistic`.

Orbital localization
--------------------

``Localizer`` rotates orbitals to concentrate the charge of each orbital on as few atoms as
possible, as in Pipek-Mezey localization. By default, it measures the charges in intrinsic atomic
orbitals (IAOs), which gives intrinsic bond orbitals (IBOs). To localize the AVAS active orbitals
before the CASSCF calculation, insert a ``Localizer`` between them::

    loc = forte2.Localizer()(avas)
    casscf = forte2.MCOptimizer(ci_solver)(loc)

Rotations within the active space don't change CASCI or CASSCF energies. Localized active orbitals
help active-space solvers whose accuracy depends on the orbital basis, such as selected CI, because
the strongest correlations then connect orbitals on neighboring atoms.

To use other atomic charges, set ``charges`` to ``"lowdin"`` or ``"mulliken"``, and set the
localization power ``exponent`` to 2 or 4. The original Pipek-Mezey orbitals use Mulliken charges
and exponent 2::

    loc = forte2.Localizer(charges="mulliken", exponent=2)(avas)

By default, ``Localizer`` localizes the active orbitals, one GAS at a time. If the method before it
defines no orbital spaces, for example an SCF method, ``Localizer`` localizes the occupied orbitals,
the doubly and singly occupied ones separately. To localize other orbitals, pass their MO indices to
``orbitals``, or a list of index lists to localize several sets separately, for example
``orbitals=[[0, 1, 2], [5, 6]]``. If a set mixes orbital spaces, ``Localizer`` warns that the
energies of the methods that follow change. To keep orbitals with different reference occupations
apart, set ``split_by_occupation=True``; a doubly occupied orbital then never mixes with a singly
occupied one.

The IAOs span the frozen core, core, and active orbitals, or the occupied orbitals if there are no
orbital spaces, so IAO charges can't localize virtual orbitals. To localize virtual orbitals, use
Löwdin charges. Mulliken charges are poorly defined for diffuse virtual orbitals, and localization
with them converges slowly. Localization breaks point-group symmetry, so build the ``System`` with
``symmetry=False``.

Parallelism
-----------

Forte2 automatically detects the number of threads to use for some parallel sections that are not already parallelized by, for example, BLAS.
The effective count is printed at ``import forte2``.

If the environment variable ``FORTE_NUM_THREADS_OVERRIDE`` is set, Forte2 uses that count.
Otherwise, it uses the smallest of the logical CPU count, the CPU affinity mask, and ``OMP_NUM_THREADS``, ``OMP_THREAD_LIMIT``, and ``SLURM_CPUS_PER_TASK`` where they are set.
The logical CPU count includes hyperthreads.
