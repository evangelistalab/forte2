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

Exporting orbitals
------------------

Once a method has been run, you can export its orbitals for visualization in any program that reads Molden files (VibeMol, Molden, Jmol, IboView, ...)::

    forte2.write_molden(rhf, "rhf.molden")
    forte2.write_molden(casscf, "casscf.molden")

This works for SCF references (RHF, ROHF, UHF), AVAS, and active-space drivers and solvers (``CI``, ``MCOptimizer``, ``CISolver``, ...).
Each orbital carries its symmetry label, an energy and an occupation:

* SCF exports orbital energies and occupations from the SCF density. For reordered orbitals, it projects the stored Fock matrix onto the exported orbitals.
* AVAS: the reference Fock diagonal, and the nominal (0, 1 or 2) occupations of the SCF reference it rotates.
* CI, MCOptimizer and their solvers: the diagonal of the generalized Fock matrix; 2 for core orbitals, the diagonal of the state-averaged 1-RDM for active orbitals (the natural occupation numbers when ``final_orbitals="natural"``), and 0 for virtual orbitals.

Methods whose orbitals derive from a correlated calculation by other means, such as ``ASET`` and ``DSRG_MRPT2``, are not supported directly; write the orbitals of the underlying ``MCOptimizer`` instead, or pass the coefficients explicitly as shown below.

To write any other set of orbitals, for example localized orbitals, pass the ``System`` and the coefficients.
:func:`forte2.molden_string <forte2.orbitals.molden.molden_string>` returns the same text without writing a file::

    forte2.write_molden(rhf.system, "localized.molden", C=C_loc, occupations=occ)
    text = forte2.molden_string(casscf)

Explicit coefficient arrays must contain at least one orbital, and all supplied
coefficients, energies, and occupations must be finite.

Molden files cover real, one-component orbitals with shells up to g.
Pass ``ignore_unsupported_shells=True`` to omit shells beyond g and their
MO coefficients. The exported orbitals may lose normalization.

For two-component orbitals, or for grid-based visualization, write cube files::

    forte2.write_orbital_cubes(rhf.system, rhf.mos.C[0], indices=[0, 1, 2], filepath="cubes")

Parallelism
-----------

Forte2 automatically detects the number of threads to use for some parallel sections that are not already parallelized by, for example, BLAS.
The effective count is printed at ``import forte2``.

If the environment variable ``FORTE_NUM_THREADS_OVERRIDE`` is set, Forte2 uses that count.
Otherwise, it uses the smallest of the logical CPU count, the CPU affinity mask, and ``OMP_NUM_THREADS``, ``OMP_THREAD_LIMIT``, and ``SLURM_CPUS_PER_TASK`` where they are set.
The logical CPU count includes hyperthreads.
