Obtaining integrals
===================

Forte2 computes atomic integrals with the `Libint2 <https://github.com/evaleev/libint>`_ integral engine and, when Forte2 is built with libcint (the default), with `libcint <https://github.com/sunqm/libcint>`_.
Almost all operators supported by Libint2 (see the `Libint2 documentation <https://github.com/evaleev/libint/wiki/using-modern-CPlusPlus-API#create-an-integral-engine>`_) are available.
libcint provides the one-electron integrals and the two- and three-center Coulomb integrals, including spinor variants for relativistic calculations.
The ``integral_backend`` option of ``System`` selects the engine. The default, ``"auto"``, uses Libint2 up to the highest angular momentum it supports and libcint above that.
Here are some examples of how to obtain the most common integrals. First one needs to set up the molecular system::

    import forte2

    # Set up your molecular system
    system = forte2.System(
        xyz="""C 0 0 0
        N 0 0 1.4""",
        basis_set={"C": "cc-pvdz", "N": "cc-pvtz"},
        auxiliary_basis_set="cc-pvtz-jkfit",
        minao_basis_set="ano-r0",
    )

The ``system`` object will now contain parsed geometry under ``atoms``, the basis set under ``basis``, the auxiliary basis set under ``auxiliary_basis``, and the minimal atomic basis set under ``minao_basis``.

There are two ways of obtaining integrals: using the ``forte2.lib.ints`` module (direct C++ API calls to Libint2 and libcint), or using the ``forte2.integrals`` module (Python wrappers around the C++ API calls). The two ways are equivalent, but the latter can be more user-friendly.

Getting integrals through ``forte2.lib.ints`` can be achieved as follows::

    # overlap integrals
    overlap = forte2.lib.ints.overlap(system.basis)

    # "mixed basis" overlap integrals are available simply as:
    mixed_overlap = forte2.lib.ints.overlap(system.minao_basis, system.basis)

    # kinetic energy integrals
    kinetic = forte2.lib.ints.kinetic(system.basis)

    # potential energy integrals
    potential = forte2.lib.ints.nuclear(system.basis, system.atoms)

    # dipole integrals (ordered x,y,z)
    # the zeroth element is the overlap
    dipole = forte2.lib.ints.emultipole1(system.basis)[1:]

    # 4-center-2-electron integrals
    eri = forte2.lib.ints.coulomb_4c(system.basis)

    # 3-center-2-electron integrals (for density-fitting)
    B = forte2.lib.ints.coulomb_3c(system.auxiliary_basis, system.basis, system.basis)

Equivalently, getting integrals through ``forte2.integrals`` can be achieved as follows::

    # overlap integrals
    overlap = forte2.integrals.overlap(system)

    # "mixed basis" overlap integrals are available simply as:
    mixed_overlap = forte2.integrals.overlap(system, system.minao_basis, system.basis)

    # kinetic energy integrals
    kinetic = forte2.integrals.kinetic(system)

    # potential energy integrals
    potential = forte2.integrals.nuclear(system)

    # dipole integrals (ordered x,y,z)
    # the zeroth element is the overlap
    dipole = forte2.integrals.emultipole1(system)[1:]

    # 4-center-2-electron integrals
    eri = forte2.integrals.coulomb_4c(system)

    # 3-center-2-electron integrals (for density-fitting)
    B = forte2.integrals.coulomb_3c(system)

As shown above, the ``forte2.integrals`` module automatically supplies sensible default basis sets and geometry information from the ``system`` object, making it more convenient to use in many cases.
