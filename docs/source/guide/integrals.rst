Obtaining integrals
===================

Forte2 uses the `Libint2 <https://github.com/evaleev/libint>`_ integral engine. 
It provides a straightforward way of accessing atomic integrals through its API. 
You can obtain the integrals for a given molecular system by first specifying a molecular system and using the ``forte2.lib.ints`` module. 
Almost all operators supported by Libint2 (see `Libint2 documentation <https://github.com/evaleev/libint/wiki/using-modern-CPlusPlus-API#create-an-integral-engine>`_) are available.
Forte2 also provides an interface with `libcint <https://github.com/sunqm/libcint>`_, with a templated API that supports easy addition of new integral types. Currently, only a small subset (mainly two-center integrals) of libcint integrals are imported, but more can be easily added as needed.
Here are some examples of how to obtain the most common integrals. First one needs to set up the molecular system::

    import forte2

    # Set up your molecular system
    system = forte2.System(
        xyz="""C 0 0 0
        N 0 0 1.4""",
        basis_set={"C": "cc-pvdz", "N": "cc-pvtz"},
        auxiliary_basis_set="cc-pvdz-jkfit",
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
    dipole = forte2.lib.ints.emultipole1(system.basis, system.atoms)[1:]

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

As shown above, the ``forte2.integrals`` module automatically supplies sensibly default basis sets and geometry information from the ``system`` object, making it more convenient to use in many cases.

Two-electron integral factorization
------------------------------------

Forte2 never stores the full four-index ERI tensor for production calculations. Instead the
two-electron integrals are represented by a three-index ``B`` tensor, :math:`(mn|rs) \approx \sum_Q
B^Q_{mn} B^Q_{rs}`, built in one of two ways selected on the ``System``:

* **Density fitting (DF)** -- the default. Requires an ``auxiliary_basis_set``; the fitting error is
  controlled by the choice of auxiliary basis (for example ``cc-pvtz-jkfit``).
* **Cholesky decomposition (CD)** -- enabled with ``cholesky_tei``. No auxiliary basis is needed;
  the accuracy is controlled directly by ``cholesky_tol``, and CD reconstructs the *exact* ERI to
  that tolerance rather than fitting onto a fixed auxiliary basis.

To use Cholesky-decomposed integrals, set ``cholesky_tei`` when constructing the system::

    system = forte2.System(
        xyz="N 0 0 0; N 0 0 1.1",
        basis_set="cc-pvdz",
        cholesky_tei="otf",   # or True
        cholesky_tol=1e-8,
    )

``cholesky_tei`` accepts:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Meaning
   * - ``False`` (default)
     - Cholesky disabled; use density fitting with the ``auxiliary_basis_set``.
   * - ``True`` or ``"otf"``
     - On-the-fly one-step pivoted Cholesky (Koch 2003). Never forms the full four-index tensor;
       Schwarz-screened and proactively drained. Recommended when CD is wanted.
   * - ``"pivoted"``
     - On-the-fly two-step Cholesky (Folkestad 2019): Step I selects the pivot AO pairs, Step II
       builds the vectors by an RI fit onto that Cholesky basis. Also never forms the full tensor;
       typically keeps a few more vectors than ``"otf"``.
   * - ``"naive"``
     - Dense reference path: forms the full four-index ERI and decomposes it (:math:`O(N^4)`
       memory). A numerical oracle only -- not for production-size systems.

``cholesky_tol`` (default ``1e-6``) sets the decomposition threshold: the reconstruction error of
``B`` is bounded roughly elementwise by this value, so ``1e-6`` is adequate for energies while
``1e-8`` to ``1e-10`` gives near-exact ERIs at the cost of more Cholesky vectors. All three CD modes
reconstruct the same operator to ``cholesky_tol`` and therefore give the same energies (SCF, MCSCF,
DSRG-MRPT2, ...) to that accuracy; they differ only in how the vectors are built.

.. note::

   Analytic gradients are implemented only for the density-fitting path. Requesting a gradient with
   any ``cholesky_tei`` mode raises ``NotImplementedError``.