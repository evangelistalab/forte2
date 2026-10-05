Basic usage
=================

Forte2 is written in C++ and Python. 
Most core functionalities are implemented in Python, with only the most demanding parts implemented in C++. 
The C++ code is exposed to Python through the use of nanobind, which allows for a seamless integration between the two languages.


Forte2 uses the `functional composition <https://en.wikipedia.org/wiki/Function_composition_(computer_science)>`_ style (like the `TensorFlow functional API <https://www.tensorflow.org/guide/keras/functional_api>`_) for most quantum chemical methods, with the following programmatic flow:

>>> rhf = forte2.scf.RHF(charge=0)(system)
>>> ci = forte2.CI(forte2.CISolver(states=state, active_orbitals=[...]))(rhf)
>>> ci.run()
-0.75102385

This allows you to chain methods together in a very flexible way, with argument sanity checks taking place at initialization (*i.e.*, without running the potentially time-consuming chain of methods first) and you can execute the whole chain with a single ``run`` call.

Setting up a molecular system is as simple as:

>>> system = forte2.System(
    xyz="C 0 0 0; N 0 0 1.4", 
    basis_set="cc-pvdz", 
    auxiliary_basis_set="cc-pvtz-jkfit",
    )

You can then attach a Hartree-Fock calculation on the system:

>>> rhf = forte2.scf.RHF(charge=-1)(system)

The relativistic Hamiltonian is selected by passing a :class:`forte2.X2CParams`
instance to the ``x2c`` option. SAP-X2C is available for both scalar-relativistic
and spin-orbit calculations. For example:

>>> system = forte2.System(
    xyz="H 0 0 0; Br 0 0 1.4",
    basis_set="cc-pvdz",
    auxiliary_basis_set="cc-pvtz-jkfit",
    x2c=forte2.X2CParams(x2c_type="so", x2c_model="sap"),
    )
>>> ghf = forte2.scf.GHF(charge=0)(system)

Conventional one-electron X2C uses ``x2c_model="1e"`` with ``x2c_type="sf"`` or
``"so"``. Screened nuclear spin-orbit variants set ``snso_type`` to
``"boettger"``, ``"dc"``, ``"dcb"``, or ``"row-dependent"`` (only valid with
``x2c_type="so"`` and ``x2c_model="1e"``).

or for a restricted open-shell Hartree-Fock calculation:

>>> rohf = forte2.scf.ROHF(charge=0, ms=0.5)(system)

You might then want to perform an atomic valence active space (AVAS) calculation to prepare for a CASSCF calculation:

>>> avas = forte2.AVAS(subspace=["C(2s)", "C(2p)", "N(2p)"])(rohf)

And you can prepare a complicated state-averaged CASSCF solver using the AVAS orbitals (AVAS hasn't been run yet at this point):

>>> doublet = forte2.State(nel=13, multiplicity=2, ms=0.0)
>>> singlet = forte2.State(nel=14, multiplicity=1, ms=0.0)
>>> triplet = forte2.State(nel=14, multiplicity=3, ms=0.5)
>>> casscf = forte2.MCOptimizer(
    states=[doublet, singlet, triplet],
    nroots=[2,3,1],
    weights=[[2,1],[1,1,1],[0.5]],
)(avas)
 
If you execute the code now, the methods will click together under the hood, doing the necessary checks, but no computation will be performed yet.
You can then run the whole chain with a single call:

>>> casscf.run()

Molecular symmetry
------------------

Set ``symmetry=True`` on ``System`` to detect the largest Abelian point group.
Spatial Hartree-Fock methods (RHF, ROHF, UHF, and CUHF) then diagonalize the
Fock matrix separately in each irrep, preventing numerical mixing of nearly
degenerate orbitals with different symmetries. Without occupation constraints,
orbitals remain ordered by energy.
Use ``symmetry=False`` to allow solutions that break spatial symmetry.

MO symmetry detection resolves coupled orbital blocks under all point-group
operations, including noncontiguous orbitals. If the orbital space cannot be
resolved into valid irreps, it raises ``RuntimeError`` instead of assigning
totally symmetric labels to every orbital.

HF symmetry and occupations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

All HF methods accept ``target_symmetry`` to select the total electronic
determinant irrep. Specify an irrep label (case insensitive) or its Cotton index.
For example, an open-shell calculation can target ``"b1u"``:

.. code-block:: python

    hf = forte2.UHF(charge=1, ms=0.5, target_symmetry="b1u")(system)
    hf.run()
    print(hf.state_symmetry)

At every SCF iteration, occupations are selected using orbital energies subject
to the electron counts and requested determinant symmetry. Closed-shell RHF
determinants are always totally symmetric. Use explicit occupations to choose
a particular configuration when several configurations share the same irrep.

``irrep_occupations`` fixes the number of occupied orbitals in each irrep:

* RHF: an integer number of occupied **spatial orbitals**, each doubly occupied.
* ROHF, UHF, and CUHF: an ``(alpha, beta)`` pair of occupied spatial-orbital counts.
* GHF: an integer number of occupied **spinors**, each occupied by one electron.

Unlisted irreps have zero occupation, so the supplied counts must account for
all electrons. ROHF and CUHF require minority-spin occupations to be nested
within majority-spin occupations in every irrep. Both options can be supplied
together; incompatible requests raise ``ValueError``.

For stretched N2 at 2.5 Angstrom with cc-pVTZ, this selects the configuration
with three occupied ``ag`` orbitals, two ``b1u`` orbitals, and one of each
``pi_u`` component:

.. code-block:: python

    system = forte2.System(
        xyz="N 0 0 0; N 0 0 2.5",
        basis_set="cc-pvtz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
    )
    hf = forte2.RHF(
        charge=0,
        irrep_occupations={"ag": 3, "b1u": 2, "b2u": 1, "b3u": 1},
    )(system)
    hf.run()

Constrained orbitals are stored with occupied orbitals first and virtual
orbitals last, sorted by energy within each block. ROHF stores doubly occupied,
singly occupied, and virtual blocks separately. The raw constructor options
are preserved when a method chain is rebuilt or rebound.

Spin-free GHF uses the detected spatial point group. With spin-orbit coupling,
GHF uses its full double group, including spin rotations and the 2π rotation.
``hf.orbital_point_group`` identifies it with a trailing ``*`` (for example,
``"D2H*"``). Character projectors resolve multidimensional irreps rather than
assigning ordinary spatial labels to spinors. Separate alpha/beta occupation
counts are not defined for spin-orbit GHF.

The spinor labels are:

.. list-table:: Fermionic double-group irreps
    :header-rows: 1

    * - Spatial group
      - Spinor labels
      - Dimension
    * - C1
      - ``a1/2``
      - 1
    * - Ci
      - ``a1/2g``, ``a1/2u``
      - 1
    * - C2, Cs
      - ``1e1/2``, ``2e1/2``
      - 1
    * - C2h
      - ``1e1/2g``, ``2e1/2g``, ``1e1/2u``, ``2e1/2u``
      - 1
    * - D2, C2v
      - ``e1/2``
      - 2
    * - D2h
      - ``e1/2g``, ``e1/2u``
      - 2

Ordinary (bosonic) irreps retain their Cotton indices; the spinor irreps follow
in the order shown above. For C2, Cs, and C2h, the ``1``/``2`` partners have
characters ``+i``/``-i`` under the unbarred C2 rotation (mirror for Cs).
Odd-electron targets must be fermionic and even-electron targets bosonic.
For example, ``GHF(charge=1, target_symmetry="e1/2u")`` targets an odd-electron
ungerade state in D2h*, and ``irrep_occupations={"e1/2g": 4, "e1/2u": 3}``
fixes seven occupied spinors in those two irreps.

For D2*, C2v*, and D2h*, a symmetry-pure even-electron single determinant must
fill complete two-dimensional spinor multiplets and is totally symmetric.
Targeting that irrep enforces paired occupations; other bosonic target irreps
require a multideterminant wavefunction and raise ``ValueError``. Explicit
spinor counts without a target can produce a mixture of total irreps.
``hf.state_symmetry_weights`` reports their character-projector weights, and
``hf.state_symmetry`` is ``None`` when the determinant is mixed. Every requested
target is checked against the determinant itself after convergence.

These double-group constraints apply to HF. Spin-orbit CI continues to solve
in C1; its spatial-irrep XOR string machinery does not support double-group
state selection. The HF double-group indices are therefore not passed to it.

See :doc:`../technical/hf_symmetry` for the algorithms, determinant projector
formulas, refactoring boundaries, and validation strategy.

Parallelism
-----------

Forte2 automatically detects the number of threads to use for some parallel sections that are not already parallelized by e.g. BLAS.
The effective count is printed at ``import forte2``.

The envioronment variable ``FORTE_NUM_THREADS_OVERRIDE`` will be used if set. 
Otherwise, the smallest among the number of logical CPU counts, ``OMP_NUM_THREADS``,``OMP_THREAD_LIMIT``, and ``SLURM_CPUS_PER_TASK`` will be used if set.
Note that the logical CPU count includes e.g., hyperthreads.
