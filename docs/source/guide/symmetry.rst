Point-group symmetry
====================

Forte2 uses the largest Abelian subgroup of a molecule's point group: D\ :sub:`2h` or one of its
subgroups. Symmetry labels every orbital with an irreducible representation (irrep), keeps
self-consistent field (SCF) orbitals pure in symmetry, and lets you choose the irrep or the
irrep occupations of a Hartree-Fock determinant.

To use symmetry, set ``symmetry=True`` when you build the ``System``::

    system = forte2.System(
        xyz="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        symmetry=True,
    )

Point-group detection
---------------------

If ``symmetry=True``, the ``System`` prepares the geometry in three steps:

1. Moves the molecule to its center of mass.
2. Finds the largest Abelian point group and rotates the molecule into that group's standard
   orientation.
3. Symmetrizes the molecule.

``symmetry_tol`` is a distance in bohr, with a default of ``1e-4``. A symmetry operation is accepted
when it maps every atom to within ``symmetry_tol`` of an atom of the same element, so averaging
moves each atom by at most ``symmetry_tol``.

The standard orientations are as follows:

* C\ :sub:`2`, C\ :sub:`2v`, and C\ :sub:`2h`: the C\ :sub:`2` axis is z.
* C\ :sub:`s` and C\ :sub:`2h`: the mirror plane is xy.
* D\ :sub:`2` and D\ :sub:`2h`: the C\ :sub:`2` axes are x, y, and z.

In C\ :sub:`2v`, which of the two remaining axes becomes x depends on the molecule, so pairs of
labels such as b\ :sub:`1` and b\ :sub:`2` can swap between molecules. To see the orientation,
check the logged geometry.

Irrep labels and indices follow Cotton ordering:

.. list-table::
   :header-rows: 1

   * - Group
     - Irreps, from index 0
   * - D\ :sub:`2h`
     - ``ag``, ``b1g``, ``b2g``, ``b3g``, ``au``, ``b1u``, ``b2u``, ``b3u``
   * - D\ :sub:`2`
     - ``a``, ``b1``, ``b2``, ``b3``
   * - C\ :sub:`2v`
     - ``a1``, ``a2``, ``b1``, ``b2``
   * - C\ :sub:`2h`
     - ``ag``, ``bg``, ``au``, ``bu``
   * - C\ :sub:`2`
     - ``a``, ``b``
   * - C\ :sub:`s`
     - ``a'``, ``a''``
   * - C\ :sub:`i`
     - ``g``, ``u``
   * - C\ :sub:`1`
     - ``a``

.. note::
   Methods that rebuild the ``System`` at displaced geometries, such as ``FDGradient`` and
   ``GeometryOptimizer``, need ``symmetry=False``.

Symmetry in Hartree-Fock
------------------------

RHF, ROHF, and UHF diagonalize the Fock matrix within each irrep, so every orbital transforms as a
single irrep. The labels are in ``scf.irrep_labels`` and the indices in ``scf.mos.irrep_indices``.
After convergence, ``scf.determinant_symmetry`` holds the irrep of the Hartree-Fock determinant,
which is also logged.

* GHF doesn't use point-group symmetry. If the ``System`` has a point group, GHF logs a warning and
  runs in C\ :sub:`1`.
* If the Hamiltonian couples different irreps, for example through an external field that lowers
  the symmetry, the SCF raises an error. In that case, use ``symmetry=False``.
* If you supply initial orbitals through ``scf.C``, Forte2 first makes them symmetric, keeping as
  much of their occupied space as it can.
* If ``guess_mix=True``, UHF mixes the highest occupied and lowest unoccupied orbitals when they
  share an irrep. Otherwise it mixes the occupied and virtual orbitals of one irrep with the
  smallest energy gap, and logs a warning. A broken-symmetry solution that also breaks the point
  group, such as UHF for a stretched homonuclear diatomic, needs ``symmetry=False``.

Occupation constraints
----------------------

By default, the SCF occupies the lowest orbitals, whatever their irreps. To choose the occupation,
pass one or both of these options to the SCF method:

``target_symmetry``
    The irrep of the determinant, as a label or index. In each iteration, the SCF occupies the
    orbitals with the lowest orbital energy sum among possible determinants of this irrep.

``irrep_occupations``
    Dictionary of the number of electrons in each irrep, keyed by irrep label or index. An integer is the irrep's total
    number of electrons, split equally between the spins, so it must be even. An ``(alpha, beta)``
    pair gives the electrons of each spin. Every electron must be assigned, and irreps that aren't
    listed are empty.

If you pass both options, the occupations must give a determinant of the target irrep.

For example, both of these ROHF calculations give the :sup:`2`\ A\ :sub:`1` state of
H\ :sub:`2`\ O\ :sup:`+`, which has its hole in 3a\ :sub:`1` instead of 1b\ :sub:`1`::

    rohf = forte2.ROHF(charge=1, ms=0.5, target_symmetry="a1")(system)
    rohf = forte2.ROHF(
        charge=1, ms=0.5, irrep_occupations={"a1": (3, 2), "b1": 2, "b2": 2}
    )(system)

Each method accepts the following options:

.. list-table::
   :header-rows: 1

   * - Method
     - ``target_symmetry``
     - ``irrep_occupations``
   * - RHF
     - Totally symmetric only.
     - Integers must be even, and pairs must have equal alpha and beta electrons.
   * - ROHF
     - Any irrep, or totally symmetric only if ``ms=0``.
     - Integers must be even. In each irrep, the spin with more electrons overall must have at
       least as many electrons as the other spin.
   * - UHF
     - Any irrep.
     - Integers must be even.
   * - GHF
     - Not supported.
     - Not supported.

Forte2 checks these options when you attach the method to the ``System``, so a mistake raises an
error before the SCF runs.

Active-space methods
--------------------

AVAS and ASET rotate orbitals only within an irrep, and their orbitals keep correct labels. If the
AVAS subspace or the ASET fragment breaks the symmetry, for example a subspace on only one of two
equivalent atoms, they raise an error. In that case, use a subspace that the symmetry operations
map onto itself, or use ``symmetry=False``.

MCSCF mixes only orbitals of the same irrep. To select CI states of one irrep, pass its Cotton index
to ``State``, for example ``State(nel=14, multiplicity=1, ms=0.0, symmetry=0)``.
