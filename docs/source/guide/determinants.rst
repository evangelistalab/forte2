Determinants
============

The ``forte2.lib.det`` module provides a ``Determinant`` class for Slater determinants, generators for lists of determinants, and builders for dense matrices in a determinant basis.
A ``Determinant`` stores an alpha and a beta occupation string, each covering up to 64 orbitals.

Construct a determinant
-----------------------

To build a determinant from its occupation pattern, pass a string with one character per orbital, starting from orbital 0:

- ``0``: empty
- ``a``: occupied by an alpha electron
- ``b``: occupied by a beta electron
- ``2``: doubly occupied

To build a determinant one orbital at a time, start from ``Determinant.zero()`` and call ``set_na`` and ``set_nb``.

Generate lists of determinants
------------------------------

Two functions generate lists of determinants:

- ``forte2.lib.det.hilbert_space`` generates every determinant with a given number of alpha and beta electrons, optionally restricted by symmetry or by excitation rank.
- ``forte2.lib.ci_helpers.CIStrings`` generates determinants in a generalized active space (GAS), which restricts the number of electrons in each subspace of orbitals.

Full and truncated spaces
~~~~~~~~~~~~~~~~~~~~~~~~~

``hilbert_space(nmo, na, nb)`` returns all determinants with ``na`` alpha and ``nb`` beta electrons in ``nmo`` orbitals.

- To keep only determinants of one irrep, pass ``nirrep``, the irrep of each orbital in ``mo_symmetry``, and the target irrep in ``symmetry``.
- To truncate the space, pass a reference determinant ``ref`` and the maximum excitation rank ``truncation``. Excitation rank counts alpha and beta excitations together.

.. code-block:: python

    from forte2.lib.det import Determinant, hilbert_space

    print(len(hilbert_space(nmo=4, na=2, nb=2)))  # 36

    # Determinants of irrep 1 when orbitals 1 and 3 belong to irrep 1
    dets = hilbert_space(nmo=4, na=1, nb=1, nirrep=2, mo_symmetry=[0, 1, 0, 1], symmetry=1)
    print(len(dets))  # 8

    # The reference plus all single excitations
    ref = Determinant("2200")
    dets = hilbert_space(nmo=4, na=2, nb=2, ref=ref, truncation=1)
    print(len(dets))  # 9

Generalized active spaces
~~~~~~~~~~~~~~~~~~~~~~~~~

``CIStrings`` takes these arguments:

``na``, ``nb``
    The number of alpha and beta electrons.

``symmetry``
    The irrep of the determinants.

``orbital_symmetry``
    One list per GAS space, holding the irrep of each orbital in that space. Orbitals are numbered consecutively across spaces, and up to six spaces are supported.

``gas_min``, ``gas_max``
    The minimum and maximum number of electrons, alpha and beta combined, in each space. An empty list leaves every space unrestricted.

``make_determinants`` returns the determinants in the order of the CI vectors used by the forte2 CI solvers, and ``determinant_index`` returns the position of a given determinant.

This example splits four orbitals into two spaces and allows at most one electron in the second space. The result is the same set as the single excitations from ``|2200>`` in `Full and truncated spaces`_.

.. code-block:: python

    from forte2.lib.ci_helpers import CIStrings

    ci_strings = CIStrings(
        na=2,
        nb=2,
        symmetry=0,
        orbital_symmetry=[[0, 0], [0, 0]],
        gas_min=[],
        gas_max=[4, 1],
    )
    dets = ci_strings.make_determinants()
    print(len(dets))  # 9

Doubly occupied and empty orbitals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To pad the determinants with doubly occupied orbitals, add a fully occupied GAS space:

- For a block of ``n`` doubly occupied orbitals, set both its ``gas_min`` and ``gas_max`` entries to ``2 * n``.

Orbitals after the last space are empty, so trailing empty orbitals don't need a space of their own.
This example places two electrons in orbitals 3 and 4, with orbitals 0 and 1 doubly occupied and orbital 2 empty:

.. code-block:: python

    from forte2.lib.ci_helpers import CIStrings

    ci_strings = CIStrings(
        na=3,
        nb=3,
        symmetry=0,
        orbital_symmetry=[[0, 0], [0], [0, 0]],
        gas_min=[4, 0, 0],
        gas_max=[4, 0, 2],
    )
    print([d.str(5) for d in ci_strings.make_determinants()])
    # ['|22002>', '|220ba>', '|220ab>', '|22020>']

Spinor determinants
-------------------

A determinant of two-component spinors is an all-alpha determinant: the alpha string stores the occupied spinors, and the beta string is empty.
A ``Determinant`` therefore holds up to 64 spinors.

To generate spinor determinants, set the number of alpha electrons to the number of electrons and the number of beta electrons to 0:

.. code-block:: python

    from forte2.lib.det import Determinant, hilbert_space
    from forte2.lib.ci_helpers import CIStrings

    print(Determinant("aa0a").count_beta())  # 0

    dets = hilbert_space(nmo=4, na=3, nb=0)

    ci_strings = CIStrings(
        na=3, nb=0, symmetry=0, orbital_symmetry=[[0] * 4], gas_min=[], gas_max=[]
    )
    print([d.str(4) for d in ci_strings.make_determinants()])
    # ['|0aaa>', '|a0aa>', '|aa0a>', '|aaa0>']

Build determinant matrices
--------------------------

The module builds two kinds of dense matrices from a list of determinants:

- ``make_s2_matrix`` builds the spin-squared matrix for restricted spatial orbitals.
- ``make_hamiltonian_matrix`` builds the Hamiltonian matrix for restricted spatial orbitals or two-component spinors.

Both return NumPy arrays whose rows and columns follow the order of the supplied determinants.
If a determinant occupies an orbital outside the range of the integrals, ``make_hamiltonian_matrix`` raises ``ValueError``.

Spin-squared matrix
~~~~~~~~~~~~~~~~~~~

The determinants ``ab`` and ``ba`` describe one alpha and one beta electron distributed over two singly occupied orbitals.

.. code-block:: python

    import numpy as np
    from forte2.lib.det import Determinant, make_s2_matrix

    dets = [Determinant("ab"), Determinant("ba")]
    s2 = make_s2_matrix(dets)

    print(s2)
    print(np.linalg.eigvalsh(s2))  # [0. 2.]

The eigenvalues are :math:`S(S+1)`, giving a singlet with :math:`S=0` and the :math:`M_S=0` component of a triplet with :math:`S=1`.

Hamiltonian of molecular hydrogen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For H\ :sub:`2` in the STO-6G basis, run restricted Hartree–Fock and compute the integrals for both spatial orbitals.
``SlaterRules`` takes the scalar energy ``E``, the effective one-electron integrals ``H``, and the two-electron integrals ``V`` in physicist's notation, :math:`\langle pq | rs \rangle`.
Use the default ``RestrictedMOIntegrals`` convention, which supplies ``V`` before antisymmetrization.

Diagonalizing the Hamiltonian in the four determinants from ``hilbert_space`` gives the full CI energies in this basis.
The same determinant list gives the ground state's spin expectation value.

.. code-block:: python

    import numpy as np
    from forte2 import RHF, System
    from forte2.jkbuilder import RestrictedMOIntegrals
    from forte2.lib.det import (
        SlaterRules,
        hilbert_space,
        make_hamiltonian_matrix,
        make_s2_matrix,
    )

    system = System(
        xyz="H 0.0 0.0 0.0; H 0.0 0.0 0.74",
        basis_set="sto-6g",
        unit="angstrom",
        cholesky_tei=True,
    )
    rhf = RHF(charge=0)(system)
    rhf.run()

    orbitals = [0, 1]
    ints = RestrictedMOIntegrals(system=system, C=rhf.mos.C[0], orbitals=orbitals)
    rules = SlaterRules(len(orbitals), ints.E, ints.H, ints.V)
    dets = hilbert_space(nmo=len(orbitals), na=rhf.na, nb=rhf.nb)

    hamiltonian = make_hamiltonian_matrix(dets, rules)
    energies, vectors = np.linalg.eigh(hamiltonian)
    ground_state = vectors[:, 0]
    s2 = make_s2_matrix(dets)

    print(f"Ground-state energy = {energies[0]:.6f} hartree")
    print(f"Ground-state <S^2> = {ground_state @ s2 @ ground_state:.6f}")

The ground state is a singlet, so its spin expectation value is zero to numerical precision.

Two-component Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~~~~

To build a complex Hermitian Hamiltonian from two-component spinor integrals, pass a ``RelSlaterRules`` object and a list of `Spinor determinants`_ to ``make_hamiltonian_matrix``.
In a molecular calculation, ``SpinorbitalIntegrals`` supplies the scalar energy and the one- and two-electron integrals.
This example uses one electron in two spinors:

.. code-block:: python

    import numpy as np
    from forte2.lib.det import RelSlaterRules, hilbert_space, make_hamiltonian_matrix

    h = np.array([[0.2, 0.1j], [-0.1j, 0.5]])
    v = np.zeros((2, 2, 2, 2), dtype=complex)
    rules = RelSlaterRules(2, 0.0, h, v)
    dets = hilbert_space(nmo=2, na=1, nb=0)

    hamiltonian = make_hamiltonian_matrix(dets, rules)
    print(np.linalg.eigvalsh(hamiltonian))
