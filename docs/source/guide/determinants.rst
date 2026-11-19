Determinant matrices
====================

The ``forte2.lib.det`` module builds dense Hamiltonian and spin-squared matrices in a basis of determinants formed from a common set of spatial orbitals.
Both builders return NumPy arrays whose rows and columns follow the order of the supplied determinants.

Spin-squared matrix from two determinants
-----------------------------------------

In a determinant string, ``a`` and ``b`` denote singly occupied orbitals with alpha and beta spin, respectively.
The characters ``0`` and ``2`` denote empty and doubly occupied orbitals.
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
---------------------------------

For H\ :sub:`2` in the STO-6G basis, run restricted Hartree–Fock and compute the integrals for both spatial orbitals.
``SlaterRules`` takes the scalar energy ``E``, the effective one-electron integrals ``H``, and the two-electron integrals ``V`` in physicist's notation, :math:`\langle pq | rs \rangle`.
Use the default ``RestrictedMOIntegrals`` convention, which supplies ``V`` before antisymmetrization.

``hilbert_space`` generates the four determinants for one alpha and one beta electron in two orbitals.
Diagonalizing the resulting Hamiltonian gives the full CI energies in this basis.
The same determinant list can be used to compute the ground state's spin expectation value.

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
