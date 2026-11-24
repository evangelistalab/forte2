Exporting and visualizing orbitals
==================================

A companion orbital visualization tool to Forte2 is VibeMol
(https://github.com/evangelistalab/vibemol), which can read Molden files and cube files.

Molden files
------------

Once a method has been run, you can export its orbitals for visualization in
programs that read Molden files, such as VibeMol, Molden, Jmol, and IboView.

.. code-block:: python

    from forte2.orbitals import write_molden

    # code that generates the rhf object
    write_molden(rhf, "rhf.molden")
    
This works for SCF references (RHF, ROHF, UHF), AVAS, and active-space drivers
and solvers such as ``CI``, ``MCOptimizer``, and ``CISolver``. Each file stores
the geometry, orbital basis, MO coefficients, and per-orbital symmetry labels,
energies, and occupations. Active-space exports use the generalized Fock
diagonal for the energies and the diagonal of the state-averaged one-particle
density matrix for the active occupations.

To export selected or localized orbitals, pass the ``System`` and the
coefficients explicitly. Each column of ``C`` is one orbital in the original AO
basis.

.. code-block:: python

    write_molden(
        rhf.system,
        "selected.molden",
        C=rhf.mos.C[0][:, :3],
        energies=rhf.eps[0][:3],
        occupations=[2.0, 2.0, 2.0],  # Three doubly occupied orbitals
    )

For unrestricted orbitals, pass ``C=[C_alpha, C_beta]`` and supply one set of
values per spin for any energies, occupations, or symmetry labels you include.
Explicit arrays default to zero energies and occupations and symmetry label
``"A"``. :func:`~forte2.orbitals.molden.molden_string` accepts the same orbital
inputs and returns the file contents as a string.

Molden export supports real spatial orbitals with spherical shells through g.
By default, shells beyond g raise an error. Use
``ignore_unsupported_shells=True`` to omit those shells and their MO coefficient
rows. The resulting orbitals may lose normalization. For the writer's options
and compatibility notes, see `write_molden` and `molden_string`.

Cube files
----------

Use the function `write_orbital_cubes` to write orbitals in the cube format.
The function supports spatial orbitals and two-component spinors.

.. code-block:: python

    from forte2.orbitals import write_orbital_cubes

    write_orbital_cubes(rhf.system, rhf.mos.C[0], indices=[0, 1, 2], filepath="cubes")

``indices`` selects zero-based columns of the supplied coefficient matrix.
Omitting it writes every orbital in that matrix. The example creates
``cubes/orbital_0.cube``, ``cubes/orbital_1.cube``, and
``cubes/orbital_2.cube``. Use ``prefix`` to give the files another name. For a
UHF calculation, use ``uhf.mos.C[0]`` for alpha orbitals and ``uhf.mos.C[1]``
for beta orbitals, with a different prefix for each set.

``spacing`` sets the grid step and defaults to 0.2 bohr. ``padding`` extends
the grid beyond the atomic positions on each side and defaults to 4.0 bohr.
Smaller spacing gives a finer grid with larger files. Increase padding when
the orbital extends beyond the grid.

Two-component cube files
~~~~~~~~~~~~~~~~~~~~~~~~

For a two-component method, pass its spinor coefficient matrix and
request ``format="2ccube"`` to keep both complex spin components in one file
per orbital. This format can be visualized with VibeMol.
See :doc:`relativistic` for examples of two-component methods.

.. code-block:: python

    write_orbital_cubes(
        spinor_method.system,
        spinor_method.mos.C[0],
        indices=[0],
        filepath="spinor_cubes",
        format="2ccube",
    )

The ``.2ccube`` file stores four real-valued datasets in order of alpha real,
alpha imaginary, beta real, and beta imaginary. The default ``format="cube"``
writes the magnitudes of the alpha and beta components to separate
``_a.cube`` and ``_b.cube`` files. Pass ``format=("cube", "2ccube")`` to
write both layouts.
