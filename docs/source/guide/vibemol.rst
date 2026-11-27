VibeMol notebook adapter
========================

Use ``forte2.vibemol`` to display an executed Forte2 calculation in a VibeMol
notebook viewer. The adapter needs a viewer with ``sequential_commands = True``,
available in vibemol-widget 0.2.0 and later. Install the widget in the notebook
kernel environment. Forte2 itself can use the data functions independently
of the widget.

.. code-block:: python

    from forte2.vibemol import show_orbitals

    show_orbitals(viewer, mc, indices=[11])
    viewer.snapshot(width=640, height=460).display()

``orbitals_molden`` exports geometry and orbitals. ``atomic_numbers`` and
``forces`` follow the System atom order, with forces in Hartree per bohr.
``state_dipole`` and ``transition_dipole`` return dipoles in atomic units.
Request ``do_transition_dipole=True`` on the active-space driver. Charged
state dipoles are origin dependent and cannot be shown by this adapter.

The ``show_*`` functions return a ``Shown`` object with the last command's
future. The scale returned for force and state dipole arrows can be reused
to compare roots. See the :download:`cyclopropene notebook
<../../../tutorials/vibemol/Tutorial_VibeMol.ipynb>` for a complete example.
