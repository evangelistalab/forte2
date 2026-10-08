Relativistic Hamiltonians
=========================

Forte2 includes scalar and spin-orbit relativistic effects through the exact two-component (X2C)
Hamiltonian. To use it, pass an :class:`forte2.X2CParams` instance to the ``x2c`` option of
``System``. ``X2CParams`` takes three options:

``x2c_type``
    ``"sf"`` for spin-free (scalar) X2C, or ``"so"`` for spin-orbit X2C.

``x2c_model``
    ``"1e"`` (the default) for one-electron X2C, or ``"sap"`` for SAP-X2C, which models the
    two-electron picture-change effects.

``snso_type``
    A screened nuclear spin-orbit scaling of the spin-orbit terms: ``"boettger"``, ``"dc"``,
    ``"dcb"``, or ``"row-dependent"``. Valid only with ``x2c_type="so"`` and ``x2c_model="1e"``.

Forte2 solves the X2C decoupling in the decontracted basis and projects the result onto the basis
set you choose. Basis sets contracted for nonrelativistic calculations describe relativistic core
orbitals poorly, so the examples on this page decontract them with the ``decon-`` prefix.

Spin-free calculations
----------------------

Spin-free X2C changes only the one-electron Hamiltonian, so every one-component method works as it
does without X2C. For example, a CASSCF calculation on the fluorine atom::

    system = forte2.System(
        xyz="F 0 0 0",
        basis_set="decon-cc-pvtz",
        auxiliary_basis_set="cc-pvqz-jkfit",
        x2c=forte2.X2CParams(x2c_type="sf"),
    )
    rohf = forte2.ROHF(charge=0, ms=0.5)(system)
    avas = forte2.AVAS(
        subspace=["F(2s)", "F(2p)"],
        selection_method="separate",
        num_active_docc=3,
    )(rohf)
    doublet = forte2.State(nel=9, multiplicity=2, ms=0.5)
    mc = forte2.MCOptimizer(forte2.CISolver(doublet, nroots=3))(avas)

Spin-orbit coupling on spin-free orbitals
-----------------------------------------

To add spin-orbit coupling after a spin-free calculation, insert a ``SpinorUpcaster``. It converts
one-component orbitals to spinors and, with ``x2c_override``, switches the System to the given X2C
Hamiltonian. Continuing the fluorine example, a two-component CI on the CASSCF orbitals
gives the :sup:`2`\ P\ :sub:`3/2` and :sup:`2`\ P\ :sub:`1/2` levels::

    so_x2c = forte2.X2CParams(x2c_type="so", snso_type="row-dependent")
    upcaster = forte2.SpinorUpcaster(x2c_override=so_x2c)(mc)
    ci_solver = forte2.RelCISolver(
        nel=9, nroots=6, core_orbitals=2, active_orbitals=8
    )
    ci = forte2.CI(ci_solver)(upcaster)
    ci.run()

Orbital indices count spinors, so ``core_orbitals=2`` holds the 1s pair and ``active_orbitals=8``
spans the 2s and 2p spinors.

``SpinorUpcaster`` changes the System in place, so every method bound to that System sees the new
Hamiltonian from then on.

Spin-orbit calculations
-----------------------

Spin-orbit X2C mixes the two spin components, so the orbitals are two-component spinors. Only GHF
accepts a System with ``x2c_type="so"``; RHF, ROHF, UHF, and CUHF raise an error. GHF doesn't use
point-group symmetry: if the System has a point group, GHF logs a warning and runs in C\ :sub:`1`
(see :doc:`symmetry`). The methods that
follow GHF must be two-component: ``RelCISolver`` or ``RelSelectedCISolver`` in a ``CI`` or
``MCOptimizer``, and ``RelDSRG_MRPT2``. For example, the :sup:`3`\ P levels of the carbon atom with
CASSCF and DSRG-MRPT2::

    system = forte2.System(
        xyz="C 0 0 0",
        basis_set="decon-cc-pvtz",
        auxiliary_basis_set="cc-pvqz-jkfit",
        x2c=forte2.X2CParams(x2c_type="so", snso_type="row-dependent"),
    )
    ghf = forte2.GHF(charge=0)(system)
    ci_solver = forte2.RelCISolver(
        nel=6, nroots=9, core_orbitals=2, active_orbitals=8
    )
    mc = forte2.MCOptimizer(ci_solver)(ghf)
    dsrg = forte2.RelDSRG_MRPT2(
        flow_param=0.24, relax_reference="once"
    )(mc)
    dsrg.run()

A two-component solver takes the electron count as ``nel``, or one or more ``RelState`` objects.
Spin isn't a good quantum number once spin-orbit coupling is included, so the ``multiplicity`` and
``ms`` of a ``RelState`` only seed the initial guess. Likewise, GHF's ``ms_guess`` only sets the
occupation of its initial guess.

SAP-X2C
-------

SAP-X2C works with both ``x2c_type`` values and with every SCF method that accepts that type. For
example, spin-orbit SAP-X2C on HBr::

    system = forte2.System(
        xyz="H 0 0 0; Br 0 0 1.4",
        basis_set="cc-pvdz",
        auxiliary_basis_set="cc-pvtz-jkfit",
        x2c=forte2.X2CParams(x2c_type="so", x2c_model="sap"),
    )
    ghf = forte2.GHF(charge=0)(system)

Finite nuclear charges
----------------------

To replace the point nuclei with Gaussian charge distributions, set ``use_gaussian_charges=True``
on ``System``.
