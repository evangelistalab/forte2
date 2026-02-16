Basic usage
=================

Forte2 is written in C++ and Python. 
Most core functionalities are implemented in Python, with only the most demanding parts implemented in C++. 
The C++ code is exposed to Python through the use of nanobind, which allows for a seamless integration between the two languages.


Forte2 uses the `functional composition <https://en.wikipedia.org/wiki/Function_composition_(computer_science)>`_ style (like the `TensorFlow functional API <https://www.tensorflow.org/guide/keras/functional_api>`_) for most quantum chemical methods, with the following programmatic flow:

>>> rhf = forte2.scf.RHF(charge=0)(system)
>>> ci = forte2.ci.CI(state=state, orbitals=[...], ...)(rhf)
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
