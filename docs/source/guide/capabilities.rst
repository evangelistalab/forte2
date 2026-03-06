List of Forte2 capabilities
===========================

Forte2 is still under active development, with a focus on multi-reference and relativistic methods.
Here is a (non-exhaustive) list of the current capabilities of Forte2:

* Support for density-fitted or Cholesky-decomposed integrals (no support for conventional, 4-index integrals)
* Support for arbitrary model Hamiltonians
* Support for utilizing molecular symmetry (largest Abelian subgroup) at the post-Hartree-Fock level
* Support for the finite (Gaussian-distributed) nuclear charge model [1]_
* Scalar and vector relativistic Hamiltonians
  
  * Spin-free 1-electron exact two-component (sf-1eX2C) [2]_
  * Spin-orbit 1-electron exact two-component (so-1eX2C) [2]_
  * Various empirical scaling schemes to approximate two-electron spin-orbit couplings ("Boettger factors") [3]_

* Flavors of Hartree-Fock theory
  
  * Restricted Hartree-Fock (RHF)
  * Restricted Open-Shell Hartree-Fock (ROHF)
  * Unrestricted Hartree-Fock (UHF)
  * Constrained unrestricted Hartree-Fock (CUHF)
  * Generalized Hartree-Fock (GHF)

* A flexible configuration interaction (CI) module
  
  * Spin-adapted CI (CSF basis) for non-relativistic Hamiltonians
  * Support for generalized active spaces (GAS) / occupation-restricted multiple active spaces (ORMAS)
  * Two-component CI for relativistic Hamiltonians
  
* Multi-configuration self-consistent field (MCSCF) methods
  
  * Support for CAS-SCF and GAS-SCF/ORMAS-SCF, with state-averaging
  * Two-component CAS/GAS/ORMAS-SCF
  * Atomic valence active space (AVAS) active space selection (support for both one- and two-component Hartree-Fock) [4]_

* Multi-reference driven similarity renormalization group (MR-DSRG) methods
  
  * Non-relativistic DSRG-MRPT2 with reference relaxation and state-averaging [7]_
  * Two-component relativistic DSRG-MRPT2 with reference relaxation and state-averaging
  
* Various orbital manipulation routines
  
  * Zeroth order active space embedding theory (ASET(0)) [5]_
  * Intrinsic atomic orbitals (IAO) [6]_
  * Intrinsic bond orbitals (IBO) [6]_
  * A cube file generator for visualizing molecular orbitals
  

References
----------
.. [1] Visscher, L.; Dyall, K. G. Dirac-Fock Atomic Electronic Structure Calculations Using Different Nuclear Charge Distributions. Atomic Data and Nuclear Data Tables 1997, 67 (2), 207-224. https://doi.org/10.1006/adnd.1997.0751.
.. [2] Liu, W.; Peng, D. Exact Two-Component Hamiltonians Revisited. The Journal of Chemical Physics 2009, 131 (3), 031104. https://doi.org/10.1063/1.3159445.
.. [3] Ehrman, J.; Martinez-Baez, E.; Jenkins, A. J.; Li, X. Improving One-Electron Exact-Two-Component Relativistic Methods with the Dirac-Coulomb-Breit-Parameterized Effective Spin-Orbit Coupling. J. Chem. Theory Comput. 2023, 19 (17), 5785-5790. https://doi.org/10.1021/acs.jctc.3c00479.
.. [4] Sayfutyarova, E. R.; Sun, Q.; Chan, G. K.-L.; Knizia, G. Automated Construction of Molecular Active Spaces from Atomic Valence Orbitals. J. Chem. Theory Comput. 2017, 13 (9), 4063-4078. https://doi.org/10.1021/acs.jctc.7b00128.
.. [5] He, N.; Evangelista, F. A. A Zeroth-Order Active-Space Frozen-Orbital Embedding Scheme for Multireference Calculations. The Journal of Chemical Physics 2020, 152 (9), 094107. https://doi.org/10.1063/1.5142481.
.. [6] Knizia, G. Intrinsic Atomic Orbitals: An Unbiased Bridge between Quantum Theory and Chemical Concepts. J. Chem. Theory Comput. 2013, 9 (11), 4834-4843. https://doi.org/10.1021/ct400687b.
.. [7] Li, C.; Evangelista, F. A. Multireference Driven Similarity Renormalization Group: A Second-Order Perturbative Analysis. J. Chem. Theory Comput. 2015, 11 (5), 2097-2108. https://doi.org/10.1021/acs.jctc.5b00134.
