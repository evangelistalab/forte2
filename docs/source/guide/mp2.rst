Density-fitted MP2
==================

Forte2 provides density-fitted second-order Møller--Plesset methods for RHF,
ROHF, and UHF references through ``RMP2``, ``ROMP2``, and ``UMP2``. These
methods require a nonrelativistic reference and an auxiliary basis set. They
currently correlate all occupied and virtual orbitals.

Basic usage
-----------

The following example runs RMP2 after an RHF calculation::

    from forte2 import System
    from forte2.mp import RMP2
    from forte2.scf import RHF

    system = System(
        xyz="""
        H  0.0  0.0  0.0
        H  0.0  0.0  0.74
        """,
        basis_set="cc-pVDZ",
        auxiliary_basis_set="cc-pVTZ-JKFIT",
    )

    scf = RHF(charge=0)(system)
    mp2 = RMP2()(scf)
    energy = mp2.run()

Use ``ROMP2`` with an ``ROHF`` reference and ``UMP2`` with a ``UHF``
reference. The total energy is available as either ``mp2.E`` or
``mp2.E_total``, and the correlation energy is ``mp2.E_corr``.

Amplitude storage and density matrices
--------------------------------------

``store_t2=False`` is the default. It avoids retaining the rank-four doubles
amplitudes after evaluating the energy. Density matrices remain available:
Forte2 rebuilds the required amplitudes locally and releases them after the
requested density is formed. A full two-particle density matrix still has its
inherent rank-four memory cost.

For RMP2, request spin-free MO-basis densities after ``run``::

    gamma1 = mp2.make_1rdm()
    gamma2 = mp2.make_2rdm(gamma1)

ROMP2 and UMP2 return spin-resolved MO-basis blocks by default::

    gamma1_a, gamma1_b = mp2.make_1rdm()
    gamma2_aa, gamma2_ab, gamma2_bb = mp2.make_2rdm(
        (gamma1_a, gamma1_b)
    )

For a genuine UHF reference, the alpha and beta blocks are expressed in
different MO bases defined by :math:`C^\alpha` and :math:`C^\beta`. Set
``ao_repr=True`` to transform each block with its corresponding coefficients::

    gamma1_ao_a, gamma1_ao_b = mp2.make_1rdm_sd(ao_repr=True)
    gamma2_ao_aa, gamma2_ao_ab, gamma2_ao_bb = mp2.make_2rdm_sd(
        (gamma1_a, gamma1_b), ao_repr=True
    )

The spin-summed helpers perform these separate transformations before adding
the blocks and therefore return AO-basis tensors::

    gamma1_ao = mp2.make_1rdm_sf()
    gamma2_ao = mp2.make_2rdm_sf((gamma1_a, gamma1_b))
