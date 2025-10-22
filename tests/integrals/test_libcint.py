from forte2 import System, ints
from forte2.ints.libcint_integrals import int1e_ovlp

def test_libcint_overlap():
    xyz = """
    Li 0 0 0
    Li 0 0 1
    """
    system = System(xyz, basis_set="sto-3g")
    s_ref = ints.overlap(system.basis)
    print(s_ref)

    s = int1e_ovlp(system)
    print(s)

test_libcint_overlap()