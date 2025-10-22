from forte2 import System, ints

def test_libcint_overlap():
    xyz = """
    Li 0 0 0
    Li 0 0 1
    """
    system = System(xyz, basis_set="cc-pvdz")
    s_ref = ints.overlap(system.basis)
    print(s_ref.shape)

    s = ints.cint_int1e_ovlp_sph(28, system.cint_atm, system.cint_bas, system.cint_env)
    print(s)
    

test_libcint_overlap()