import forte2

xyz = """

H  0.0 0.0 0.0
Li 0.0 0.0 3.0
"""

system = forte2.System(xyz=xyz, basis="sto-6g")
print(system)

forte2.overlap(system.basis)

# # C++ (calls to libint2)
# aoints = fortecore.AOIntegrals(system.shells, True)
# print(aoints)

# # C++ (calls to libint2)
# aoints.overlap()
# # T = aoints.ints("kinetic")
# # V = aoints.ints("nuclear")
# # [Dx,Dy,Dz] = aoints.ints("dipole") ? tuple(tensors), 3D tensor?

# # DF-based
# # J, K = aoints.jk(D)

# # J, K = aoints.jk(C[0:10])

# # AO to MO ?
