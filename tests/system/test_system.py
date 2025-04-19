import forte2
import numpy as np

xyz = """

H  0.0 0.0 0.0
"""

system = forte2.System(xyz=xyz, basis="cc-pvdz")
system_large_basis = forte2.System(xyz=xyz, basis="cc-pvdz")
print(system)

S = forte2.overlap(system.basis)
T = forte2.kinetic(system.basis)
V = forte2.nuclear(system.basis, system.atoms)
H = T + V

# Solve the generalized eigenvalue problem H C = S C ε
from scipy.linalg import eigh
from numpy import isclose

ε, _ = eigh(H, S)
print("ε", ε)
assert isclose(ε[0], -0.4992784, atol=1e-7)


M1 = forte2.emultipole1(system.basis)
print("S", S)
print("M", M1)
print(np.linalg.norm(S - M1[0]))

M2 = forte2.emultipole2(system.basis)
# print("M2", M2)
for i in range(4):
    print(np.linalg.norm(M1[i] - M2[i]))

M3 = forte2.emultipole3(system.basis)
# print("M3", M3)
for i in range(10):
    print(np.linalg.norm(M2[i] - M3[i]))

opVop = forte2.opVop(system.basis, system.atoms)
print("opVop", opVop)

# mport numpy as np
# import scipy.linalg
# import scipy.linalg as linalg
# import scipy.linalg.eig as eig


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
