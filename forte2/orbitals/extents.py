from numpy import diag
import forte2
import numpy as np


def orbital_extents(basis, C, indices=None):
    """Compute the average position of the basis functions and their spread."""

    # number of orbitals
    N = C.shape[1]

    # evaluate the multipole moments in the AO basis
    AO_ints = forte2.ints.emultipole2(basis)

    print(f"C.shape = {C.shape}")
    # transform the multipole moments to the MO basis
    MO_ints = [C.T @ M @ C for M in AO_ints]

    # unpack the multipole moments
    S, Mx, My, Mz, Mxx, Mxy, Mxz, Myy, Myz, Mzz = MO_ints

    # compute the average position of the basis functions
    s = diag(S)
    x = diag(Mx)
    y = diag(My)
    z = diag(Mz)
    xx = diag(Mxx) - x**2
    xy = diag(Mxy) - x * y
    xz = diag(Mxz) - x * z
    yy = diag(Myy) - y**2
    yz = diag(Myz) - y * z
    zz = diag(Mzz) - z**2

    # with np.printoptions(precision=6, suppress=True):
    #     print("s", s)
    #     print("x", x)
    #     print("y", y)
    #     print("z", z)
    #     print("xx", xx)
    #     print("xy", xy)
    #     print("xz", xz)
    #     print("yy", yy)
    #     print("yz", yz)
    #     print("zz", zz)

    coords = np.stack((x, y, z), axis=1)
    moments = np.stack((xx, xy, xz, yy, yz, zz), axis=1)
    return coords, moments
