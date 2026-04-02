import numpy as np


class MP2MCASolverLike:
    def __init__(
        self,
        gamma1_sf: np.ndarray,
        lambda2_sf: np.ndarray,
        U: np.ndarray | None = None,
        orbital_indices=None,
    ):
        norb = gamma1_sf.shape[0]

        if U is not None:
            # Rotate 1-RDM and 2-cumulant ONCE into NO basis
            Γ1 = U.T @ gamma1_sf @ U
            λsf_no = np.einsum(
                "pqrs,pi,qj,rk,sl->ijkl", lambda2_sf, U, U, U, U, optimize=True
            )

            # Map spin-free cumulant into λab only (your convention)
            self.λaa = np.zeros((norb, norb, norb, norb), dtype=lambda2_sf.dtype)
            self.λbb = np.zeros_like(self.λaa)
            self.λab = 0.5 * λsf_no

            self.Γ1 = Γ1
            self.orbital_indices = list(range(norb))  # NO labels
            self.U_no = U

        else:
            self.Γ1 = gamma1_sf
            self.λaa = np.zeros((norb, norb, norb, norb), dtype=lambda2_sf.dtype)
            self.λbb = np.zeros_like(self.λaa)
            self.λab = 0.5 * lambda2_sf

            self.orbital_indices = (
                list(orbital_indices)
                if orbital_indices is not None
                else list(range(norb))
            )
