import numpy as np
from numpy.linalg import eigh, qr, norm


class DavidsonLiuSolver:
    def __init__(
        self,
        size: int,
        nroot: int,
        collapse_per_root: int = 2,
        subspace_per_root: int = 4,
        maxiter: int = 100,
        e_tol: float = 1e-12,
        r_tol: float = 1e-6,
        disable_collapse: bool = False,
    ):
        # size of the space
        self.size = size
        # number of roots to find
        self.nroot = nroot
        # number of vectors to collapse per root
        self.collapse_per_root = collapse_per_root
        # subspace size per root
        self.subspace_per_root = subspace_per_root
        # disable collapse if True
        self.disable_collapse = disable_collapse
        # maximum number of iterations
        self.maxiter = maxiter
        # convergence tolerance for eigenvalues
        self.e_tol = e_tol
        # convergence tolerance for residuals
        self.r_tol = r_tol

        # sanity checks
        if size <= 0:
            raise ValueError(
                "Davidson-Liu solver called with space of dimension smaller than 1."
            )
        if nroot <= 0:
            raise ValueError("Davidson-Liu solver called with zero roots.")
        if collapse_per_root < 1:
            raise ValueError(
                f"Davidson-Liu solver: collapse_per_root ({collapse_per_root}) must be greater than or equal to 1."
            )
        if subspace_per_root < collapse_per_root + 1:
            raise ValueError(
                f"Davidson-Liu solver: subspace_per_root ({subspace_per_root}) must be greater than or equal to collapse_per_root + 1 ({collapse_per_root + 1})."
            )

        # fixed subspace and collapse dims
        self.collapse_size = min(collapse_per_root * nroot, size)
        self.max_subspace_size = min(subspace_per_root * nroot, size)

        # allocate all arrays as (size, subspace_size) so each column is a vector
        self.b = np.zeros((size, self.max_subspace_size))  # basis
        self.sigma = np.zeros((size, self.max_subspace_size))  # H·basis
        self.r = np.zeros((size, self.max_subspace_size))  # residuals
        self.h_diag = None  # matrix diagonal, shape (size,)

        ## subspace Hamiltonian and eigenpairs
        self.G = np.zeros((self.max_subspace_size, self.max_subspace_size))
        # eigenpairs of G
        self.alpha = np.zeros_like(self.G)
        self.lam = np.zeros(self.max_subspace_size)
        self.lam_old = np.zeros_like(self.lam)

        ## configuration parameters
        # The threshold used to discard correction vectors
        self.schmidt_discard_threshold = 1e-7
        # The threshold used to guarantee orthogonality among the roots
        self.schmidt_orthogonality_threshold = 1e-12

        ## bookkeeping
        # size of the basis block
        self.basis_size = 0
        # size of the sigma block
        self.sigma_size = 0

        ## function to build sigma block
        self._build_sigma = None
        self._executed = False

    def add_sigma_builder(self, sigma_builder):
        """
        sigma_builder should have signature
            sigma_block = sigma_builder(basis_block)
        where
            basis_block.shape = (size, m)
        and it returns
            sigma_block.shape = (size, m).
        """
        self._build_sigma = sigma_builder

    def add_h_diag(self, h_diag):
        arr = np.asarray(h_diag, float)
        if arr.shape != (self.size,):
            raise ValueError("h_diag must be shape (size,)")
        self.h_diag = arr.copy()

    def add_guesses(self, guesses):
        """
        guesses: array shape (size, n_guess), with n_guess between 1 and subspace_size.
        """
        G = np.asarray(guesses, float)
        if G.ndim != 2 or G.shape[0] != self.size:
            raise ValueError("guesses must be shape (size, n_guess)")
        self._guesses = G

    def add_project_out(self, project_out):
        """
        project_out: list of arrays each shape (size,)
        """
        self._proj_out = [np.asarray(v, float) for v in project_out]

    def solve(self):

        self._print_information()

        # 0. sanity checks
        if self._build_sigma is None:
            raise ValueError("Sigma builder function is not set.")
        if self.h_diag is None:
            raise ValueError("Hamiltonian diagonal is not set.")

        if not self._executed:
            # 1. setup guesses
            if not hasattr(self, "_guesses"):
                # random if no guesses
                rng = np.random.default_rng()
                G = rng.uniform(-1, 1, size=(self.size, self.nroot))
            else:
                G = self._guesses

            if hasattr(self, "_proj_out"):
                for v in self._proj_out:
                    # v shape (size,)
                    coeffs = v @ G  # shape (nroot,)
                    G -= np.outer(v, coeffs)

            # orthonormalize via QR
            Q, _ = qr(G, mode="reduced")
            self.b[:, : Q.shape[1]] = Q
            self.basis_size = Q.shape[1]

            self.lam_old[:] = 0.0

        # check orthonormality of the initial basis
        self.orthonormality_check(self.b[:, : self.basis_size])

        for it in range(self.maxiter):
            # 2. compute new sigma block if needed
            m_new = self.basis_size - self.sigma_size
            if m_new > 0:
                Bblock = self.b[:, self.sigma_size : self.basis_size]
                Sblock = self._build_sigma(Bblock)
                self.sigma[:, self.sigma_size : self.basis_size] = Sblock
                self.sigma_size = self.basis_size

            # 3. form and diagonalize subspace Hamiltonian
            Bblk = self.b[:, : self.basis_size]
            Sblk = self.sigma[:, : self.basis_size]
            Gm = Bblk.T @ Sblk
            Gm = 0.5 * (Gm + Gm.T)  # symmetrize
            lam, alpha = eigh(Gm)
            self.lam[: self.basis_size] = lam
            self.alpha[: self.basis_size, : self.basis_size] = alpha

            # 4. residuals for first nroot
            ar = alpha[:, : self.nroot]  # (basis_size, nroot)
            lamr = lam[: self.nroot]  # (nroot,)

            # build residual vectors
            Balpha = Bblk @ ar  # (size, nroot)
            Salpha = Sblk @ ar  # (size, nroot)
            R = Salpha - Balpha * lamr[np.newaxis, :]

            rnorms = norm(R, axis=0)

            # precondition
            denom = lamr[np.newaxis, :] - self.h_diag[:, np.newaxis]
            mask = np.abs(denom) > 1e-6
            R = np.where(mask, R / denom, 0.0)
            self.r[:, : self.nroot] = R

            # norms & convergence
            avg_e = lamr.mean()
            max_de = np.max(np.abs(lamr - self.lam_old[: self.nroot]))
            max_r = rnorms.max()

            conv_e = np.all(np.abs(lamr - self.lam_old[: self.nroot]) < self.e_tol)
            conv_r = np.all(rnorms < self.r_tol)
            if (conv_e and conv_r) or (self.basis_size == self.size):
                break
            self.lam_old[: self.nroot] = lamr

            # ——— 5. collapse if needed ———
            if not self.disable_collapse and (
                self.basis_size + self.nroot > self.max_subspace_size
            ):
                self._collapse(alpha)

            # ——— 6. add correction vectors ———
            # 6a. orthogonalize residuals against current basis
            R0 = self.r[:, : self.nroot]
            # subtract projection onto existing basis
            R0 -= Bblk @ (Bblk.T @ R0)

            # 6b. project out undesirable vectors if provided
            if hasattr(self, "_proj_out"):
                for v in self._proj_out:
                    # v shape (size,)
                    coeffs = v @ R0  # shape (nroot,)
                    R0 -= np.outer(v, coeffs)

            # write back the cleaned residuals
            self.r[:, : self.nroot] = R0

            to_add = min(self.nroot, self.max_subspace_size - self.basis_size)
            if to_add == 0:
                break

            # attempt to add directly from R0
            added = self.add_rows_and_orthonormalize(
                self.b[:, : self.basis_size],
                R0[:, :to_add],
                self.b[:, self.basis_size :],
                # self.b, self.basis_size, R0, to_add
            )
            self.basis_size += added
            # if we couldn't add enough, fill with random orthogonal vectors
            msg = ""
            if added < to_add:
                missing = to_add - added
                temp = np.random.default_rng().uniform(
                    -1.0, 1.0, size=(self.size, missing)
                )
                if hasattr(self, "_proj_out"):
                    for v in self._proj_out:
                        # v shape (size,)
                        coeffs = v @ temp  # shape (nroot,)
                        temp -= np.outer(v, coeffs)

                added2 = self.add_rows_and_orthonormalize(
                    self.b[:, : self.basis_size],
                    temp,
                    self.b[:, self.basis_size :],
                    # self.b, self.basis_size, temp, missing
                )
                self.basis_size += added2
                msg = f" <- +{added2} random"
            print(
                f"{it:4d}  ⟨E⟩ ={avg_e:18.12f}  max(ΔE) ={max_de:18.12f}  max(r) ={max_r:12.9f}  basis = {self.basis_size:4d} {msg}"
            )

        # compute final eigenpairs
        lamr = self.lam[: self.nroot]
        evecs = (
            self.b[:, : self.basis_size] @ self.alpha[: self.basis_size, : self.nroot]
        )

        # orthonormalize final evecs
        # Qf, _ = qr(evecs, mode="reduced")
        self.basis_size = self.nroot
        self.b[:, : self.nroot] = evecs
        self.sigma_size = 0
        self._executed = True
        return lamr, evecs

    def _collapse(self, alpha):
        """
        collapse both b and sigma down to collapse_size using alpha:
            new_b = b[:, :basis_size] @ alpha[:, :collapse_size]
        """
        k = self.collapse_size
        Bblk = self.b[:, : self.basis_size]
        self.orthonormality_check(
            Bblk,
            f"\nBefore collapse: Checking orthonormality of Bblk with size {Bblk.shape[1]}",
        )
        Sblk = self.sigma[:, : self.basis_size]
        newB = Bblk @ alpha[:, :k]
        newS = Sblk @ alpha[:, :k]
        # overwrite
        self.b[:, :k] = newB
        self.sigma[:, :k] = newS
        self.basis_size = k
        self.sigma_size = k
        self.orthonormality_check(
            newB,
            f"\nAfter collapse: Checking orthonormality of newB with size {newB.shape[1]}",
        )

    # def add_rows_and_orthonormalize(
    #     self, A_existing: np.ndarray, B_candidates: np.ndarray, A_slots: np.ndarray
    # ) -> int:
    #     """
    #     Add candidate vectors from B_candidates into A_slots, orthonormalizing each against
    #     the existing basis in A_existing and previously added vectors. Returns the number
    #     of vectors actually added.
    #     """
    #     added = 0
    #     n_existing = A_existing.shape[1]

    #     # Loop over candidate vectors in B_candidates
    #     for j in range(B_candidates.shape[1]):
    #         # Make a working copy of the j-th column of B_candidates
    #         v = B_candidates[:, j].copy()

    #         # 1) Orthogonalize v against the existing basis A_existing
    #         for i in range(n_existing):
    #             ai = A_existing[:, i]
    #             v -= np.dot(ai, v) * ai

    #         # 2) Orthogonalize against any vectors already added into A_slots
    #         for i in range(added):
    #             si = A_slots[:, i]
    #             v -= np.dot(si, v) * si

    #         # 3) Discard if v projected is too small (below discard threshold)
    #         normv = norm(v)
    #         if normv < self.schmidt_discard_threshold:
    #             continue

    #         # 4) Normalize v
    #         v /= normv

    #         # 5) Check orthogonality to both sets
    #         max_overlap = 0.0
    #         if n_existing > 0:
    #             overlaps_existing = np.abs(A_existing.T @ v)
    #             max_overlap = overlaps_existing.max()
    #         if added > 0:
    #             overlaps_new = np.abs(A_slots[:, :added].T @ v)
    #             max_overlap = max(max_overlap, overlaps_new.max())
    #         if max_overlap > self.schmidt_orthogonality_threshold:
    #             continue

    #         # 6) Accept: store into the next free column of A_slots
    #         A_slots[:, added] = v
    #         added += 1

    #     return added

    def add_rows_and_orthonormalize(
        self, A_existing: np.ndarray, B_candidates: np.ndarray, A_slots: np.ndarray
    ) -> int:
        """
        Add candidate column vectors from B_candidates into A_slots, using iterative orthogonalization per vector.
        Returns the number of columns added.
        """
        n_cand = B_candidates.shape[1]
        cap = A_slots.shape[1]
        if n_cand > cap:
            raise ValueError(f"Cannot add {n_cand} candidates into {cap} slots")

        added = 0
        for j in range(n_cand):
            v0 = B_candidates[:, j].copy()
            slots_filled = A_slots[:, :added]
            success, vnew = self._add_row_and_orthonormalize(
                A_existing, slots_filled, v0
            )
            if success:
                A_slots[:, added] = vnew
                added += 1

        return added

    def _add_row_and_orthonormalize(
        self, A_existing: np.ndarray, A_new: np.ndarray, v: np.ndarray
    ) -> tuple[bool, np.ndarray]:
        """
        Attempt to orthonormalize vector v against the existing basis A_existing
        and any already accepted vectors in A_new (columns). Returns (success, v),
        where v is normalized if success is True.
        """
        n_existing = A_existing.shape[1]
        n_new = A_new.shape[1]
        max_cycles = 10

        for cycle in range(max_cycles):
            # orthogonalize against existing basis
            for i in range(n_existing):
                ai = A_existing[:, i]
                v -= np.dot(ai, v) * ai
            # orthogonalize against new vectors
            for i in range(n_new):
                si = A_new[:, i]
                v -= np.dot(si, v) * si

            # compute norm and discard if too small
            normv = norm(v)
            if normv < self.schmidt_discard_threshold:
                return False, v

            # normalize
            v /= normv

            # compute maximum overlap
            max_overlap = 0.0
            if n_existing > 0:
                max_overlap = np.abs(A_existing.T @ v).max()
            if n_new > 0:
                max_overlap = max(max_overlap, np.abs(A_new.T @ v).max())

            # check normalization and orthogonality
            norm2 = np.dot(v, v)
            if (
                max_overlap < self.schmidt_orthogonality_threshold
                and abs(norm2 - 1.0) < self.schmidt_orthogonality_threshold
            ):
                return True, v

        # failed to orthonormalize within max_cycles
        return False, v

    def orthonormality_check(
        self, b: np.ndarray, msg: str = "Orthonormality check failed."
    ):
        """
        Check if the columns of b are orthonormal.
        """
        if not np.allclose(b.T @ b, np.eye(b.shape[1]), atol=1e-12):
            print(f"{msg}")
            print(f"S = {b.T @ b}")
            raise ValueError(msg)

    def _print_information(self):
        print(f"\nDavidson-Liu solver configuration:")
        print(f"  Size of the space:        {self.size}")
        print(f"  Number of roots:          {self.nroot}")
        print(f"  Collapse size:            {self.collapse_size}")
        print(f"  Maximum subspace size:    {self.max_subspace_size}")
        print(f"  Disable collapse:         {self.disable_collapse}")
        print(f"  Maximum iterations:       {self.maxiter}\n")
