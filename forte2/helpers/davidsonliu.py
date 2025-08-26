from forte2.helpers import logger
import numpy as np
from numpy.linalg import eigh, qr, norm


class DavidsonLiuSolver:
    """
    Davidson-Liu solver for iterative diagonalization of Hermitian matrices.

    Parameters
    ----------
    size : int
        Dimension of the matrix / number of basis vectors.
    nroot : int
        Number of roots to find.
    basis_per_root : int, optional, default=4
        Number of basis vectors to keep per root.
    collapse_per_root : int, optional, default=2
        Number of vectors to collapse to per root.
    maxiter : int, optional, default=100
        Maximum number of iterations to perform.
    e_tol : float, optional, default=1e-12
        Convergence tolerance for eigenvalues.
    r_tol : float, optional, default=1e-6
        Convergence tolerance for residuals.
    eta : float, optional
        Target eigenvalue shift for sorting eigenpairs.
        If None, no shift is applied.
    log_level : int, optional, default=logger.get_verbosity_level()
        Logging level for output messages.

    Attributes
    ----------
    iter : int
        Current iteration number.
    converged : bool
        Whether the solver has converged.
    """

    def __init__(
        self,
        size: int,
        nroot: int,
        basis_per_root: int = 4,
        collapse_per_root: int = 2,
        maxiter: int = 100,
        e_tol: float = 1e-12,
        r_tol: float = 1e-6,
        eta: float | None = None,
        log_level: int = logger.get_verbosity_level(),
        dtype: type = float,
    ):
        # size of the space
        self.size = size
        # number of roots to find
        self.nroot = nroot
        # number of vectors to collapse per root
        self.collapse_per_root = collapse_per_root
        # basis size per root
        self.basis_per_root = basis_per_root
        # maximum number of iterations
        self.maxiter = maxiter
        # convergence tolerance for eigenvalues
        self.e_tol = e_tol
        # convergence tolerance for residuals
        self.r_tol = r_tol
        # eigenvalue target shift
        self.eta = eta
        # logging level
        self.log_level = log_level
        # data type
        self.dtype = dtype

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
        if basis_per_root < collapse_per_root + 1:
            raise ValueError(
                f"Davidson-Liu solver: basis_per_root ({basis_per_root}) must be greater than or equal to collapse_per_root + 1 ({collapse_per_root + 1})."
            )

        # fixed subspace and collapse dims
        self.collapse_size = min(collapse_per_root * nroot, size)
        self.max_subspace_size = min(basis_per_root * nroot, size)

        # allocate all arrays as (size, subspace_size) so each column is a vector
        self.b = np.zeros((size, self.max_subspace_size), dtype=self.dtype)  # basis
        self.sigma = np.zeros(
            (size, self.max_subspace_size), dtype=self.dtype
        )  # H·basis
        self.r = np.zeros((size, self.max_subspace_size), dtype=self.dtype)  # residuals
        self.h_diag = None  # matrix diagonal, shape (size,)

        ## subspace Hamiltonian and eigenpairs
        self.G = np.zeros(
            (self.max_subspace_size, self.max_subspace_size), dtype=self.dtype
        )
        # eigenpairs of G
        self.alpha = np.zeros_like(self.G)
        self.lam = np.zeros(self.max_subspace_size, dtype=self.dtype)
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
        Add the function that builds the matrix-vector product.

        Parameters
        ----------
        sigma_builder : callable
            This function should have signature ``sigma_builder(basis_block, sigma_block) -> None``,
            where ``basis_block.shape == (size, m)`` and ``sigma_block.shape == (size, m)``.
            and it modifies ``sigma_block`` in place, performing the operation
            ``sigma_block = matrix @ basis_block``.
        """
        self._build_sigma = sigma_builder

    def add_h_diag(self, h_diag):
        arr = np.asarray(h_diag, dtype=self.dtype)
        if arr.shape != (self.size,):
            raise ValueError("h_diag must be shape (size,)")
        self.h_diag = arr.copy()

    def add_guesses(self, guesses):
        """
        Add initial guesses for the eigenvectors.

        Parameters
        ----------
        guesses: NDArray
            ``guesses.shape == (size, n_guess)``, with n_guess between 1 and subspace_size.
        """
        G = np.asarray(guesses, dtype=self.dtype)
        if G.ndim != 2 or G.shape[0] != self.size:
            raise ValueError("guesses must be shape (size, n_guess)")
        self._guesses = G

    def add_project_out(self, project_out):
        """
        project_out: list of arrays each shape (size,)
        """
        self._proj_out = [np.asarray(v, dtype=self.dtype) for v in project_out]

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
                    G -= np.outer(v, coeffs.conj())

            # orthonormalize via QR
            Q, _ = qr(G, mode="reduced")
            self.b[:, : Q.shape[1]] = Q
            self.basis_size = Q.shape[1]

            self.lam_old[:] = 0.0

        # check orthonormality of the initial basis
        self.orthonormality_check(self.b[:, : self.basis_size])

        self.iter = 0
        self.converged = False

        logger.log(
            ("=" * 64)
            + f"\nIter                 ⟨E⟩             max(ΔE)        max(r) basis\n"
            + ("-" * 64),
            self.log_level,
        )

        for self.iter in range(self.maxiter):
            # 2. compute new sigma block if needed
            m_new = self.basis_size - self.sigma_size
            if m_new > 0:
                Bblock = self.b[:, self.sigma_size : self.basis_size]
                Sblock = self.sigma[:, self.sigma_size : self.basis_size]
                self._build_sigma(Bblock, Sblock)
                # self.sigma[:, self.sigma_size : self.basis_size] = Sblock
                self.sigma_size = self.basis_size

            # 3. form and diagonalize subspace Hamiltonian
            Bblk = self.b[:, : self.basis_size]
            Sblk = self.sigma[:, : self.basis_size]
            Gm = Bblk.T.conj() @ Sblk
            Gm = 0.5 * (Gm + Gm.T.conj())  # Hermitize
            lam, alpha = eigh(Gm)

            # sort eigenpair around user-specified shift
            if self.eta is not None:
                idx = np.argsort(np.abs(lam - self.eta))
                lam = lam[idx]
                alpha = alpha[:, idx]

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
                self.converged = True
                break
            self.lam_old[: self.nroot] = lamr

            # 5. collapse if we cannot add more vectors
            if self.basis_size + self.nroot > self.max_subspace_size:
                self._collapse(alpha)

            # 6. add correction vectors
            # 6a. orthogonalize residuals against current basis
            R0 = self.r[:, : self.nroot]
            # subtract projection onto existing basis
            R0 -= Bblk @ (Bblk.T.conj() @ R0)

            # 6b. project out undesirable vectors if provided
            if hasattr(self, "_proj_out"):
                for v in self._proj_out:
                    # v shape (size,)
                    coeffs = v @ R0  # shape (nroot,)
                    R0 -= np.outer(v, coeffs.conj())

            # write back the cleaned residuals
            self.r[:, : self.nroot] = R0

            to_add = min(self.nroot, self.max_subspace_size - self.basis_size)
            if to_add == 0:
                self.converged = True
                break

            # attempt to add new basis vectors from R0
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
                temp = temp.astype(self.dtype)
                if hasattr(self, "_proj_out"):
                    for v in self._proj_out:
                        # v shape (size,)
                        coeffs = v @ temp  # shape (nroot,)
                        temp -= np.outer(v, coeffs.conj())

                added2 = self.add_rows_and_orthonormalize(
                    self.b[:, : self.basis_size],
                    temp,
                    self.b[:, self.basis_size :],
                    # self.b, self.basis_size, temp, missing
                )
                self.basis_size += added2
                msg = f" <- +{added2} random"
            logger.log(
                f"{self.iter:4d}  {avg_e:18.12f}  {max_de:18.12f}  {max_r:12.9f}  {self.basis_size:4d} {msg}",
                self.log_level,
            )

        logger.log(
            ("=" * 64),
            self.log_level,
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
                max_overlap = np.abs(A_existing.T.conj() @ v).max()
            if n_new > 0:
                max_overlap = max(max_overlap, np.abs(A_new.T.conj() @ v).max())

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
        if not np.allclose(b.conj().T @ b, np.eye(b.shape[1]), atol=1e-12):
            logger.log_warning(f"{msg}")
            logger.log_warning(f"S = {b.conj().T @ b}")
            raise ValueError(msg)

    def _print_information(self):
        logger.log("\nDavidson-Liu solver configuration:", self.log_level)
        logger.log(f"  Size of the space:        {self.size}", self.log_level)
        logger.log(f"  Number of roots:          {self.nroot}", self.log_level)
        logger.log(
            f"  Maximum size of subspace: {self.max_subspace_size}", self.log_level
        )
        logger.log(f"  Size of collapsed space:  {self.collapse_size}", self.log_level)
        logger.log(f"  Energy convergence:       {self.e_tol}", self.log_level)
        logger.log(f"  Residual convergence:     {self.r_tol}", self.log_level)
        if self.eta is not None:
            logger.log(f"  Target eigenval shift:    {self.eta}", self.log_level)
        logger.log(f"  Maximum iterations:       {self.maxiter}\n", self.log_level)
