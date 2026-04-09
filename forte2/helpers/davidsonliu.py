from dataclasses import dataclass, field
import time

import numpy as np
from numpy.linalg import eigh, qr, norm

from forte2.helpers import logger
from forte2.helpers.table import AsciiTable
from forte2.base_classes.params import DavidsonLiuParams


@dataclass
class DavidsonLiuSolver:
    """
    Davidson-Liu solver for iterative diagonalization of Hermitian matrices.

    Parameters
    ----------
    size : int
        Dimension of the matrix / number of basis vectors.
    nroot : int
        Number of roots to find.
    davidson_liu_params : DavidsonLiuParams, optional
        Parameters for the Davidson-Liu solver.
    energy_shift : float, optional
        Target eigenvalue shift for sorting eigenpairs.
        If None, no shift is applied.
    log_level : int, optional, default=logger.get_verbosity_level()
        Logging level for output messages.
    dtype : type, optional, default=np.float64
        Data type of the matrix to diagonalize. Must be float or complex.

    Attributes
    ----------
    iter : int
        Current iteration number.
    converged : bool
        Whether the solver has converged.
    """

    size: int
    nroot: int
    davidson_liu_params: DavidsonLiuParams = field(default_factory=DavidsonLiuParams)
    energy_shift: float = field(default=None)
    log_level: int = field(default=logger.get_verbosity_level())
    dtype: type = field(default=np.float64)

    def __post_init__(self):
        self.collapse_per_root = self.davidson_liu_params.collapse_per_root
        # basis size per root
        self.basis_per_root = self.davidson_liu_params.basis_per_root
        # maximum number of iterations
        self.maxiter = self.davidson_liu_params.maxiter
        # convergence tolerance for eigenvalues
        self.e_tol = self.davidson_liu_params.e_tol
        # convergence tolerance for residuals
        self.r_tol = self.davidson_liu_params.r_tol
        # sanity checks
        if self.size <= 0:
            raise ValueError(
                "Davidson-Liu solver called with space of dimension smaller than 1."
            )
        if self.nroot <= 0:
            raise ValueError("Davidson-Liu solver called with zero roots.")

        assert np.issubdtype(self.dtype, np.floating) or np.issubdtype(
            self.dtype, np.complexfloating
        ), "dtype must be a float or complex type"

        # fixed subspace and collapse dims
        self.collapse_size = min(self.collapse_per_root * self.nroot, self.size)
        self.max_subspace_size = min(self.basis_per_root * self.nroot, self.size)

        # allocate all arrays as (size, subspace_size) so each column is a vector
        self.b = np.zeros(
            (self.size, self.max_subspace_size), dtype=self.dtype
        )  # basis
        self.sigma = np.zeros(
            (self.size, self.max_subspace_size), dtype=self.dtype
        )  # H·basis
        self.r = np.zeros(
            (self.size, self.max_subspace_size), dtype=self.dtype
        )  # residuals
        self.h_diag = None  # matrix diagonal, shape (self.size,)

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
        self.schmidt_discard_threshold = 1e-8
        # The threshold used to guarantee orthogonality among the roots
        self.schmidt_orthogonality_threshold = 1e-12
        # The threshold used in the Davidson preconditioner denominator to avoid division by small numbers
        self.preconditioner_denom_threshold = 1e-8

        ## bookkeeping
        # size of the basis block
        self.basis_size = 0
        # size of the sigma block
        self.sigma_size = 0

        ## function to build sigma block
        self._build_sigma = None
        self._executed = False

        ## random number generator
        self._rng = np.random.default_rng()

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
        arr = np.asarray(h_diag, self.dtype)
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
        G = np.asarray(guesses, self.dtype)
        if G.ndim != 2 or G.shape[0] != self.size:
            raise ValueError("guesses must be shape (size, n_guess)")
        self._guesses = G

    def add_project_out(self, project_out):
        """
        Add vectors to project out during the solve.

        Parameters
        ----------
        project_out: list of arrays
            Each array should have the same dtype and shape, but can have lengths less than or equal to `size`.
            The vectors will be orthogonalized and stored as rows for projection during the solve.
            If the lengths are less than `size`, they will be considered to be zero-padded to `size` during projection.
        """
        if len(project_out) == 0:
            return
        for v in project_out:
            if len(v) > self.size:
                raise ValueError(
                    f"Each project_out vector must have length at most {self.size}, but got {len(v)}"
                )
        # orthogonalize and store the project_out vectors as rows for easier projection later
        A = np.stack(project_out, axis=1)  # shape (size, n_proj)
        # orthogonalize via QR
        Q, _ = qr(A, mode="reduced")
        self._proj_out = np.ascontiguousarray(Q.T)  # shape (n_proj, size)

    def do_project_out(self, A):
        if not hasattr(self, "_proj_out"):
            return
        proj_size = self._proj_out.shape[1]
        for u in self._proj_out:
            # u shape (size,)
            # A -> (1 - u u.H) A
            v = u.conj() @ A[:proj_size]
            A[:proj_size] -= u[:, np.newaxis] * v

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
                if np.issubdtype(self.dtype, np.complexfloating):
                    # uniform distribution on unit disc around 0 in complex plane
                    G = np.sqrt(
                        self._rng.uniform(0, 1, size=(self.size, self.nroot))
                    ) * np.exp(
                        1j
                        * self._rng.uniform(0, 2 * np.pi, size=(self.size, self.nroot))
                    )
                else:
                    G = self._rng.uniform(-1, 1, size=(self.size, self.nroot))
            else:
                G = self._guesses

            self.do_project_out(G)

            # orthonormalize via QR
            Q, _ = qr(G, mode="reduced")
            self.b[:, : Q.shape[1]] = Q
            self.basis_size = Q.shape[1]

            self.lam_old[:] = 0.0

        # check orthonormality of the initial basis
        self.orthonormality_check(self.b[:, : self.basis_size])

        self.iter = 0
        self.converged = False

        table = AsciiTable(
            columns=["Iter", "⟨E⟩", "max(ΔE)", "max(r)", "basis", "time (s)", "note"],
            formats=[
                "{:>4}",
                "{:>20.12f}",
                "{:>+12.4e}",
                "{:>+12.4e}",
                "{:>4}",
                "{:>10.4f}",
                "{:>20}",
            ],
        )

        logger.log(table.header(), self.log_level)

        for self.iter in range(self.maxiter):
            t0 = time.perf_counter()
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
            # 3b. project out undesirable directions in the sigma vectors, if provided
            self.do_project_out(Sblk)
            Gm = Bblk.T.conj() @ Sblk
            Gm = 0.5 * (Gm + Gm.T.conj())  # Hermitize
            lam, alpha = eigh(Gm)

            # sort eigenpair around user-specified shift
            if self.energy_shift is not None:
                idx = np.argsort(np.abs(lam - self.energy_shift))
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
            if hasattr(self, "_proj_out"):
                rproj = R.copy()
                self.do_project_out(rproj)
                rproj_norms = norm(rproj, axis=0)
                r_delta = np.abs(rnorms - rproj_norms)
            else:
                r_delta = None

            # precondition
            denom = lamr[np.newaxis, :] - self.h_diag[:, np.newaxis]
            mask = np.abs(denom) > self.preconditioner_denom_threshold
            # vectorize division only where denom is not too small, setting others to 0
            R[~mask] = 0.0
            np.divide(R, denom, out=R, where=mask)
            self.r[:, : self.nroot] = R

            # norms & convergence
            avg_e = lamr.mean()
            max_de = np.max(np.abs(lamr - self.lam_old[: self.nroot]))
            max_r = rnorms.max()

            conv_e = max_de < self.e_tol
            conv_r = max_r < self.r_tol
            if (conv_e and conv_r) or (self.basis_size == self.size):
                self.converged = True
                break
            if self.iter == self.maxiter - 1:
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
            self.do_project_out(R0)

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
            )
            self.basis_size += added
            # if we couldn't add enough, fill with random orthogonal vectors
            msg = ""
            if r_delta is not None:
                if r_delta.max() > 1e-9:
                    msg += f"r leakage {r_delta.max():.2e}"

            if added < to_add:
                missing = to_add - added
                # 'float' and 'np.float64' are not subdtypes of np.complexfloating
                if np.issubdtype(self.dtype, np.complexfloating):
                    # uniform distribution on unit disc around 0 in complex plane
                    temp = np.sqrt(
                        self._rng.uniform(0, 1, size=(self.size, missing))
                    ) * np.exp(
                        1j * self._rng.uniform(0, 2 * np.pi, size=(self.size, missing))
                    )
                else:
                    temp = self._rng.uniform(-1.0, 1.0, size=(self.size, missing))

                self.do_project_out(temp)

                added2 = self.add_rows_and_orthonormalize(
                    self.b[:, : self.basis_size],
                    temp,
                    self.b[:, self.basis_size :],
                    # self.b, self.basis_size, temp, missing
                )
                self.basis_size += added2
                sep = ", " if msg else ""
                msg += f"{sep}+{added2} rand"
            
            t1 = time.perf_counter()

            logger.log(
                table.row(self.iter, avg_e, max_de, max_r, self.basis_size, t1 - t0, msg),
                self.log_level,
            )

        logger.log(table.footer(), self.log_level)

        # compute final eigenpairs
        lamr = self.lam[: self.nroot]
        evecs = (
            self.b[:, : self.basis_size] @ self.alpha[: self.basis_size, : self.nroot]
        )

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
        k = min(self.collapse_size, self.basis_size)
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
                v -= np.dot(ai.conj(), v) * ai
            # orthogonalize against new vectors
            for i in range(n_new):
                si = A_new[:, i]
                v -= np.dot(si.conj(), v) * si

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
            norm2 = np.dot(v.conj(), v)
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
        if not np.allclose(b.T.conj() @ b, np.eye(b.shape[1]), atol=1e-11):
            S = b.T.conj() @ b
            logger.log_warning(f"{msg}")
            logger.log_warning(
                f"Largest deviation from orthonormality: {np.max(np.abs(S - np.eye(S.shape[0])))}"
            )
            logger.log_warning(f"S = {b.T.conj() @ b}")
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
        if self.energy_shift is not None:
            logger.log(
                f"  Target eigenval shift:    {self.energy_shift}", self.log_level
            )
        logger.log(f"  Maximum iterations:       {self.maxiter}\n", self.log_level)
