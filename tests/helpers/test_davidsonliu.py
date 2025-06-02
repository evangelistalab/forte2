import numpy as np
import pytest
from forte2.helpers.davidsonliu import DavidsonLiuSolver


def test_davidson_vs_numpy():
    # 1. Build a small random symmetric matrix H
    np.random.seed(0)
    size = 100
    nroot = 3
    H = np.random.randn(size, size)
    H = 0.5 * (H + H.T)

    # 2. σ-builder that handles multiple columns at once
    def sigma_builder(basis_block: np.ndarray, sigma_block: np.ndarray) -> None:
        sigma_block[:] = H @ basis_block

    # 3. Instantiate and configure solver
    solver = DavidsonLiuSolver(
        size=size, nroot=nroot, collapse_per_root=3, basis_per_root=6
    )
    solver.add_h_diag(np.diag(H))
    solver.add_sigma_builder(sigma_builder)
    # no guesses → solver will generate random initial vectors

    # 4. Run Davidson
    evals, evecs = solver.solve()

    # 5. Reference solution via NumPy
    ref_vals, ref_vecs = np.linalg.eigh(H)
    ref_vals = ref_vals[:nroot]

    # 6. Compare eigenvalues (up to ordering)
    assert np.allclose(
        np.sort(evals), np.sort(ref_vals), atol=1e-12
    ), f"Eigenvalues mismatch: {evals} vs {ref_vals}"

    # 7. Compare eigenvector subspaces via projector difference
    P_solver = evecs @ evecs.T
    P_ref = ref_vecs[:, :3] @ ref_vecs[:, :3].T
    diff_norm = np.linalg.norm(P_solver - P_ref)
    assert diff_norm < 1e-5, f"Eigenvector subspaces differ (norm {diff_norm:.2e})"


def test_dl_1():
    """Test the Davidson-Liu solver with a 4x4 matrix.
    Pass the standard basis vectors as initial guesses."""
    # 1. Define a 4x4 symmetric matrix and reference eigenvalues
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    ref_vals, _ = np.linalg.eigh(matrix)

    # 2. Build the sigma-builder function
    def sigma_builder(basis_block: np.ndarray, sigma_block: np.ndarray) -> None:
        sigma_block[:] = matrix @ basis_block

    # 3. Instantiate the solver and add diagonal elements
    solver = DavidsonLiuSolver(
        size=size, nroot=nroot, collapse_per_root=1, basis_per_root=5
    )
    solver.add_h_diag(np.diag(matrix))

    # 4. Provide the standard basis vectors as guesses (each column is a guess vector)
    guesses = np.eye(size)
    solver.add_guesses(guesses)

    # 5. Set the sigma-builder
    solver.add_sigma_builder(sigma_builder)

    # 6. Run the solver
    evals, _ = solver.solve()

    # 7. Assert that the lowest computed eigenvalue is close to the reference value
    assert np.isclose(
        evals[0], ref_vals[0], atol=1e-6
    ), f"Eigenvalue mismatch: {evals[0]} vs {ref_vals[0]}"


def test_dl_2():
    """Test the Davidson-Liu solver with a 4x4 matrix.
    Pass one guess only that is not normalized."""
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    ref_vals, _ = np.linalg.eigh(matrix)

    # Build the sigma-builder function
    def sigma_builder(basis_block: np.ndarray, sigma_block: np.ndarray) -> None:
        sigma_block[:] = matrix @ basis_block

    # Instantiate and configure the solver
    solver = DavidsonLiuSolver(
        size=size, nroot=nroot, collapse_per_root=1, basis_per_root=5
    )
    solver.add_h_diag(np.diag(matrix))

    # Provide a single non-normalized guess vector (as a column vector)
    guess = np.array([[1.0], [0.0], [0.0], [0.0]])
    solver.add_guesses(guess)

    solver.add_sigma_builder(sigma_builder)

    # Run the solver
    evals, evecs = solver.solve()

    # Compare the computed lowest eigenvalue with the reference value
    assert np.isclose(
        evals[0], ref_vals[0], atol=1e-6
    ), f"Eigenvalue mismatch: {evals[0]} vs {ref_vals[0]}"


def solve_dl(size, nroot):
    """Test the Davidson-Liu solver with a matrix of size x size"""
    # Build a symmetric matrix of the given size
    matrix = np.zeros((size, size))
    for i in range(size):
        matrix[i, i] = -1.0 + i * 0.1
        for j in range(i):
            matrix[i, j] = 0.05 / (1.0 + abs(i - j))
            matrix[j, i] = matrix[i, j]

    # Compute the reference eigenvalues and eigenvectors using NumPy
    ref_evals, _ = np.linalg.eigh(matrix)

    # Define sigma-builder function as in the tests above
    def sigma_builder(basis_block: np.ndarray, sigma_block: np.ndarray) -> None:
        sigma_block[:] = matrix @ basis_block

    # Instantiate and configure the DavidsonLiuSolver
    solver = DavidsonLiuSolver(
        size=size,
        nroot=nroot,
        collapse_per_root=1,
        basis_per_root=5,
    )
    # Add the diagonal of the matrix
    solver.add_h_diag(np.diag(matrix))

    # Provide initial guesses: standard basis vectors for the first nroot columns
    guesses = np.eye(size)[:, :nroot]
    solver.add_guesses(guesses)

    # Add the sigma-builder
    solver.add_sigma_builder(sigma_builder)

    # Run the solver
    evals, _ = solver.solve()

    # Compare the computed eigenvalues with the reference eigenvalues (first nroot)
    test_evals = ref_evals[:nroot]
    assert np.allclose(
        np.sort(evals), np.sort(test_evals), atol=1e-6
    ), f"Eigenvalue mismatch: computed {evals} vs reference {test_evals}"


def test_dl_3():
    """Test the Davidson-Liu solver with matrices of different sizes and different number of roots"""
    for nroot in range(1, 11):
        solve_dl(10, nroot)
        solve_dl(100, nroot)
        solve_dl(1000, nroot)


def test_dl_4():
    """Test the Davidson-Liu solver with matrices of different sizes from 1 to all roots"""
    for size in range(1, 40):
        for nroot in range(1, size + 1):
            solve_dl(size, nroot)


def test_dl_no_guess():
    """Test the DavidsonLiuSolver with no guesses.
    Random guesses will be generated."""
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    ref_evals, _ = np.linalg.eigh(matrix)

    def sigma_builder(basis_block: np.ndarray, sigma_block: np.ndarray) -> None:
        sigma_block[:] = matrix @ basis_block

    # Instantiate the solver without an explicit guess.
    solver = DavidsonLiuSolver(
        size=size, nroot=nroot, collapse_per_root=1, basis_per_root=5
    )
    solver.add_h_diag(np.diag(matrix))
    solver.add_sigma_builder(sigma_builder)
    # Do not add any guesses so that the solver generates random ones.
    evals, _ = solver.solve()

    assert np.isclose(
        evals[0], ref_evals[0], atol=1e-6
    ), f"Expected {ref_evals[0]}, got {evals[0]}"


def test_project_out():
    """Test projecting out a vector.
    Random guesses will be generated."""
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    ref_evals, ref_evecs = np.linalg.eigh(matrix)
    # Project out the first eigenvector
    proj_out = ref_evecs[:, 0]  # .reshape(-1, 1)

    def sigma_builder(basis_block: np.ndarray, sigma_block: np.ndarray) -> None:
        sigma_block[:] = matrix @ basis_block

    solver = DavidsonLiuSolver(
        size=size, nroot=nroot, collapse_per_root=1, basis_per_root=5
    )
    solver.add_h_diag(np.diag(matrix))
    solver.add_sigma_builder(sigma_builder)
    solver.add_project_out([proj_out])
    evals, _ = solver.solve()

    # Since the lowest eigenvector was projected out, the next eigenvalue should appear.
    assert np.isclose(
        evals[0], ref_evals[1], atol=1e-6
    ), f"Expected {ref_evals[1]}, got {evals[0]}"


def test_dl_restart_1():
    """Test restarting DavidsonLiuSolver.
    Calling solve() twice with the same matrix."""
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    ref_evals, _ = np.linalg.eigh(matrix)

    def sigma_builder(basis_block: np.ndarray, sigma_block: np.ndarray) -> None:
        sigma_block[:] = matrix @ basis_block

    solver = DavidsonLiuSolver(
        size=size, nroot=nroot, collapse_per_root=1, basis_per_root=5
    )
    solver.add_h_diag(np.diag(matrix))
    solver.add_sigma_builder(sigma_builder)
    guesses = np.eye(size)[:, :nroot]
    solver.add_guesses(guesses)
    evals, _ = solver.solve()

    # Restart the solver by calling solve() again.
    evals_restart, _ = solver.solve()
    assert np.isclose(
        evals_restart[0], ref_evals[0], atol=1e-6
    ), f"Expected {ref_evals[0]}, got {evals_restart[0]}"


def test_dl_restart_2():
    """Test restarting DavidsonLiuSolver with a changed matrix.
    Calling solve() twice with an updated sigma-builder and diagonal."""
    size = 4
    nroot = 1
    matrix = np.array([[-1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    ref_evals, _ = np.linalg.eigh(matrix)

    def sigma_builder(basis_block: np.ndarray, sigma_block: np.ndarray) -> None:
        sigma_block[:] = matrix @ basis_block

    solver = DavidsonLiuSolver(
        size=size, nroot=nroot, collapse_per_root=1, basis_per_root=5
    )
    solver.add_h_diag(np.diag(matrix))
    solver.add_sigma_builder(sigma_builder)
    guesses = np.eye(size)[:, :nroot]
    solver.add_guesses(guesses)
    solver.solve()

    # Update the matrix.
    matrix2 = np.array([[-2, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]])
    ref_evals2, _ = np.linalg.eigh(matrix2)

    def sigma_builder2(basis_block: np.ndarray, sigma_block: np.ndarray) -> None:
        sigma_block[:] = matrix2 @ basis_block

    solver.add_h_diag(np.diag(matrix2))
    solver.add_sigma_builder(sigma_builder2)
    evals, _ = solver.solve()

    assert np.isclose(
        evals[0], ref_evals2[0], atol=1e-6
    ), f"Expected {ref_evals2[0]}, got {evals[0]}"


if __name__ == "__main__":
    test_dl_1()
    test_dl_2()
    test_dl_3()
    test_dl_4()
    test_dl_no_guess()
    test_project_out()
    test_dl_restart_1()
    test_dl_restart_2()
    test_davidson_vs_numpy()
