import numpy as np
from forte2 import ints


class CholeskyIntegrals:
    def __init__(self, basis, memory=1000, delta=1e-12):
        self.basis = basis
        # convert MB to number of double elements
        self.memory = memory * 1024 * 1024 / 8
        self.delta = delta
        self.nbf = len(basis)

    def _compute_diagonal(self):
        return ints.coulomb_4c_diagonal(self.basis)

    def _compute_row(self, pivot):
        return ints.coulomb_4c_row(self.basis, pivot)

    def compute(self):
        n = self.nbf**2
        Q = 0

        # mimic the C++ int max for memory checks
        max_rows = (self.memory - n) // (2 * n)

        # initialize diagonal
        diag = self._compute_diagonal()

        L = []  # list of rows (each row is length n)
        pivots = []

        # main loop
        while Q < n:
            pivot = int(np.argmax(diag))
            Dmax = float(diag[pivot])

            if Dmax < self.delta or Dmax < 0:
                break

            pivots.append(pivot)
            L_QQ = np.sqrt(Dmax)

            if Q > max_rows:
                raise RuntimeError("Cholesky: Memory constraint exceeded.")

            # get new row (m|pivot)
            row = self._compute_row(pivot)

            # subtract previous contributions
            if Q > 0:
                L_temp = np.vstack(L[:Q])  # shape (Q, n)
                alphas = L_temp[:, pivots[Q]]  # shape (Q, )
                row -= alphas @ L_temp
                del L_temp

            # scale
            row /= L_QQ

            # zero the upper triangle (enforce Cholesky structure)
            row[pivots] = 0.0

            # set pivot element
            row[pivot] = L_QQ

            # update Schur complement diagonal
            diag -= row * row

            # force pivot entries to zero
            diag[pivots] = 0.0

            L.append(row)
            Q += 1

        # stack into final L
        if Q == 0:
            raise RuntimeError("Cholesky: No Cholesky vectors were computed.")

        self.B = np.vstack(L[:Q]).reshape((Q, self.nbf, self.nbf)).copy()
