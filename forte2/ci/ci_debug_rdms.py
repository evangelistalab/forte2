from forte2.lib.sparse_ops import SparseState


def sparse_state_from_ci_vector(dets, coefficients):
    """
    Build a `SparseState` from a list of determinants and a coefficient vector.

    Used to validate the fast RDM kernels (block-addressed or hash-map based) against
    the generic, brute-force `forte2.lib.rdms` reference implementations, which operate
    on `SparseState` and therefore need no knowledge of a solver's internal string/
    determinant bookkeeping.

    Parameters
    ----------
    dets : Sequence[Determinant]
        The determinants spanning the coefficient vector.
    coefficients : np.ndarray
        The CI coefficients, one per determinant.

    Returns
    -------
    SparseState
        A sparse state mapping each determinant to its coefficient.
    """
    return SparseState({d: c for d, c in zip(dets, coefficients)})
