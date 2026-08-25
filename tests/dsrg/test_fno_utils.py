import numpy as np

from forte2.dsrg.fno_utils import determine_fno_n_keep


def test_fno_n_keep_p_o():
    occ = np.array([5.0, 3.0, 2.0])
    # naive cutoff at 70% (=7.0) lands between the first and second orbital
    n_keep = determine_fno_n_keep(occ, p_o=0.7, n_kappa=None, degeneracy_tol=1e-2)
    assert n_keep == 2
    n_keep = determine_fno_n_keep(occ, p_o=1.0, n_kappa=None, degeneracy_tol=1e-2)
    assert n_keep == 3


def test_fno_n_keep_n_kappa():
    occ = np.array([5.0, 3.0, 2.0, 0.5])
    n_keep = determine_fno_n_keep(occ, p_o=None, n_kappa=1.0, degeneracy_tol=1e-2)
    assert n_keep == 3


def test_fno_n_keep_degeneracy_padding():
    # occ[1] and occ[2] are near-degenerate (Kramers-partner-like); a naive
    # p_o cutoff that would land between them must be pushed past both.
    occ = np.array([5.0, 3.0, 2.999, 1.0, 0.5])
    naive_target = 5.0 + 3.0  # cumulative sum through index 1
    total = occ.sum()
    n_keep = determine_fno_n_keep(
        occ, p_o=naive_target / total, n_kappa=None, degeneracy_tol=1e-2
    )
    assert n_keep == 3, "near-degenerate pair (indices 1, 2) must not be split"

    # a genuinely well-separated boundary should not be padded
    n_keep = determine_fno_n_keep(occ, p_o=None, n_kappa=0.7, degeneracy_tol=1e-2)
    assert n_keep == 4
