import numpy as np

from forte2.symmetry.sph_harm_utils import sph_real_to_complex, clebsh_gordan_spin_half
from forte2.helpers.comparisons import approx

def test_sph_real_to_complex():
    for i in range(10):
        U = sph_real_to_complex(i)
        assert U.shape == (2 * i + 1, 2 * i + 1)
        # every column/row should have norm 1
        assert all(np.isclose(np.linalg.norm(U[:, _], ord=2), 1.0, rtol=0, atol=1e-12) for _ in range(2 * i + 1))

def test_sph_real_to_complex_p():
    U = sph_real_to_complex(1)

def test_clebsh_gordan_spin_half():
    # l = 0
    assert clebsh_gordan_spin_half(0, 1, 1, 1) == approx(1.0)
    assert clebsh_gordan_spin_half(0, 1, 1, -1) == approx(0.0)
    assert clebsh_gordan_spin_half(0, -1, 1, -1) == approx(1.0)

    # l = 1, m_j = 3/2
    assert clebsh_gordan_spin_half(1, 1, 3, 3) == approx(1.0)
    # l = 1, m_j = 1/2
    assert clebsh_gordan_spin_half(1, -1, 3, 1) == approx(np.sqrt(1 / 3))
    assert clebsh_gordan_spin_half(1, -1, 1, 1) == approx(np.sqrt(2 / 3))
    assert clebsh_gordan_spin_half(1, 1, 3, 1) == approx(np.sqrt(2 / 3))
    assert clebsh_gordan_spin_half(1, 1, 1, 1) == approx(-np.sqrt(1 / 3))

    # l = 2, m_j = 5/2
    assert clebsh_gordan_spin_half(2, 1, 5, 5) == approx(1.0)
    # l = 2, m_j = 3/2
    assert clebsh_gordan_spin_half(2, -1, 5, 3) == approx(np.sqrt(1 / 5))
    assert clebsh_gordan_spin_half(2, -1, 3, 3) == approx(np.sqrt(4 / 5))
    assert clebsh_gordan_spin_half(2, 1, 5, 3) == approx(np.sqrt(4 / 5))
    assert clebsh_gordan_spin_half(2, 1, 3, 3) == approx(-np.sqrt(1 / 5))
    # l = 2, m_j = 1/2
    assert clebsh_gordan_spin_half(2, -1, 5, 1) == approx(np.sqrt(2 / 5))
    assert clebsh_gordan_spin_half(2, -1, 3, 1) == approx(np.sqrt(3 / 5))
    assert clebsh_gordan_spin_half(2, 1, 5, 1) == approx(np.sqrt(3 / 5))
    assert clebsh_gordan_spin_half(2, 1, 3, 1) == approx(-np.sqrt(2 / 5))
