#include <algorithm>
#include <cassert>

#include "determinant_helpers.h"

namespace forte2 {

std::pair<String, double> create_single_excitation(const String& str, size_t i, size_t a) {
    String new_str = str;
    double sign = new_str.destroy(i);
    sign *= new_str.create(a);
    return {new_str, sign};
}

std::pair<String, double> create_double_excitation(const String& str, size_t i, size_t j, size_t a,
                                                   size_t b) {
    String new_str = str;
    double sign = new_str.destroy(i);
    sign *= new_str.destroy(j);
    sign *= new_str.create(b);
    sign *= new_str.create(a);
    return {new_str, sign};
}

std::pair<String, double> create_single_excitation_fast(const String& str, size_t i, size_t a) {
    String new_str = str;
    double sign = new_str.destroy_fast(i);
    sign *= new_str.create_fast(a);
    return {new_str, sign};
}

std::pair<String, double> create_double_excitation_fast(const String& str, size_t i, size_t j,
                                                        size_t a, size_t b) {
    String new_str = str;
    double sign = new_str.destroy_fast(i);
    sign *= new_str.destroy_fast(j);
    sign *= new_str.create_fast(b);
    sign *= new_str.create_fast(a);
    return {new_str, sign};
}

std::pair<Determinant, double> create_single_a_excitation(const Determinant& det, size_t i,
                                                          size_t a) {
    Determinant new_det = det;
    double sign = new_det.destroy_a(i);
    sign *= new_det.create_a(a);
    return {new_det, sign};
}

std::pair<Determinant, double> create_single_b_excitation(const Determinant& det, size_t i,
                                                          size_t a) {
    Determinant new_det = det;
    double sign = new_det.destroy_b(i);
    sign *= new_det.create_b(a);
    return {new_det, sign};
}

std::pair<Determinant, double> create_double_aa_excitation(const Determinant& det, size_t i,
                                                           size_t j, size_t a, size_t b) {
    Determinant new_det = det;
    double sign = new_det.destroy_a(i);
    sign *= new_det.destroy_a(j);
    sign *= new_det.create_a(b);
    sign *= new_det.create_a(a);
    return {new_det, sign};
}

std::pair<Determinant, double> create_double_bb_excitation(const Determinant& det, size_t i,
                                                           size_t j, size_t a, size_t b) {
    Determinant new_det = det;
    double sign = new_det.destroy_b(i);
    sign *= new_det.destroy_b(j);
    sign *= new_det.create_b(b);
    sign *= new_det.create_b(a);
    return {new_det, sign};
}

std::pair<Determinant, double> create_double_ab_excitation(const Determinant& det, size_t i,
                                                           size_t j, size_t a, size_t b) {
    Determinant new_det = det;
    double sign = new_det.destroy_a(i);
    sign *= new_det.destroy_b(j);
    sign *= new_det.create_b(b);
    sign *= new_det.create_a(a);
    return {new_det, sign};
}

void compute_fast_virtual(const std::vector<size_t>& occ, std::vector<size_t>& vir,
                          const size_t n) {
    // Debug sanity checks
#ifndef NDEBUG
    for (size_t j = 1; j < occ.size(); ++j)
        assert(occ[j] > occ[j - 1]); // check that occ is sorted
    for (auto o : occ)
        assert(o < n);                    // check that all occupied orbitals are less than n
    assert(vir.size() == n - occ.size()); // check that vir has enough space
#endif
    size_t i = 0; // index into vir vector
    size_t k = 0; // index into orbitals
    for (auto o : occ) {
        // add all orbitals before the occupied orbital o
        for (; k < o; ++k) {
            vir[i++] = k;
        }
        // set new restarting point after the occupied orbital
        k = o + 1;
    }
    // add all remaining orbitals after the last occupied orbital
    for (; k < n; ++k) {
        vir[i++] = k;
    }
}

// std::shared_ptr<psi::Matrix> make_s2_matrix(const std::vector<Determinant>& dets) {
//     const size_t n = dets.size();
//     auto S2 = std::make_shared<psi::Matrix>("S^2", n, n);

//     auto threads = omp_get_max_threads();

// #pragma omp parallel for schedule(dynamic) num_threads(threads)
//     for (size_t I = 0; I < n; I++) {
//         const Determinant& detI = dets[I];
//         for (size_t J = I; J < n; J++) {
//             const Determinant& detJ = dets[J];
//             const double S2IJ = spin2(detI, detJ);
//             S2->set(I, J, S2IJ);
//             S2->set(J, I, S2IJ);
//         }
//     }
//     return S2;
// }

// std::shared_ptr<psi::Matrix>
// make_hamiltonian_matrix(const std::vector<Determinant>& dets,
//                         std::shared_ptr<ActiveSpaceIntegrals> as_ints) {
//     const size_t n = dets.size();
//     auto H = std::make_shared<psi::Matrix>("H", n, n);

//     // If we are running DiskDF then we need to revert to a single thread loop
//     auto threads = (as_ints->get_integral_type() == DiskDF) ? 1 : omp_get_max_threads();

// #pragma omp parallel for schedule(dynamic) num_threads(threads)
//     for (size_t I = 0; I < n; I++) {
//         const Determinant& detI = dets[I];
//         for (size_t J = I; J < n; J++) {
//             const Determinant& detJ = dets[J];
//             double HIJ = as_ints->slater_rules(detI, detJ);
//             H->set(I, J, HIJ);
//             H->set(J, I, HIJ);
//         }
//     }
//     return H;
// }

std::vector<std::vector<String>> make_strings(int n, int k, size_t nirrep,
                                              const std::vector<int>& mo_symmetry) {
    // n is the number of orbitals
    // k is the number of electrons
    std::vector<std::vector<String>> strings(nirrep);
    if ((k >= 0) and (k <= n)) { // check that (n > 0) makes sense.
        auto I = String::zero();
        const auto I_begin = I.begin();
        const auto I_end = std::next(I.begin(), n);
        // Generate the string 00000001111111
        //                      {n-k}  { k }
        for (int i = std::max(0, n - k); i < n; ++i)
            I[i] = true; // 1
        do {
            int sym{0};
            for (int i = 0; i < n; ++i) {
                if (I[i])
                    sym ^= mo_symmetry[i];
            }
            strings[sym].push_back(I);
        } while (std::next_permutation(I_begin, I_end));
    }
    return strings;
}

std::vector<Determinant> make_hilbert_space(size_t nmo, size_t na, size_t nb, Determinant ref,
                                            int truncation, size_t nirrep,
                                            std::vector<int> mo_symmetry, int symmetry) {
    std::vector<Determinant> dets;
    if (mo_symmetry.size() != nmo) {
        mo_symmetry = std::vector<int>(nmo, 0);
    }
    // find the maximum value in mo_symmetry and check that it is less than nirrep
    int max_sym = *std::max_element(mo_symmetry.begin(), mo_symmetry.end());
    if (max_sym >= static_cast<int>(nirrep)) {
        throw std::runtime_error("The symmetry of the MOs is greater than the number of irreps.");
    }
    // implement other sensible checks, like making sure that symmetry is less than nirrep and na
    //<=
    // nmo, nb <= nmo
    if (symmetry >= static_cast<int>(nirrep)) {
        throw std::runtime_error(
            "The symmetry of the determinants is greater than the number of irreps.");
    }
    if (na > nmo) {
        throw std::runtime_error(
            "The number of alpha electrons is greater than the number of MOs.");
    }
    if (nb > nmo) {
        throw std::runtime_error("The number of beta electrons is greater than the number of MOs.");
    }
    if (truncation < 0 || truncation > static_cast<int>(na + nb)) {
        throw std::runtime_error("The truncation level must an integer between 0 and na + nb.");
    }

    auto strings_a = make_strings(nmo, na, nirrep, mo_symmetry);
    auto strings_b = make_strings(nmo, nb, nirrep, mo_symmetry);
    for (size_t ha = 0; ha < nirrep; ha++) {
        int hb = symmetry ^ ha;
        for (const auto& Ia : strings_a[ha]) {
            Determinant det;
            det.set_a_string(Ia);
            for (const auto& Ib : strings_b[hb]) {
                det.set_b_string(Ib);
                if (det.fast_a_xor_b_count(ref) / 2 <= truncation) {
                    dets.push_back(det);
                }
            }
        }
    }
    return dets;
}

std::vector<Determinant> make_hilbert_space(size_t nmo, size_t na, size_t nb, size_t nirrep,
                                            std::vector<int> mo_symmetry, int symmetry) {
    std::vector<Determinant> dets;
    if (mo_symmetry.size() != nmo) {
        mo_symmetry = std::vector<int>(nmo, 0);
    }
    // find the maximum value in mo_symmetry and check that it is less than nirrep
    int max_sym = *std::max_element(mo_symmetry.begin(), mo_symmetry.end());
    if (max_sym >= static_cast<int>(nirrep)) {
        throw std::runtime_error("The symmetry of the MOs is greater than the number of irreps.");
    }
    // implement other sensible checks, like making sure that symmetry is less than nirrep and na
    //<=
    // nmo, nb <= nmo
    if (symmetry >= static_cast<int>(nirrep)) {
        throw std::runtime_error(
            "The symmetry of the determinants is greater than the number of irreps.");
    }
    if (na > nmo) {
        throw std::runtime_error(
            "The number of alpha electrons is greater than the number of MOs.");
    }
    if (nb > nmo) {
        throw std::runtime_error("The number of beta electrons is greater than the number of MOs.");
    }

    auto strings_a = make_strings(nmo, na, nirrep, mo_symmetry);
    auto strings_b = make_strings(nmo, nb, nirrep, mo_symmetry);
    for (size_t ha = 0; ha < nirrep; ha++) {
        int hb = symmetry ^ ha;
        for (const auto& Ia : strings_a[ha]) {
            for (const auto& Ib : strings_b[hb]) {
                dets.push_back(Determinant(Ia, Ib));
            }
        }
    }
    return dets;
}

} // namespace forte2
