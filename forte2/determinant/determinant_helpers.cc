#include <algorithm>
#include <cassert>

#include "determinant/determinant_helpers.h"

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

std::pair<String, double> create_single_excitation_unchecked(const String& str, size_t i,
                                                             size_t a) {
    String new_str = str;
    double sign = new_str.destroy_unchecked(i);
    sign *= new_str.create_unchecked(a);
    return {new_str, sign};
}

std::pair<String, double> create_double_excitation_unchecked(const String& str, size_t i, size_t j,
                                                             size_t a, size_t b) {
    String new_str = str;
    double sign = new_str.destroy_unchecked(i);
    sign *= new_str.destroy_unchecked(j);
    sign *= new_str.create_unchecked(b);
    sign *= new_str.create_unchecked(a);
    return {new_str, sign};
}

std::pair<Determinant, double> create_single_a_excitation(const Determinant& det, size_t i,
                                                          size_t a) {
    Determinant new_det = det;
    double sign = new_det.destroy_alpha(i);
    sign *= new_det.create_alpha(a);
    return {new_det, sign};
}

std::pair<Determinant, double> create_single_b_excitation(const Determinant& det, size_t i,
                                                          size_t a) {
    Determinant new_det = det;
    double sign = new_det.destroy_beta(i);
    sign *= new_det.create_beta(a);
    return {new_det, sign};
}

std::pair<Determinant, double> create_double_aa_excitation(const Determinant& det, size_t i,
                                                           size_t j, size_t a, size_t b) {
    Determinant new_det = det;
    double sign = new_det.destroy_alpha(i);
    sign *= new_det.destroy_alpha(j);
    sign *= new_det.create_alpha(b);
    sign *= new_det.create_alpha(a);
    return {new_det, sign};
}

std::pair<Determinant, double> create_double_bb_excitation(const Determinant& det, size_t i,
                                                           size_t j, size_t a, size_t b) {
    Determinant new_det = det;
    double sign = new_det.destroy_beta(i);
    sign *= new_det.destroy_beta(j);
    sign *= new_det.create_beta(b);
    sign *= new_det.create_beta(a);
    return {new_det, sign};
}

std::pair<Determinant, double> create_double_ab_excitation(const Determinant& det, size_t i,
                                                           size_t j, size_t a, size_t b) {
    Determinant new_det = det;
    double sign = new_det.destroy_alpha(i);
    sign *= new_det.destroy_beta(j);
    sign *= new_det.create_beta(b);
    sign *= new_det.create_alpha(a);
    return {new_det, sign};
}

void collect_virtual_orbitals(const std::vector<size_t>& occ, std::vector<size_t>& vir,
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

double apply_operator_to_det_unchecked(const Determinant& d, Determinant& new_d,
                                       const Determinant& cre, const Determinant& ann,
                                       const Determinant& sign) {
    size_t n = 0;
    if constexpr (Determinant::nbits == 128) {
        // specialization for one 64-orbital word per spin sector
        const auto w0 = d.get_word(0) & (~ann.get_word(0));
        const auto w1 = d.get_word(1) & (~ann.get_word(1));
        n += std::popcount(w0 & sign.get_word(0));
        n += std::popcount(w1 & sign.get_word(1));
        new_d.set_word(0, w0 | cre.get_word(0));
        new_d.set_word(1, w1 | cre.get_word(1));
    } else if constexpr (Determinant::nbits == 256) {
        const auto w0 = d.get_word(0) & (~ann.get_word(0));
        const auto w1 = d.get_word(1) & (~ann.get_word(1));
        const auto w2 = d.get_word(2) & (~ann.get_word(2));
        const auto w3 = d.get_word(3) & (~ann.get_word(3));
        n += std::popcount(w0 & sign.get_word(0));
        n += std::popcount(w1 & sign.get_word(1));
        n += std::popcount(w2 & sign.get_word(2));
        n += std::popcount(w3 & sign.get_word(3));
        new_d.set_word(0, w0 | cre.get_word(0));
        new_d.set_word(1, w1 | cre.get_word(1));
        new_d.set_word(2, w2 | cre.get_word(2));
        new_d.set_word(3, w3 | cre.get_word(3));
    } else {
        // loop over packed storage words
        for (size_t i = 0; i < Determinant::nwords_; ++i) {
            // apply the annihilation operator
            const auto w = d.get_word(i) & (~ann.get_word(i));
            // compute the sign
            n += std::popcount(w & sign.get_word(i));
            // apply the creation operator
            new_d.set_word(i, w | cre.get_word(i));
        }
    }
    return parity_to_sign(n);
}

double spin2(const Determinant& lhs, const Determinant& rhs) {
    int nmo = Determinant::norb_capacity;
    const Determinant& I = lhs;
    const Determinant& J = rhs;

    // Compute the matrix elements of the operator S^2
    // S^2 = S- S+ + Sz (Sz + 1)
    //     = Sz (Sz + 1) + Nbeta + Npairs - sum_pq' a+(qa) a+(pb) a-(qb) a-(pa)
    double matrix_element = 0.0;

    // Make sure that Ms is the same otherwise the matrix element is automatically zero
    if ((lhs.count_alpha() != rhs.count_alpha()) or (lhs.count_beta() != rhs.count_beta())) {
        return 0.0;
    }

    Determinant lr_diff = lhs ^ rhs;

    int nadiff = lr_diff.count_alpha() / 2;
    int nbdiff = lr_diff.count_beta() / 2;
    int na = lhs.count_alpha();
    int nb = lhs.count_beta();
    int npair = lhs.npair();

    double Ms = 0.5 * static_cast<double>(na - nb);

    // PhiI = PhiJ -> S^2 = Sz (Sz + 1) + Nbeta - Npairs
    if ((nadiff == 0) and (nbdiff == 0)) {
        matrix_element += Ms * (Ms + 1.0) + double(nb) - double(npair);
    }

    // PhiI = a+(qa) a+(pb) a-(qb) a-(pa) PhiJ
    if ((nadiff == 1) and (nbdiff == 1)) {
        // Find a pair of spin coupled electrons
        int i = -1;
        int j = -1;
        // The logic here follows the spin-flip coupling between opposite-spin occupations.
        for (int p = 0; p < nmo; ++p) {
            if (J.na(p) and I.nb(p) and (not J.nb(p)) and (not I.na(p)))
                i = p;
            if (J.nb(p) and I.na(p) and (not J.na(p)) and (not I.nb(p)))
                j = p;
        }
        if (i != j and i >= 0 and j >= 0) {
            double sign = rhs.slater_sign_a(i) * rhs.slater_sign_b(j) * lhs.slater_sign_a(j) *
                          lhs.slater_sign_b(i);
            matrix_element -= sign;
        }
    }
    return (matrix_element);
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
            det.set_alpha_string(Ia);
            for (const auto& Ib : strings_b[hb]) {
                det.set_beta_string(Ib);
                if (det.symmetric_difference_count(ref) / 2 <= truncation) {
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
