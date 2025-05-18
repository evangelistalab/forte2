#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "determinant_helpers.h"

namespace forte2 {

std::vector<std::vector<OccupationVector>> make_strings(int n, int k, size_t nirrep,
                                                        const std::vector<int>& mo_symmetry) {
    // n is the number of orbitals
    // k is the number of electrons
    std::vector<std::vector<OccupationVector>> result(nirrep);

    // sanity checks
    if (k < 0 or k > n or n > 64)
        return result;

    uint64_t x = (1ULL << k) - 1;
    const uint64_t limit = (1ULL << n);

    while (x < limit) {
        OccupationVector ov(x);
        int sym{0};
        for (int i = 0; i < n; ++i) {
            if (ov[i])
                sym ^= mo_symmetry[i];
        }
        result[sym].push_back(ov);

        // Gosper's hack to generate the next combination
        uint64_t c = x & -x; // Isolate rightmost 1-bit
        uint64_t r = x + c;  // Ripple carry
        if (c == 0)
            break; // Overflow
        x = (((x ^ r) >> 2) / c) | r;
    }
    return result;
}

std::vector<Determinant> make_hilbert_space(size_t nmo, size_t na, size_t nb, Determinant ref,
                                            int truncation, size_t nirrep,
                                            std::vector<int> mo_symmetry, int symmetry) {
    std::vector<Determinant> dets;
    if (nmo == 0 and na == 0 and nb == 0) {
        dets.push_back(Determinant());
        return dets;
    }

    if (nmo == 0 and (na != 0 or nb != 0)) {
        throw std::runtime_error("The number of MOs is 0 but the number of electrons is not.");
    }

    if (mo_symmetry.size() != nmo) {
        mo_symmetry = std::vector<int>(nmo, 0);
    }
    // find the maximum value in mo_symmetry and check that it is less than nirrep
    int max_sym = *std::max_element(mo_symmetry.begin(), mo_symmetry.end());
    if (max_sym >= static_cast<int>(nirrep)) {
        throw std::runtime_error("The symmetry of the MOs is greater than the number of irreps.");
    }
    // implement other sensible checks, like making sure that symmetry is less than nirrep and
    // na <= nmo, nb <= nmo
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
            det.a = Ia;
            for (const auto& Ib : strings_b[hb]) {
                det.b = Ib;
                if (det.count_diff(ref) / 2 <= truncation) {
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
    if (nmo == 0 and na == 0 and nb == 0) {
        dets.push_back(Determinant());
        return dets;
    }

    if (nmo == 0 and (na != 0 or nb != 0)) {
        throw std::runtime_error("The number of MOs is 0 but the number of electrons is not.");
    }

    if (mo_symmetry.size() != nmo) {
        mo_symmetry = std::vector<int>(nmo, 0);
    }
    // find the maximum value in mo_symmetry and check that it is less than nirrep
    int max_sym = *std::max_element(mo_symmetry.begin(), mo_symmetry.end());

    if (max_sym >= static_cast<int>(nirrep)) {
        throw std::runtime_error("The symmetry of the MOs is greater than the number of irreps.");
    }
    // implement other sensible checks, like making sure that symmetry is less than nirrep and
    // na <= nmo, nb <= nmo
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
                dets.emplace_back(Ia, Ib);
            }
        }
    }
    return dets;
}

} // namespace forte2
