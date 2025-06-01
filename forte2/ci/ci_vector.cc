
#include "helpers/np_matrix_functions.h"

#include "ci_vector.h"
#include "ci_string_lists.h"
#include "ci_string_address.h"

namespace forte2 {

// Global debug flag
bool debug_gas_vector = true;

// Wrapper function
template <typename Func> void debug(Func func) {
    if (debug_gas_vector) {
        func();
    }
}

CIVector::CIVector(const CIStrings& lists)
    : symmetry_(lists.symmetry()), lists_(lists), alfa_address_(lists_.alfa_address()),
      beta_address_(lists_.beta_address()) {
    startup();
}

void CIVector::startup() {
    // nirrep_ = lists_.nirrep();
    // ncmo_ = lists_.ncmo();
    // cmopi_ = lists_.cmopi();
    // cmopi_offset_ = lists_.cmopi_offset();

    // ndet_ = 0;
    // for (const auto& [_, class_Ia, class_Ib] : lists_.determinant_classes()) {
    //     auto size_alfa = alfa_address_->strpcls(class_Ia);
    //     auto size_beta = beta_address_->strpcls(class_Ib);
    //     auto detpcls = size_alfa * size_beta;
    //     // ndet_ += detpcls;
    //     detpcls_.push_back(detpcls);
    // }

    ndet_ = lists_.ndet();
    // Allocate the wave function
    for (const auto& [_, class_Ia, class_Ib] : lists_.determinant_classes()) {
        C_.push_back(make_zeros<nb::numpy, double, 2>(
            {alfa_address_->strpcls(class_Ia), beta_address_->strpcls(class_Ib)}));
    }
}

// size_t CIVector::symmetry() const { return symmetry_; }

// size_t CIVector::nirrep() const { return nirrep_; }

// size_t CIVector::ncmo() const { return ncmo_; }

// size_t CIVector::size() const { return ndet_; }

// const std::vector<size_t>& CIVector::detpi() const { return detpi_; }

// psi::Dimension CIVector::cmopi() const { return cmopi_; }

// const std::vector<size_t>& CIVector::cmopi_offset() const { return cmopi_offset_; }

// const std::shared_ptr<CIStrings>& CIVector::lists() const { return lists_; }

// void CIVector::print(double threshold) const {
//     const_for_each_element([&](const size_t& n, const int& class_Ia, const int& class_Ib,
//                                const size_t& Ia, const size_t& Ib, const double& c) {
//         if (std::fabs(c) >= threshold) {
//             Determinant I(lists_.alfa_str(class_Ia, Ia), lists_.beta_str(class_Ib, Ib));
//             psi::outfile->Printf("\n  %+15.9f %s [%2d](%2d,%2d) -> (%2d,%2d)", c,
//                                  str(I, lists_.ncmo()).c_str(), static_cast<int>(n), class_Ia,
//                                  class_Ib, static_cast<int>(Ia), static_cast<int>(Ib));
//         }
//     });
// }

// SparseState CIVector::as_state_vector() const {
//     SparseState state_vector;
//     const_for_each_element([&](const size_t& /*n*/, const int& class_Ia, const int& class_Ib,
//                                const size_t& Ia, const size_t& Ib, const double& c) {
//         if (std::fabs(c) > 1.0e-12) {
//             Determinant I(lists_.alfa_str(class_Ia, Ia), lists_.beta_str(class_Ib, Ib));
//             state_vector[I] = c;
//         }
//     });
//     return state_vector;
// }

// void CIVector::copy(CIVector& wfn) {
//     for (const auto& [n, _1, _2] : lists_.determinant_classes()) {
//         C_[n]->copy(wfn.C_[n]);
//     }
// }

void CIVector::copy(np_vector vec) {
    auto v = vec.view();
    for_each_index_element([&](const size_t& I, double& c) { c = v(I); });
}

void CIVector::copy_to(np_vector vec) const {
    auto v = vec.view();
    const_for_each_index_element([&](const size_t& I, const double& c) { v(I) = c; });
}

// void CIVector::set_to(double value) {
//     for_each_index_element([&](const size_t& /*I*/, double& c) { c = value; });
// }

// void CIVector::set(std::vector<std::tuple<size_t, size_t, size_t, double>>& sparse_vec) {
//     zero();
//     for (const auto& [n, Ia, Ib, c] : sparse_vec) {
//         C_[n]->set(Ia, Ib, c);
//     }
// }

// double CIVector::dot(const CIVector& wfn) const {
//     double dot = 0.0;
//     for (const auto& [n, _1, _2] : lists_.determinant_classes()) {
//         dot += C_[n]->vector_dot(wfn.C_[n]);
//     }
//     return (dot);
// }

// double CIVector::norm(double power) {
//     double norm = dot(*this);
//     return std::pow(norm, 1.0 / power);
// }

// void CIVector::normalize() {
//     double factor = norm(2.0);
//     for (auto& c : C_)
//         c->scale(1.0 / factor);
// }

void CIVector::zero() {
    for (auto& c : C_)
        matrix::zero(c);
}

// void CIVector::print_natural_orbitals(std::shared_ptr<MOSpaceInfo> mo_space_info,
//                                       std::shared_ptr<RDMs> rdms) {
//     print_h2("Natural Orbitals Occupation Numbers");
//     const auto active_dim = mo_space_info->dimension("ACTIVE");
//     const auto idocc_pi = mo_space_info->dimension("INACTIVE_DOCC");

//     auto G1 = rdms->SF_G1();
//     auto& G1_data = G1.data();

//     auto opdm = std::make_shared<psi::Matrix>("OPDM", active_dim, active_dim);

//     int offset = 0;
//     for (int h = 0; h < nirrep_; h++) {
//         for (int u = 0; u < active_dim[h]; u++) {
//             for (int v = 0; v < active_dim[h]; v++) {
//                 double gamma_uv = G1_data[(u + offset) * ncmo_ + v + offset];
//                 opdm->set(h, u, v, gamma_uv);
//             }
//         }
//         offset += active_dim[h];
//     }

//     auto OCC = std::make_shared<psi::Vector>("Occupation numbers", active_dim);
//     auto NO = std::make_shared<psi::Matrix>("MO -> NO transformation", active_dim, active_dim);

//     opdm->diagonalize(NO, OCC, psi::descending);
//     std::vector<std::pair<double, std::pair<int, int>>> vec_irrep_occupation;
//     for (int h = 0; h < nirrep_; h++) {
//         for (int u = 0; u < active_dim[h]; u++) {
//             auto index = u + idocc_pi[h] + 1;
//             auto irrep_occ = std::make_pair(OCC->get(h, u), std::make_pair(h, index));
//             vec_irrep_occupation.push_back(irrep_occ);
//         }
//     }
//     std::sort(vec_irrep_occupation.begin(), vec_irrep_occupation.end(),
//               std::greater<std::pair<double, std::pair<int, int>>>());

//     size_t count = 0;
//     psi::outfile->Printf("\n    ");
//     for (auto vec : vec_irrep_occupation) {
//         psi::outfile->Printf(" %4d%-4s%11.6f  ", vec.second.second,
//                              mo_space_info->irrep_label(vec.second.first).c_str(), vec.first);
//         if (count++ % 3 == 2 && count != vec_irrep_occupation.size())
//             psi::outfile->Printf("\n    ");
//     }
//     psi::outfile->Printf("\n");
// }

np_matrix CIVector::gather_C_block(np_matrix M, bool alfa,
                                   std::shared_ptr<StringAddress> alfa_address,
                                   std::shared_ptr<StringAddress> beta_address, int class_Ia,
                                   int class_Ib, bool zero) {
    // if alfa is true just return the pointer to the block
    int block_idx = lists_.string_class()->block_index(class_Ia, class_Ib);
    auto c = C(block_idx);
    if (alfa) {
        if (zero)
            matrix::zero(c);
        return c;
    }
    // if alfa is false
    size_t maxIa = alfa_address->strpcls(class_Ia);
    size_t maxIb = beta_address->strpcls(class_Ib);
    auto m = M.view();
    if (zero) {
        for (size_t Ib = 0; Ib < maxIb; ++Ib)
            for (size_t Ia = 0; Ia < maxIa; ++Ia)
                m(Ib, Ia) = 0.0;
    } else {
        for (size_t Ia = 0; Ia < maxIa; ++Ia)
            for (size_t Ib = 0; Ib < maxIb; ++Ib)
                m(Ib, Ia) = c(Ia, Ib);
    }
    return M;
}

void CIVector::scatter_C_block(np_matrix M, bool alfa, std::shared_ptr<StringAddress> alfa_address,
                               std::shared_ptr<StringAddress> beta_address, int class_Ia,
                               int class_Ib) {
    if (!alfa) {
        size_t maxIa = alfa_address->strpcls(class_Ia);
        size_t maxIb = beta_address->strpcls(class_Ib);

        int block_idx = lists_.string_class()->block_index(class_Ia, class_Ib);
        auto m = M.view();
        auto c = C(block_idx).view();
        // Add m transposed to C
        for (size_t Ia = 0; Ia < maxIa; ++Ia)
            for (size_t Ib = 0; Ib < maxIb; ++Ib)
                c(Ia, Ib) += m(Ib, Ia);
    }
}

} // namespace forte2
