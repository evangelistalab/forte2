#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <span>

#include "helpers/ndarray.h"

#include "ci/ci_string_lists.h"
#include "ci/slater_rules.h"
#include "ci/ci_spin_adapter.h"

namespace forte2 {

class CISigmaBuilder {
  public:
    // == Class Constructor ==
    CISigmaBuilder(const CIStrings& lists, double E, np_matrix& H, np_tensor4& V);

    // == Class Public Functions ==
    void set_H(np_matrix H);
    void set_V(np_tensor4 V);

    /// @brief Form the diagonal of the Hamiltonian matrix in the CI basis
    /// @return The diagonal elements of the Hamiltonian matrix
    np_vector form_Hdiag_csf(const std::vector<Determinant>& dets,
                             const CISpinAdapter& spin_adapter,
                             bool spin_adapt_full_preconditioner) const;

    double slater_rules_csf(const std::vector<Determinant>& dets, const CISpinAdapter& spin_adapter,
                            size_t I, size_t J) const;

    /// @brief Apply the Hamiltonian to the wave function
    /// @param basis The basis vector
    /// @param sigma The resulting sigma vector |sigma> = H |basis>
    void Hamiltonian(np_vector basis, np_vector sigma) const;

    std::vector<double> avg_build_time() const {
        if (build_count_ == 0) {
            return {0.0, 0.0, 0.0, 0.0};
        } else {
            return {hdiag_timer_ / static_cast<double>(build_count_),
                    haabb_timer_ / static_cast<double>(build_count_),
                    haaaa_timer_ / static_cast<double>(build_count_),
                    hbbbb_timer_ / static_cast<double>(build_count_)};
        }
    }

    np_matrix compute_1rdm_same_irrep(np_vector C_left, np_vector C_right, bool alfa);
    np_matrix compute_sf_1rdm_same_irrep(np_vector C_left, np_vector C_right);
    np_matrix compute_2rdm_aa_same_irrep(np_vector C_left, np_vector C_right, bool alfa) const;
    np_tensor4 compute_2rdm_ab_same_irrep(np_vector C_left, np_vector C_right);

  private:
    // == Class Private Variables ==
    const CIStrings& lists_;
    double E_;
    np_matrix H_;
    np_tensor4 V_;
    SlaterRules slater_rules_;

    // == Class Mutable Variables ==
    mutable double hdiag_timer_ = 0.0;
    mutable double haaaa_timer_ = 0.0;
    mutable double haabb_timer_ = 0.0;
    mutable double hbbbb_timer_ = 0.0;
    mutable int build_count_ = 0;

    // == Class Static Variables ==

    mutable std::vector<double> TR;
    mutable std::vector<double> TL;
    mutable std::vector<double> h_pq;
    mutable std::vector<double> v_pr_qs;
    mutable std::vector<double> v_pr_qs_a;

    // == Class Private Functions ==
    void H0(std::span<double> basis, std::span<double> sigma) const;
    void H1_aa_gemm(std::span<double> basis, std::span<double> sigma, bool alfa) const;
    void H2_aaaa_gemm(std::span<double> basis, std::span<double> sigma, bool alfa) const;
    void H2_aabb_gemm(std::span<double> basis, std::span<double> sigma) const;
};

[[nodiscard]] std::span<double> gather_block(std::span<double> source, std::span<double> dest,
                                             bool alfa, const CIStrings& lists, int class_Ia,
                                             int class_Ib);

void zero_block(std::span<double> dest, bool alfa, const CIStrings& lists, int class_Ia,
                int class_Ib);

void scatter_block(std::span<double> source, std::span<double> dest, bool alfa,
                   const CIStrings& lists, int class_Ia, int class_Ib);

} // namespace forte2
