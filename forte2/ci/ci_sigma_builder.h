#pragma once

#include <functional>
#include <vector>
#include <cmath>

#include "helpers/ndarray.h"

#include "ci_string_lists.h"
#include "ci/slater_rules.h"
#include "ci_spin_adapter.h"

namespace forte2 {

class CISigmaBuilder {
  public:
    // == Class Constructor ==
    CISigmaBuilder(const CIStrings& lists, double E, np_matrix& H, np_tensor4& V);

    // == Class Public Functions ==
    // /// @brief Form the diagonal of the Hamiltonian matrix in the CI basis
    // /// @return The diagonal elements of the Hamiltonian matrix
    // np_vector form_Hdiag_det() const;

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

    double avg_build_time() const {
        return build_count_ > 0 ? hdiag_timer_ / static_cast<double>(build_count_) : 0.0;
    }

    // == Class Static Functions ==
    // Temporary memory allocation
    static void allocate_temp_space(const CIStrings& lists);
    static void release_temp_space();

  private:
    // == Class Private Variables ==
    const CIStrings& lists_;
    double E_;
    np_matrix H_;
    np_tensor4 V_;
    SlaterRules slater_rules_;

    // == Class Mutable Variables ==
    mutable double hdiag_timer_ = 0.0;
    mutable int build_count_ = 0;

    // == Class Static Variables ==
    /// Temporary matrix as large as the largest block of C for the right coefficient vector
    static np_matrix CR;
    /// Temporary matrix as large as the largest block of C for the left coefficient vector
    static np_matrix CL;

    // == Class Private Functions ==
    void H0(np_vector basis, np_vector sigma) const;
    void H1(np_vector basis, np_vector sigma, bool alfa) const;
    void H2_aaaa(np_vector basis, np_vector sigma, bool alfa) const;
    void H2_aabb(np_vector basis, np_vector sigma) const;
    void H2_aabb_gather_scatter(np_vector basis, np_vector sigma) const;

    static void gather_C_block(np_vector source, np_matrix dest, bool alfa, const CIStrings& lists,
                               int class_Ia, int class_Ib);

    static void zero_block(np_matrix dest, bool alfa, const CIStrings& lists, int class_Ia,
                           int class_Ib);

    static void scatter_C_block(np_matrix source, np_vector dest, bool alfa, const CIStrings& lists,
                                int class_Ia, int class_Ib);
};

} // namespace forte2
