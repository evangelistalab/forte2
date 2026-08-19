#pragma once

#include <complex>
#include <functional>
#include <vector>
#include <cmath>
#include <span>

#include "helpers/ndarray.h"
#include "helpers/parallel.h"

#include "determinant/determinant.h"
#include "determinant/rel_slater_rules.h"
#include "dsrg/dsrg_utils.h"
#include "sci/sci_helper.h" // for ScreeningCriterion, EnergyCorrection, PT2Regularizer
#include "sci/sci_strings.h"

namespace forte2 {

/// @brief A hash map from determinants to complex accumulated couplings (used by the
/// reference selection path, the complex analogue of the real helper's DetMap)
using RelDetMap =
    ankerl::unordered_dense::map<Determinant, std::complex<double>, Determinant::Hash>;
/// @brief A hash map from determinants to their index in a (complex) coefficient vector
using RelDetRootMap = ankerl::unordered_dense::map<Determinant, size_t, Determinant::Hash>;
/// @brief A set of determinants (shared with the real helper via <sci/sci_helper.h>)
// DetSet is defined in sci/sci_helper.h

/// @brief A helper class for two-component (relativistic) selected CI methods.
///
/// This is the spinor-basis counterpart of `SelectedCIHelper`. It runs the same
/// Heat-Bath CI (HBCI) selection and sigma-vector build, but on complex Hermitian
/// integrals. Because every active electron occupies the "alpha" string with an empty
/// beta string (`nb == 0`), only the single-alpha and double-alpha-alpha excitation
/// classes ever contribute, so the beta / alpha-beta / spin machinery of the real helper
/// is absent here. The reduced density matrices (`compute_a_1rdm` / `compute_aa_2rdm`)
/// likewise only need the alpha string; they conjugate the bra so the diagonal RDMs are
/// Hermitian and the two-root case yields the transition RDM.
class RelSelectedCIHelper {
  public:
    // == Class Constructor ==
    /// @brief Construct a RelSelectedCIHelper object
    /// @param norb Number of spinors (active spin-orbitals)
    /// @param dets The initial determinants in the variational space
    /// @param c The initial CI coefficients for the determinants (shape: (n_dets, n_roots))
    /// @param E Hamiltonian scalar energy (real)
    /// @param H One-electron integrals, complex Hermitian (shape: (n_orb, n_orb))
    /// @param V Two-electron integrals in physicist notation <pq|rs>, complex
    ///          (shape: (n_orb, n_orb, n_orb, n_orb))
    /// @param log_level Logging level
    /// @param screening_criterion "hbci" (eHBCI is not supported in the 2C helper)
    /// @param frozen_creation List of orbitals that cannot be created into
    /// @param frozen_annihilation List of orbitals that cannot be annihilated from
    RelSelectedCIHelper(size_t norb, const std::vector<Determinant>& dets, np_matrix_complex& c,
                        double E, np_matrix_complex& H, np_tensor4_complex& V, int log_level = 3,
                        const std::string& screening_criterion = "hbci",
                        const std::vector<size_t>& frozen_creation = {},
                        const std::vector<size_t>& frozen_annihilation = {});

    // == Class Public Methods ==

    /// @brief Return the determinants in the variational space
    const std::vector<Determinant>& variational_dets() const { return dets_; }

    /// @brief Set the Hamiltonian integrals
    void set_Hamiltonian(double E, np_matrix_complex H, np_tensor4_complex V);

    /// @brief Set the CI coefficients (shape: (n_dets, n_roots))
    void set_c(np_matrix_complex& c);

    /// @brief Set the energy of the roots (as computed externally by the Davidson solver)
    void set_energies(np_vector e);

    /// @brief Set the number of batches per thread for parallelization
    void set_num_batches_per_thread(size_t n) { num_batches_per_thread_ = n; }

    /// @brief Set the orbitals excluded from the list of creation operators
    void set_frozen_creation(const std::vector<size_t>& frozen_creation);

    /// @brief Set the orbitals excluded from the list of annihilation operators
    void set_frozen_annihilation(const std::vector<size_t>& frozen_annihilation);

    /// @brief Set the screening criterion (only "hbci" is supported for the 2C helper)
    void set_screening_criterion(const std::string& criterion);

    /// @brief Set the energy correction method (variational or pt2)
    void set_energy_correction(const std::string& correction);

    /// @brief Set the PT2 regularization method (none, shift, dsrg)
    void set_pt2_regularizer(const std::string& regularizer, double strength = 0.5);

    /// @return The energies of the roots
    const std::vector<double>& energies() const { return root_energies_; }

    /// @return The PT2 energy contributions of the roots due to the new variational determinants
    const std::vector<double>& ept2_var() const { return ept2_var_; }

    /// @return The PT2 energy contributions of the roots due to the determinants excluded
    const std::vector<double>& ept2_pt() const { return ept2_pt_; }

    /// @return The number of determinants in the variational space
    const size_t num_dets_var() const { return dets_.size(); }

    /// @return The number of new variational determinants added in the last selection step
    size_t num_new_dets_var() const { return num_new_dets_var_; }

    /// @return The number of new PT2 determinants added in the last selection step
    size_t num_new_dets_pt2() const { return num_new_dets_pt2_; }

    /// @return The time taken for the last selection step
    double selection_time() const { return selection_time_; }

    /// @brief Perform HBCI selection with a reference implementation
    void select_hbci_ref(double var_threshold, double pt2_threshold);

    /// @brief Perform HBCI selection with an optimized batch implementation
    void select_hbci(double var_threshold, double pt2_threshold);

    /// @brief Compute the diagonal of the Hamiltonian matrix (real, since H is Hermitian)
    np_vector Hdiag() const;

    /// @brief Apply the Hamiltonian to a given basis and sigma vectors (complex)
    void Hamiltonian(np_vector_complex basis, np_vector_complex sigma) const;

    /// @brief Compute the (complex) alpha 1-RDM between two roots of the stored CI vectors.
    /// @param left_root The root supplying the (conjugated) bra coefficients
    /// @param right_root The root supplying the ket coefficients
    /// @return gamma1[p][q] = <left_root| a^+_p a_q |right_root> as a complex (norb, norb) matrix.
    ///         With left_root == right_root this is the ordinary 1-RDM; with left_root !=
    ///         right_root it is the transition 1-RDM. Only the alpha string contributes (nb == 0).
    np_matrix_complex compute_a_1rdm(size_t left_root, size_t right_root) const;

    /// @brief Compute the (complex) alpha-alpha 2-RDM between two roots of the stored CI vectors.
    /// @param left_root The root supplying the (conjugated) bra coefficients
    /// @param right_root The root supplying the ket coefficients
    /// @return gamma2[p][q][r][s] = <left_root| a^+_p a^+_q a_s a_r |right_root> as a complex
    ///         (norb, norb, norb, norb) tensor (full, antisymmetric in p<->q and r<->s). The
    ///         diagonal / transition distinction is the same as for compute_a_1rdm.
    np_tensor4_complex compute_aa_2rdm(size_t left_root, size_t right_root) const;

  private:
    // == Class Private Methods ==
    /// @brief Compute the energies of all determinants in the variational space
    void compute_det_energies();

    /// @brief Prepare the string lists for fast Hamiltonian application
    void prepare_strings();

    /// @brief Update the sorted two-electron integrals for fast HBCI selection
    void update_hbci_ints();

    /// @brief Compute the second-order energy contribution for a given determinant.
    /// `abs_v` is the magnitude |V_J| of the (complex) coupling.
    double compute_delta_ept2(double delta, double abs_v) const;

    /// @brief Single-alpha coupling <J|H|new_det> for the excitation i -> a, computed inline
    /// from the complex integrals (the 2C analogue of SlaterRules::singles_coupling_a).
    std::complex<double> singles_coupling_a(size_t i, size_t a, const Determinant& d) const;

    /// @brief Hamiltonian sub-blocks. Only H0 (diagonal), H1a (single alpha) and H2a (double
    /// alpha-alpha) contribute when the beta string is empty. Because every determinant is uniquely
    /// identified by its alpha string, H1a/H2a scatter directly into sigma (indexed by the string
    /// permutation) instead of going through a determinant hash lookup.
    void H0(std::span<std::complex<double>> basis, std::span<std::complex<double>> sigma) const;
    void H1a(std::span<std::complex<double>> basis, std::span<std::complex<double>> sigma) const;
    void H2a(std::span<std::complex<double>> basis, std::span<std::complex<double>> sigma) const;

    /// @brief Select new variational and PT2 determinants using a batch approach
    void select_hbci_batch(RelDetRootMap& V_map, RelDetRootMap& PT_map,
                           std::vector<std::complex<double>>& V_coeffs,
                           std::vector<std::complex<double>>& PT_coeffs, double var_threshold,
                           double pt2_threshold, size_t num_batches, size_t batch_id,
                           const DetSet& existing_dets);

    // == Class Private Variables ==

    /// @brief Threshold for integral screening
    static constexpr double integral_threshold = 1e-12;

    /// @brief Screening criterion for selected CI (only HBCI supported)
    ScreeningCriterion screening_criterion_ = ScreeningCriterion::HBCI;

    /// @brief Energy correction method for selected CI
    EnergyCorrection energy_correction_ = EnergyCorrection::PT2;

    /// @brief PT2 regularization method
    PT2Regularizer pt2_regularizer_ = PT2Regularizer::None;
    double pt2_regularizer_strength_ = 0.0;

    /// @brief logging level for the class
    int log_level_ = 3;

    /// @brief Number of batches per thread for parallelization
    size_t num_batches_per_thread_ = 1;

    /// @brief Number of spinors
    const size_t norb_;
    /// @brief Number of spinors squared/cubed
    const size_t norb2_;
    const size_t norb3_;

    /// @brief Number of alpha electrons (all active electrons; nb_ is enforced to be 0)
    size_t na_;
    /// @brief Number of beta electrons (must be 0 in the two-component basis)
    size_t nb_;

    /// @brief The scalar energy
    double E_;

    /// @brief The Slater rules for the current set of determinants (complex, spinor basis)
    RelSlaterRules slater_rules_;

    /// @brief Masks controlling which orbitals cannot be created into / annihilated from
    String frozen_creation_mask_ = String::zero();
    String frozen_annihilation_mask_ = String::zero();

    /// @brief Orbital energies: e[p] = Re <p|H|p> (real; H is Hermitian)
    std::vector<double> epsilon_;
    /// @brief One-electron integrals: h[p][q] = <p|H|q>
    std::vector<std::complex<double>> h_;
    /// @brief Two-electron integrals: V[p][q][r][s] = <pq|rs>
    std::vector<std::complex<double>> v_;
    /// @brief Antisymmetrized two-electron integrals: Va[p][q][r][s] = <pq||rs>
    std::vector<std::complex<double>> v_a_;

    /// @brief Sorted antisymmetrized two-electron integrals for fast HBCI selection.
    /// Each tuple is (criterion_key, integral, r, s) where the key is a real magnitude used
    /// for sorting and the sorted early-break, and the integral payload is complex.
    std::vector<std::vector<std::tuple<double, std::complex<double>, u_int32_t, u_int32_t>>>
        va_sorted_;

    /// @return The one-electron integral <i|h|j>
    inline const std::complex<double>& h(std::size_t i, std::size_t j) const noexcept {
        return h_[i * norb_ + j];
    }

    /// @return The two-electron integral <pq|rs>
    inline const std::complex<double>& V(std::size_t p, std::size_t q, std::size_t r,
                                         std::size_t s) const noexcept {
        return v_[p * norb3_ + q * norb2_ + r * norb_ + s];
    }

    /// @return The antisymmetrized two-electron integral <pq||rs>
    inline const std::complex<double>& Va(std::size_t p, std::size_t q, std::size_t r,
                                          std::size_t s) const noexcept {
        return v_a_[p * norb3_ + q * norb2_ + r * norb_ + s];
    }

    inline bool creation_allowed(std::size_t orbital) const noexcept {
        return !frozen_creation_mask_.get_bit(orbital);
    }
    inline bool annihilation_allowed(std::size_t orbital) const noexcept {
        return !frozen_annihilation_mask_.get_bit(orbital);
    }

    /// @brief The number of roots to compute
    size_t nroots_;
    /// @brief The initial guess for the CI coefficients
    np_matrix_complex c_guess_;

    /// @brief The variational determinants
    std::vector<Determinant> dets_;
    /// @brief The energies of the determinants in the variational space (real)
    std::vector<double> det_energies_;
    /// @brief The CI coefficients of the determinants in the variational space, stored as a flat
    /// vector of size dets_.size() * nroots_ (coefficient for det i, root r at index i*nroots_+r).
    std::vector<std::complex<double>> c_;

    /// @brief The alpha/beta strings for the determinants in the variational space
    SelectedCIStrings ab_list_;

    /// @brief The energies of the roots (real)
    std::vector<double> root_energies_;
    /// @brief The perturbative energy contributions due to the new variational determinants
    std::vector<double> ept2_var_;
    /// @brief The perturbative energy contributions due to the determinants excluded
    std::vector<double> ept2_pt_;

    /// @brief The number of new variational / PT2 determinants added in the last selection step
    size_t num_new_dets_var_ = 0;
    size_t num_new_dets_pt2_ = 0;

    /// @brief Time taken for the last selection step
    double selection_time_ = 0.0;
};

} // namespace forte2
