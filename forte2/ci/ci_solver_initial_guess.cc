#include "integrals/active_space_integrals.h"
#include "sparse_ci/determinant_functions.hpp"
#include "sparse_ci/ci_spin_adaptation.h"

#include "genci_solver.h"
#include "genci_vector.h"
#include "genci_string_lists.h"
#include "helpers/printing.h"
#include "helpers/string_algorithms.h"
#include "genci_string_address.h"

#include "sparse_ci/sparse_initial_guess.h"

namespace forte {

std::vector<Determinant> GenCISolver::initial_guess_generate_dets(std::shared_ptr<psi::Vector> diag,
                                                                  size_t num_guess_states) {
    size_t ndets = diag->dim();
    // number of guess to be used must be at most as large as the number of determinants
    size_t num_guess_dets = std::min(num_guess_states * ndets_per_guess_, ndets);

    // Get the address of the most important determinants
    // this list has size exactly num_guess_dets
    double emax = std::numeric_limits<double>::max();
    size_t added = 0;

    std::vector<std::tuple<double, size_t>> vec_e_I(num_guess_dets, std::make_tuple(emax, 0));

    for (size_t I = 0; I < ndets; ++I) {
        double e = diag->get(I);
        if ((e < emax) or (added < num_guess_states)) {
            // Find where to inser this determinant
            vec_e_I.pop_back();
            auto it = std::find_if(
                vec_e_I.begin(), vec_e_I.end(),
                [&e](const std::tuple<double, size_t>& t) { return e < std::get<0>(t); });
            vec_e_I.insert(it, std::make_tuple(e, I));
            emax = std::get<0>(vec_e_I.back());
            added++;
        }
    }

    std::vector<Determinant> guess_dets;
    for (const auto& [e, I] : vec_e_I) {
        guess_dets.push_back(lists_->determinant(I));
    }

    // Make sure that the spin space is complete
    enforce_spin_completeness(guess_dets, active_mo_.size());
    if (guess_dets.size() > num_guess_dets) {
        if (print_ >= PrintLevel::Brief) {
            psi::outfile->Printf("\n  Initial guess space is incomplete.\n  Adding "
                                 "%d determinant(s).",
                                 guess_dets.size() - num_guess_dets);
        }
    }
    return guess_dets;
}

std::pair<sparse_mat, sparse_mat>
GenCISolver::initial_guess_det(std::shared_ptr<psi::Vector> diag, size_t num_guess_states,
                               std::shared_ptr<ActiveSpaceIntegrals> fci_ints) {
    auto guess_dets = initial_guess_generate_dets(diag, num_guess_states);
    size_t num_guess_dets = guess_dets.size();

    std::vector<size_t> guess_dets_pos(num_guess_dets);
    for (size_t I = 0; I < num_guess_dets; ++I) {
        guess_dets_pos[I] = lists()->determinant_address(guess_dets[I]);
    }

    // here we use a standard guess procedure
    return find_initial_guess_det(guess_dets, guess_dets_pos, num_guess_states, fci_ints,
                                  state().multiplicity(), true, print_ >= PrintLevel::Default,
                                  std::vector<std::vector<std::pair<size_t, double>>>());
}

sparse_mat GenCISolver::initial_guess_csf(std::shared_ptr<psi::Vector> diag,
                                          size_t num_guess_states) {
    return find_initial_guess_csf(diag, num_guess_states, state().multiplicity(),
                                  print_ >= PrintLevel::Default);
}

std::shared_ptr<psi::Vector>
GenCISolver::form_Hdiag_csf(std::shared_ptr<ActiveSpaceIntegrals> fci_ints,
                            std::shared_ptr<SpinAdapter> spin_adapter) {
    auto Hdiag_csf = std::make_shared<psi::Vector>(spin_adapter->ncsf());
    // Compute the diagonal elements of the Hamiltonian in the CSF basis
    double E0 = fci_ints->nuclear_repulsion_energy() + fci_ints->scalar_energy();
    // Compute the diagonal elements of the Hamiltonian in the CSF basis
    if (spin_adapt_full_preconditioner_) {
        for (size_t i = 0, imax = spin_adapter->ncsf(); i < imax; ++i) {
            double energy = E0;
            int I = 0;
            for (const auto& [det_add_I, c_I] : spin_adapter_->csf(i)) {
                int J = 0;
                for (const auto& [det_add_J, c_J] : spin_adapter_->csf(i)) {
                    if (I == J) {
                        energy += c_I * c_J * fci_ints->energy(dets_[det_add_I]);
                    } else if (I < J) {
                        if (c_I * c_J != 0.0) {
                            energy += 2.0 * c_I * c_J *
                                      fci_ints->slater_rules(dets_[det_add_I], dets_[det_add_J]);
                        }
                    }
                    J++;
                }
                I++;
            }
            Hdiag_csf->set(i, energy);
        }
    } else {
        for (size_t i = 0, imax = spin_adapter->ncsf(); i < imax; ++i) {
            double energy = E0;
            for (const auto& [det_add_I, c_I] : spin_adapter_->csf(i)) {
                energy += c_I * c_I * fci_ints->energy(dets_[det_add_I]);
            }
            Hdiag_csf->set(i, energy);
        }
    }
    return Hdiag_csf;
}

} // namespace forte
