#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/complex.h>
#include <nanobind/ndarray.h>

#include "ci/ci_strings.h"
#include "ci/ci_string_address.h"
#include "ci/ci_sigma_builder.h"
#include "determinant/ci_spin_adapter.h"
#include "ci/rel_ci_sigma_builder.h"
#include "sci/sci_helper.h"
#include "sci/rel_sci_helper.h"

// Must be at global scope:
NB_MAKE_OPAQUE(std::vector<forte2::Determinant>);

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

namespace {
void export_ci_strings_api(nb::module_& m);
void export_ci_sigma_builder_api(nb::module_& m);
void export_ci_spin_adapter_api(nb::module_& m);
void export_rel_ci_sigma_builder_api(nb::module_& m);
void export_sci_helper_api(nb::module_& m);
void export_rel_sci_helper_api(nb::module_& m);
} // namespace

void export_ci_helpers_api(nb::module_& m) {
    nb::module_ sub_m = m.def_submodule("ci_helpers", "CI and selected CI helper classes");

    export_ci_strings_api(sub_m);

    export_ci_sigma_builder_api(sub_m);

    export_ci_spin_adapter_api(sub_m);

    export_rel_ci_sigma_builder_api(sub_m);

    export_sci_helper_api(sub_m);

    export_rel_sci_helper_api(sub_m);
}

namespace {
void export_ci_strings_api(nb::module_& sub_m) {
    nb::bind_vector<std::vector<forte2::Determinant>>(sub_m, "DeterminantVector");

    nb::class_<CIStrings>(sub_m, "CIStrings")
        .def(
            "__init__",
            [](CIStrings* self, Py_ssize_t na, Py_ssize_t nb, int symmetry,
               std::vector<std::vector<int>> orbital_symmetry, std::vector<int> gas_min,
               std::vector<int> gas_max) {
                if (na < 0)
                    throw nb::value_error("Number of alpha electrons must be non-negative.");
                if (nb < 0)
                    throw nb::value_error("Number of beta electrons must be non-negative.");
                new (self) CIStrings(static_cast<size_t>(na), static_cast<size_t>(nb), symmetry,
                                     orbital_symmetry, gas_min, gas_max);
            },
            "na"_a, "nb"_a, "symmetry"_a, "orbital_symmetry"_a, "gas_min"_a, "gas_max"_a,
            "Initialize the CIStrings with number of alpha and beta electrons, symmetry, "
            "orbital symmetry, minimum and maximum number of electrons in each GAS space")
        .def_prop_ro("alpha_address", &CIStrings::alpha_address)
        .def_prop_ro("na", &CIStrings::na)
        .def_prop_ro("nb", &CIStrings::nb)
        .def_prop_ro("symmetry", &CIStrings::symmetry)
        .def_prop_ro("nas", &CIStrings::nas)
        .def_prop_ro("nbs", &CIStrings::nbs)
        .def_prop_ro("ndet", &CIStrings::ndet)
        .def_prop_ro("ngas_spaces", &CIStrings::ngas_spaces)
        .def_prop_ro("gas_size", &CIStrings::gas_size)
        .def_prop_ro("gas_alpha_occupations", &CIStrings::gas_alpha_occupations)
        .def_prop_ro("gas_beta_occupations", &CIStrings::gas_beta_occupations)
        .def_prop_ro("gas_occupations", &CIStrings::gas_occupations)
        .def("determinant", &CIStrings::determinant, "address"_a)
        .def("determinant_index", &CIStrings::determinant_address, "d"_a)
        .def("make_determinants", &CIStrings::make_determinants);
}

void export_ci_sigma_builder_api(nb::module_& sub_m) {
    nb::class_<CISigmaBuilder>(sub_m, "CISigmaBuilder")
        .def(nb::init<const CIStrings&, double, np_matrix&, np_tensor4&, int>(), "lists"_a, "E"_a,
             "H"_a, "V"_a, "log_level"_a = 3,
             "Initialize the CISigmaBuilder with CIStrings, energy, Hamiltonian, and integrals")
        .def("set_algorithm", &CISigmaBuilder::set_algorithm, "algorithm"_a,
             "Set the sigma build algorithm (options = kh, hz)")
        .def("get_algorithm", &CISigmaBuilder::get_algorithm,
             "Get the current sigma build algorithm")
        .def("set_memory", &CISigmaBuilder::set_memory, "memory"_a,
             "Set the memory limit for the builder (in MB)")
        .def("form_Hdiag_csf", &CISigmaBuilder::form_Hdiag_csf, "dets"_a, "spin_adapter"_a,
             "spin_adapt_full_preconditioner"_a = false)
        .def("energy_csf", &CISigmaBuilder::energy_csf, "dets"_a, "spin_adapter"_a, "I"_a,
             "Compute the energy of a CSF")
        .def("form_H_csf", &CISigmaBuilder::form_H_csf, "dets"_a, "spin_adapter"_a,
             "Form the full Hamiltonian matrix in the CSF basis")
        .def("slater_rules_csf", &CISigmaBuilder::slater_rules_csf, "dets"_a, "spin_adapter"_a,
             "I"_a, "J"_a)
        .def("Hamiltonian", &CISigmaBuilder::Hamiltonian, "basis"_a, "sigma"_a)
        .def("make_sparse_state", &CISigmaBuilder::make_sparse_state, "C"_a, "threshold"_a = 1e-12,
             "Convert a CI vector to a sparse state")
        // Spin-free RDMs and cumulants
        .def("sf_1rdm", &CISigmaBuilder::compute_sf_1rdm, "C_left"_a, "C_right"_a,
             "Compute the spin-free one-electron reduced density matrix")
        .def("sf_2rdm", &CISigmaBuilder::compute_sf_2rdm, "C_left"_a, "C_right"_a,
             "Compute the spin-free two-electron reduced density matrix")
        .def("sf_3rdm", &CISigmaBuilder::compute_sf_3rdm, "C_left"_a, "C_right"_a,
             "Compute the spin-free three-electron reduced density matrix")
        .def("sf_2cumulant", &CISigmaBuilder::compute_sf_2cumulant, "C_left"_a, "C_right"_a,
             "Compute the spin-free two-electron cumulant")
        .def("sf_3cumulant", &CISigmaBuilder::compute_sf_3cumulant, "C_left"_a, "C_right"_a,
             "Compute the spin-free three-electron cumulant")
        // Spinful RDMs
        .def("a_1rdm", &CISigmaBuilder::compute_a_1rdm, "C_left"_a, "C_right"_a,
             "Compute the alpha one-electron reduced density matrix")
        .def("b_1rdm", &CISigmaBuilder::compute_b_1rdm, "C_left"_a, "C_right"_a,
             "Compute the beta one-electron reduced density matrix")
        .def("aa_2rdm", &CISigmaBuilder::compute_aa_2rdm, "C_left"_a, "C_right"_a,
             "Compute the alpha-alpha two-electron reduced density matrix")
        .def("bb_2rdm", &CISigmaBuilder::compute_bb_2rdm, "C_left"_a, "C_right"_a,
             "Compute the beta-beta two-electron reduced density matrix")
        .def("ab_2rdm", &CISigmaBuilder::compute_ab_2rdm, "C_left"_a, "C_right"_a,
             "Compute the alpha-beta two-electron reduced density matrix")
        .def("aaa_3rdm", &CISigmaBuilder::compute_aaa_3rdm, "C_left"_a, "C_right"_a,
             "Compute the alpha-alpha-alpha three-electron reduced density matrix")
        .def("aab_3rdm", &CISigmaBuilder::compute_aab_3rdm, "C_left"_a, "C_right"_a,
             "Compute the alpha-alpha-beta three-electron reduced density matrix")
        .def("abb_3rdm", &CISigmaBuilder::compute_abb_3rdm, "C_left"_a, "C_right"_a,
             "Compute the alpha-beta-beta three-electron reduced density matrix")
        .def("bbb_3rdm", &CISigmaBuilder::compute_bbb_3rdm, "C_left"_a, "C_right"_a,
             "Compute the beta-beta-beta three-electron reduced density matrix")
        .def("a_1trdm", &CISigmaBuilder::compute_a_1trdm, "sigmabuilder_right"_a, "C_left"_a,
             "C_right"_a, "Compute the alpha one-electron transition reduced density matrix")
        .def("b_1trdm", &CISigmaBuilder::compute_b_1trdm, "sigmabuilder_right"_a, "C_left"_a,
             "C_right"_a, "Compute the beta one-electron transition reduced density matrix")
        .def("sf_1trdm", &CISigmaBuilder::compute_sf_1trdm, "sigmabuilder_right"_a, "C_left"_a,
             "C_right"_a, "Compute the spin-free one-electron transition reduced density matrix")
        .def("avg_build_time", &CISigmaBuilder::avg_build_time)
        .def("set_log_level", &CISigmaBuilder::set_log_level, "level"_a,
             "Set the logging level for the class")
        // RDMs debugging methods
        .def("a_1rdm_debug", &CISigmaBuilder::compute_a_1rdm_debug, "C_left"_a, "C_right"_a,
             "alpha"_a)
        .def("aa_2rdm_debug", &CISigmaBuilder::compute_aa_2rdm_debug, "C_left"_a, "C_right"_a,
             "alpha"_a,
             "Compute the two-electron same-spin reduced density matrix for debugging purposes")
        .def("ab_2rdm_debug", &CISigmaBuilder::compute_ab_2rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the two-electron mixed-spin reduced density matrix for debugging purposes")
        .def("aaa_3rdm_debug", &CISigmaBuilder::compute_aaa_3rdm_debug, "C_left"_a, "C_right"_a,
             "alpha"_a,
             "Compute the three-electron same-spin reduced density matrix for debugging purposes")
        .def("aab_3rdm_debug", &CISigmaBuilder::compute_aab_3rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the aab mixed-spin three-electron reduced density matrix for debugging "
             "purposes")
        .def("abb_3rdm_debug", &CISigmaBuilder::compute_abb_3rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the abb mixed-spin three-electron reduced density matrix for debugging "
             "purposes")
        .def("aaaa_4rdm_debug", &CISigmaBuilder::compute_aaaa_4rdm_debug, "C_left"_a, "C_right"_a,
             "alpha"_a,
             "Compute the four-electron same-spin reduced density matrix for debugging purposes")
        .def("aaab_4rdm_debug", &CISigmaBuilder::compute_aaab_4rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the aaab mixed-spin four-electron reduced density matrix for debugging "
             "purposes")
        .def("aabb_4rdm_debug", &CISigmaBuilder::compute_aabb_4rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the aabb mixed-spin four-electron reduced density matrix for debugging "
             "purposes")
        .def("abbb_4rdm_debug", &CISigmaBuilder::compute_abbb_4rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the abbb mixed-spin four-electron reduced density matrix for debugging "
             "purposes")
        .def("sf_1rdm_debug", &CISigmaBuilder::compute_sf_1rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the spin-free one-electron reduced density matrix for debugging purposes")
        .def("sf_2rdm_debug", &CISigmaBuilder::compute_sf_2rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the spin-free two-electron reduced density matrix for debugging purposes")
        .def("sf_3rdm_debug", &CISigmaBuilder::compute_sf_3rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the spin-free three-electron reduced density matrix for debugging purposes")
        .def("sf_2cumulant_debug", &CISigmaBuilder::compute_sf_2cumulant_debug, "C_left"_a,
             "C_right"_a, "Compute the spin-free two-electron cumulant for debugging purposes")
        .def("sf_3cumulant_debug", &CISigmaBuilder::compute_sf_3cumulant_debug, "C_left"_a,
             "C_right"_a, "Compute the spin-free three-electron cumulant for debugging purposes");
}

void export_ci_spin_adapter_api(nb::module_& sub_m) {
    nb::class_<CISpinAdapter>(sub_m, "CISpinAdapter")
        .def(nb::init<int, int, int>(), "twoS"_a, "twoMs"_a, "norb"_a)
        .def("prepare_couplings", &CISpinAdapter::prepare_couplings, "dets"_a)
        .def("csf_C_to_det_C", &CISpinAdapter::csf_C_to_det_C, "csf_C"_a, "det_C"_a)
        .def("det_C_to_csf_C", &CISpinAdapter::det_C_to_csf_C, "det_C"_a, "csf_C"_a)
        .def_prop_ro("nconf", &CISpinAdapter::nconf)
        .def_prop_ro("ncsf", &CISpinAdapter::ncsf)
        .def("set_log_level", &CISpinAdapter::set_log_level, "level"_a,
             "Set the logging level for the class");
    // .def("ndet", &CISpinAdapter::ndet);
}

void export_rel_ci_sigma_builder_api(nb::module_& sub_m) {
    nb::class_<RelCISigmaBuilder>(sub_m, "RelCISigmaBuilder")
        .def(nb::init<const CIStrings&, double, np_matrix_complex&, np_tensor4_complex&, int>(),
             "lists"_a, "E"_a, "H"_a, "V"_a, "log_level"_a = 3,
             "Initialize the CISigmaBuilder with CIStrings, energy, Hamiltonian, and integrals")
        .def("set_algorithm", &RelCISigmaBuilder::set_algorithm, "algorithm"_a,
             "Set the sigma build algorithm (options = kh, hz)")
        .def("get_algorithm", &RelCISigmaBuilder::get_algorithm,
             "Get the current sigma build algorithm")
        .def("set_memory", &RelCISigmaBuilder::set_memory, "memory"_a,
             "Set the memory limit for the builder (in MB)")
        .def("form_Hdiag", &RelCISigmaBuilder::form_Hdiag, "dets"_a)
        .def("slater_rules", &RelCISigmaBuilder::slater_rules, "dets"_a, "I"_a, "J"_a)
        .def("Hamiltonian", &RelCISigmaBuilder::Hamiltonian, "basis"_a, "sigma"_a)
        .def("so_1rdm", &RelCISigmaBuilder::compute_1rdm, "C_left"_a, "C_right"_a,
             "Compute the spin-orbital one-electron reduced density matrix")
        .def("so_2rdm", &RelCISigmaBuilder::compute_2rdm, "C_left"_a, "C_right"_a,
             "Compute the spin-orbital two-electron reduced density matrix")
        .def("so_2cumulant", &RelCISigmaBuilder::compute_2cumulant, "C_left"_a, "C_right"_a,
             "Compute the spin-orbital two-electron cumulant")
        .def("so_3rdm", &RelCISigmaBuilder::compute_3rdm, "C_left"_a, "C_right"_a,
             "Compute the spin-orbital three-electron reduced density matrix")
        .def("so_3cumulant", &RelCISigmaBuilder::compute_3cumulant, "C_left"_a, "C_right"_a,
             "Compute the spin-orbital three-electron cumulant")
        .def("so_1rdm_debug", &RelCISigmaBuilder::compute_1rdm_debug, "C_left"_a, "C_right"_a)
        .def("so_2rdm_debug", &RelCISigmaBuilder::compute_2rdm_debug, "C_left"_a, "C_right"_a)
        .def("so_3rdm_debug", &RelCISigmaBuilder::compute_3rdm_debug, "C_left"_a, "C_right"_a);
}

void export_sci_helper_api(nb::module_& sub_m) {
    nb::class_<SelectedCIHelper>(sub_m, "SelectedCIHelper")
        .def(nb::init<size_t, const std::vector<Determinant>&, np_matrix&, double, np_matrix&,
                      np_tensor4&, int, const std::string&, const std::vector<size_t>&,
                      const std::vector<size_t>&>(),
             "norb"_a, "dets"_a, "c"_a, "E"_a, "H"_a, "V"_a, "log_level"_a = 3,
             "screening_criterion"_a = "hbci", "frozen_creation"_a = std::vector<size_t>{},
             "frozen_annihilation"_a = std::vector<size_t>{},
             "Initialize the SelectedCIHelper with the number of orbitals, initial determinants, "
             "energy, Hamiltonian, and integrals")
        .def("set_Hamiltonian", &SelectedCIHelper::set_Hamiltonian, "E"_a, "H"_a, "V"_a,
             "Set the Hamiltonian integrals")
        .def("Hamiltonian", &SelectedCIHelper::Hamiltonian, "basis"_a, "sigma"_a,
             "Apply the Hamiltonian to the basis and store the result in sigma")
        .def("Hdiag", &SelectedCIHelper::Hdiag, "Return the diagonal of the Hamiltonian matrix")
        .def("set_c", &SelectedCIHelper::set_c, "c"_a, "Set the CI coefficients")
        .def("set_num_threads", &SelectedCIHelper::set_num_threads, "n"_a,
             "Set the number of threads to use in parallel sections")
        .def("set_num_batches_per_thread", &SelectedCIHelper::set_num_batches_per_thread, "n"_a,
             "Set the number of batches each thread will process in parallel sections")
        .def("set_energies", &SelectedCIHelper::set_energies, "e"_a,
             "Set the energies of the roots")
        .def("set_frozen_creation", &SelectedCIHelper::set_frozen_creation, "frozen_creation"_a,
             "Set orbitals excluded from creation in selection")
        .def("set_frozen_annihilation", &SelectedCIHelper::set_frozen_annihilation,
             "frozen_annihilation"_a, "Set orbitals excluded from annihilation in selection")
        .def("set_screening_criterion", &SelectedCIHelper::set_screening_criterion, "criterion"_a,
             "Set the screening criterion for selection ('hbci' or 'ehbci')")
        .def("set_energy_correction", &SelectedCIHelper::set_energy_correction, "correction"_a,
             "Set the energy correction method for selection ('variational' or 'pt2')")
        .def("set_pt2_regularizer", &SelectedCIHelper::set_pt2_regularizer, "regularizer"_a,
             "strength"_a = 0.5,
             "Set the PT2 regularization method ('none', 'shift', 'dsrg') and its strength")
        .def("select_hbci_ref", &SelectedCIHelper::select_hbci_ref, "var_threshold"_a,
             "pt2_threshold"_a, "Perform HBCI selection with the given threshold")
        .def("select_hbci", &SelectedCIHelper::select_hbci, "var_threshold"_a, "pt2_threshold"_a,
             "Perform HBCI selection with the given thresholds")
        .def("compute_spin2", &SelectedCIHelper::compute_spin2,
             "Compute the expectation value of S^2 for each root and return as a list")
        .def("a_1rdm", &SelectedCIHelper::compute_a_1rdm, "left_root"_a, "right_root"_a,
             "Compute the alpha-spin 1-RDM between two roots")
        .def("b_1rdm", &SelectedCIHelper::compute_b_1rdm, "left_root"_a, "right_root"_a,
             "Compute the beta-spin 1-RDM between two roots")
        .def("sf_1rdm", &SelectedCIHelper::compute_sf_1rdm, "left_root"_a, "right_root"_a,
             "Compute the spin-free 1-RDM between two roots")
        .def("aa_2rdm", &SelectedCIHelper::compute_aa_2rdm, "left_root"_a, "right_root"_a,
             "Compute the alpha-alpha 2-RDM between two roots")
        .def("bb_2rdm", &SelectedCIHelper::compute_bb_2rdm, "left_root"_a, "right_root"_a,
             "Compute the beta-beta 2-RDM between two roots")
        .def("ab_2rdm", &SelectedCIHelper::compute_ab_2rdm, "left_root"_a, "right_root"_a,
             "Compute the alpha-beta 2-RDM between two roots")
        .def("sf_2rdm", &SelectedCIHelper::compute_sf_2rdm, "left_root"_a, "right_root"_a,
             "Compute the spin-free 2-RDM between two roots")
        .def("a_1trdm", &SelectedCIHelper::compute_a_1trdm, "right_helper"_a, "left_root"_a,
             "right_root"_a,
             "Compute the alpha-spin 1-transition RDM between two roots in different helpers")
        .def("b_1trdm", &SelectedCIHelper::compute_b_1trdm, "right_helper"_a, "left_root"_a,
             "right_root"_a,
             "Compute the beta-spin 1-transition RDM between two roots in different helpers")
        .def("sf_1trdm", &SelectedCIHelper::compute_sf_1trdm, "right_helper"_a, "left_root"_a,
             "right_root"_a,
             "Compute the spin-free 1-transition RDM between two roots in different helpers")
        .def("dets", &SelectedCIHelper::variational_dets,
             "Return the determinants in the variational space")
        .def("ndets", &SelectedCIHelper::num_dets_var,
             "Return the number of determinants in the variational space")
        .def("energies", &SelectedCIHelper::energies, "Return the energies of the roots")
        .def("ept2_var", &SelectedCIHelper::ept2_var,
             "Return the variational part of the Epstein-Nesbet second-order energy correction")
        .def("ept2_pt", &SelectedCIHelper::ept2_pt,
             "Return the perturbative part of the Epstein-Nesbet second-order energy correction")
        .def("num_new_dets_var", &SelectedCIHelper::num_new_dets_var,
             "Return the number of new variational determinants added in the last selection")
        .def("num_new_dets_pt2", &SelectedCIHelper::num_new_dets_pt2,
             "Return the number of new perturbative determinants added in the last selection")
        .def("selection_time", &SelectedCIHelper::selection_time,
             "Return the total selection time");
}
void export_rel_sci_helper_api(nb::module_& sub_m) {
    // Two-component (relativistic) selected CI helper. Mirrors SelectedCIHelper but with complex
    // Hermitian integrals and CI coefficients, and without the beta / spin machinery (nb == 0).
    // Only the alpha 1-/2-RDMs are exposed (the beta / alpha-beta / spin-free variants are absent
    // in the spinor basis).
    nb::class_<RelSelectedCIHelper>(sub_m, "RelSelectedCIHelper")
        .def(nb::init<size_t, const std::vector<Determinant>&, np_matrix_complex&, double,
                      np_matrix_complex&, np_tensor4_complex&, int, const std::string&,
                      const std::vector<size_t>&, const std::vector<size_t>&>(),
             "norb"_a, "dets"_a, "c"_a, "E"_a, "H"_a, "V"_a, "log_level"_a = 3,
             "screening_criterion"_a = "hbci", "frozen_creation"_a = std::vector<size_t>{},
             "frozen_annihilation"_a = std::vector<size_t>{},
             "Initialize the RelSelectedCIHelper with the number of spinors, initial determinants, "
             "energy, complex Hamiltonian, and complex integrals")
        .def("set_Hamiltonian", &RelSelectedCIHelper::set_Hamiltonian, "E"_a, "H"_a, "V"_a,
             "Set the (complex) Hamiltonian integrals")
        .def("Hamiltonian", &RelSelectedCIHelper::Hamiltonian, "basis"_a, "sigma"_a,
             "Apply the Hamiltonian to the (complex) basis and store the result in sigma")
        .def("Hdiag", &RelSelectedCIHelper::Hdiag,
             "Return the (real) diagonal of the Hamiltonian matrix")
        .def("set_c", &RelSelectedCIHelper::set_c, "c"_a, "Set the (complex) CI coefficients")
        .def("set_num_threads", &RelSelectedCIHelper::set_num_threads, "n"_a,
             "Set the number of threads to use in parallel sections")
        .def("set_num_batches_per_thread", &RelSelectedCIHelper::set_num_batches_per_thread, "n"_a,
             "Set the number of batches each thread will process in parallel sections")
        .def("set_energies", &RelSelectedCIHelper::set_energies, "e"_a,
             "Set the energies of the roots")
        .def("set_frozen_creation", &RelSelectedCIHelper::set_frozen_creation, "frozen_creation"_a,
             "Set orbitals excluded from creation in selection")
        .def("set_frozen_annihilation", &RelSelectedCIHelper::set_frozen_annihilation,
             "frozen_annihilation"_a, "Set orbitals excluded from annihilation in selection")
        .def("set_screening_criterion", &RelSelectedCIHelper::set_screening_criterion,
             "criterion"_a, "Set the screening criterion for selection (only 'hbci' is supported)")
        .def("set_energy_correction", &RelSelectedCIHelper::set_energy_correction, "correction"_a,
             "Set the energy correction method for selection ('variational' or 'pt2')")
        .def("set_pt2_regularizer", &RelSelectedCIHelper::set_pt2_regularizer, "regularizer"_a,
             "strength"_a = 0.5,
             "Set the PT2 regularization method ('none', 'shift', 'dsrg') and its strength")
        .def("select_hbci_ref", &RelSelectedCIHelper::select_hbci_ref, "var_threshold"_a,
             "pt2_threshold"_a, "Perform HBCI selection with the reference implementation")
        .def("select_hbci", &RelSelectedCIHelper::select_hbci, "var_threshold"_a, "pt2_threshold"_a,
             "Perform HBCI selection with the batched implementation")
        .def("a_1rdm", &RelSelectedCIHelper::compute_a_1rdm, "left_root"_a, "right_root"_a,
             "Compute the complex alpha 1-RDM (or transition 1-RDM) between two roots")
        .def("aa_2rdm", &RelSelectedCIHelper::compute_aa_2rdm, "left_root"_a, "right_root"_a,
             "Compute the complex alpha-alpha 2-RDM (or transition 2-RDM) between two roots")
        .def("dets", &RelSelectedCIHelper::variational_dets,
             "Return the determinants in the variational space")
        .def("ndets", &RelSelectedCIHelper::num_dets_var,
             "Return the number of determinants in the variational space")
        .def("energies", &RelSelectedCIHelper::energies, "Return the energies of the roots")
        .def("ept2_var", &RelSelectedCIHelper::ept2_var,
             "Return the variational part of the Epstein-Nesbet second-order energy correction")
        .def("ept2_pt", &RelSelectedCIHelper::ept2_pt,
             "Return the perturbative part of the Epstein-Nesbet second-order energy correction")
        .def("num_new_dets_var", &RelSelectedCIHelper::num_new_dets_var,
             "Return the number of new variational determinants added in the last selection")
        .def("num_new_dets_pt2", &RelSelectedCIHelper::num_new_dets_pt2,
             "Return the number of new perturbative determinants added in the last selection")
        .def("selection_time", &RelSelectedCIHelper::selection_time,
             "Return the total selection time");
}
} // namespace

} // namespace forte2
