#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>

#include "ci/ci_strings.h"
#include "ci/ci_string_address.h"
#include "ci/ci_sigma_builder.h"
#include "ci/ci_spin_adapter.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_ci_strings_api(nb::module_& m) {
    nb::class_<CIStrings>(m, "CIStrings")
        .def(nb::init<size_t, size_t, int, std::vector<std::vector<int>>, std::vector<int>,
                      std::vector<int>>(),
             "na"_a, "nb"_a, "symmetry"_a, "orbital_symmetry"_a, "gas_min"_a, "gas_max"_a,
             "Initialize the CIStrings with number of alpha and beta electrons, symmetry, "
             "orbital symmetry, minimum and maximum number of electrons in each GAS space, and "
             "logging level")
        .def_prop_ro("alfa_address", &CIStrings::alfa_address)
        .def_prop_ro("na", &CIStrings::na)
        .def_prop_ro("nb", &CIStrings::nb)
        .def_prop_ro("symmetry", &CIStrings::symmetry)
        .def_prop_ro("nas", &CIStrings::nas)
        .def_prop_ro("nbs", &CIStrings::nbs)
        .def_prop_ro("ndet", &CIStrings::ndet)
        .def_prop_ro("ngas_spaces", &CIStrings::ngas_spaces)
        .def_prop_ro("gas_size", &CIStrings::gas_size)
        .def_prop_ro("gas_alfa_occupations", &CIStrings::gas_alfa_occupations)
        .def_prop_ro("gas_beta_occupations", &CIStrings::gas_beta_occupations)
        .def_prop_ro("gas_occupations", &CIStrings::gas_occupations)
        .def("determinant", &CIStrings::determinant, "address"_a)
        .def("determinant_index", &CIStrings::determinant_address, "d"_a)
        .def("make_determinants", &CIStrings::make_determinants);
}

void export_ci_sigma_builder_api(nb::module_& m) {
    nb::class_<CISigmaBuilder>(m, "CISigmaBuilder")
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
        .def("avg_build_time", &CISigmaBuilder::avg_build_time)
        .def("set_log_level", &CISigmaBuilder::set_log_level, "level"_a,
             "Set the logging level for the class")
        // RDMs debugging methods
        .def("a_1rdm_debug", &CISigmaBuilder::compute_a_1rdm_debug, "C_left"_a, "C_right"_a,
             "alfa"_a)
        .def("aa_2rdm_debug", &CISigmaBuilder::compute_aa_2rdm_debug, "C_left"_a, "C_right"_a,
             "alfa"_a,
             "Compute the two-electron same-spin reduced density matrix for debugging purposes")
        .def("ab_2rdm_debug", &CISigmaBuilder::compute_ab_2rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the two-electron mixed-spin reduced density matrix for debugging purposes")
        .def("aaa_3rdm_debug", &CISigmaBuilder::compute_aaa_3rdm_debug, "C_left"_a, "C_right"_a,
             "alfa"_a,
             "Compute the three-electron same-spin reduced density matrix for debugging purposes")
        .def("aab_3rdm_debug", &CISigmaBuilder::compute_aab_3rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the aab mixed-spin three-electron reduced density matrix for debugging "
             "purposes")
        .def("abb_3rdm_debug", &CISigmaBuilder::compute_abb_3rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the abb mixed-spin three-electron reduced density matrix for debugging "
             "purposes")
        .def("aaaa_4rdm_debug", &CISigmaBuilder::compute_aaaa_4rdm_debug, "C_left"_a, "C_right"_a,
             "alfa"_a,
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

void export_ci_spin_adapter_api(nb::module_& m) {
    nb::class_<CISpinAdapter>(m, "CISpinAdapter")
        .def(nb::init<int, int, int>(), "twoS"_a, "twoMs"_a, "norb"_a)
        .def("prepare_couplings", &CISpinAdapter::prepare_couplings, "dets"_a)
        .def("csf_C_to_det_C", &CISpinAdapter::csf_C_to_det_C, "csf_C"_a, "det_C"_a)
        .def("det_C_to_csf_C", &CISpinAdapter::det_C_to_csf_C, "det_C"_a, "csf_C"_a)
        .def("ncsf", [](CISpinAdapter& self) { return self.ncsf(); })
        .def("set_log_level", &CISpinAdapter::set_log_level, "level"_a,
             "Set the logging level for the class");
    // .def("ndet", &CISpinAdapter::ndet);
}
} // namespace forte2
