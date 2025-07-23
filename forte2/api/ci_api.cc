#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>

#include "ci/ci_string_lists.h"
#include "ci/ci_string_address.h"
#include "ci/ci_vector.h"
#include "ci/ci_sigma_builder.h"
#include "ci/ci_spin_adapter.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_ci_strings_api(nb::module_& m) {
    nb::class_<CIStrings>(m, "CIStrings")
        .def(nb::init<size_t, size_t, int, std::vector<std::vector<int>>, std::vector<int>,
                      std::vector<int>, int>(),
             "na"_a, "nb"_a, "symmetry"_a, "orbital_symmetry"_a, "gas_min"_a, "gas_max"_a,
             "log_level"_a = 3,
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
        .def("set_log_level", &CIStrings::set_log_level, "level"_a,
             "Set the logging level for the class")
        .def("determinant", &CIStrings::determinant, "address"_a)
        .def("determinant_index", &CIStrings::determinant_address, "d"_a)
        .def("make_determinants", &CIStrings::make_determinants);
    // .def_prop_ro("orbitals", &CIStrings::orbitals)
    // .def_prop_ro("orbital_symmetry", &CIStrings::orbital_symmetry)
    // .def_prop_ro("gas_min", &CIStrings::gas_min)
    // .def_prop_ro("gas_max", &CIStrings::gas_max);
}

void export_ci_vector_api(nb::module_& m) {
    nb::class_<CIVector>(m, "CIVector")
        .def(nb::init<const CIStrings&>(), "lists"_a)
        .def("copy", &CIVector::copy, "vec"_a)
        .def("copy_to", &CIVector::copy_to, "vec"_a);
}

void export_ci_sigma_builder_api(nb::module_& m) {
    nb::class_<CISigmaBuilder>(m, "CISigmaBuilder")
        .def(nb::init<const CIStrings&, double, np_matrix&, np_tensor4&, int>(), "lists"_a, "E"_a,
             "H"_a, "V"_a, "log_level"_a = 3,
             "Initialize the CISigmaBuilder with CIStrings, energy, Hamiltonian, and integrals")
        .def("set_algorithm", &CISigmaBuilder::set_algorithm, "algorithm"_a,
             "Set the sigma build algorithm (options = kh, hz)")
        .def("set_memory", &CISigmaBuilder::set_memory, "memory"_a,
             "Set the memory limit for the builder (in MB)")
        .def("form_Hdiag_csf", &CISigmaBuilder::form_Hdiag_csf, "dets"_a, "spin_adapter"_a,
             "spin_adapt_full_preconditioner"_a = false)
        .def("slater_rules_csf", &CISigmaBuilder::slater_rules_csf, "dets"_a, "spin_adapter"_a,
             "I"_a, "J"_a)
        .def("Hamiltonian", &CISigmaBuilder::Hamiltonian, "basis"_a, "sigma"_a)
        .def("rdm1_a", &CISigmaBuilder::compute_1rdm_same_irrep, "C_left"_a, "C_right"_a, "alfa"_a)
        .def("rdm2_aa", &CISigmaBuilder::compute_2rdm_aa_same_irrep, "C_left"_a, "C_right"_a,
             "alfa"_a)
        .def("rdm2_aa_full", &CISigmaBuilder::compute_2rdm_aa_same_irrep_full, "C_left"_a,
             "C_right"_a, "alfa"_a)
        .def("rdm2_ab", &CISigmaBuilder::compute_2rdm_ab_same_irrep, "C_left"_a, "C_right"_a)
        .def("rdm3_aaa", &CISigmaBuilder::compute_3rdm_aaa_same_irrep, "C_left"_a, "C_right"_a,
             "alfa"_a)
        .def("rdm3_aab", &CISigmaBuilder::compute_3rdm_aab_same_irrep, "C_left"_a, "C_right"_a)
        .def("rdm3_abb", &CISigmaBuilder::compute_3rdm_abb_same_irrep, "C_left"_a, "C_right"_a)
        .def("rdm1_sf", &CISigmaBuilder::compute_sf_1rdm_same_irrep, "C_left"_a, "C_right"_a)
        .def("rdm2_sf", &CISigmaBuilder::compute_sf_2rdm_same_irrep, "C_left"_a, "C_right"_a)
        .def("avg_build_time", &CISigmaBuilder::avg_build_time)
        .def("set_log_level", &CISigmaBuilder::set_log_level, "level"_a,
             "Set the logging level for the class")
        // The following methods are for debugging purposes
        .def("rdm1_a_debug", &CISigmaBuilder::compute_1rdm_a_debug, "C_left"_a, "C_right"_a,
             "alfa"_a)
        .def("rdm2_aa_debug", &CISigmaBuilder::compute_2rdm_aa_debug, "C_left"_a, "C_right"_a,
             "alfa"_a,
             "Compute the two-electron same-spin reduced density matrix for debugging purposes")
        .def("rdm2_ab_debug", &CISigmaBuilder::compute_2rdm_ab_debug, "C_left"_a, "C_right"_a,
             "Compute the two-electron mixed-spin reduced density matrix for debugging purposes")
        .def("rdm3_aaa_debug", &CISigmaBuilder::compute_3rdm_aaa_debug, "C_left"_a, "C_right"_a,
             "alfa"_a,
             "Compute the three-electron same-spin reduced density matrix for debugging purposes")
        .def("rdm3_aab_debug", &CISigmaBuilder::compute_3rdm_aab_debug, "C_left"_a, "C_right"_a,
             "Compute the aab mixed-spin three-electron reduced density matrix for debugging "
             "purposes")
        .def("rdm3_abb_debug", &CISigmaBuilder::compute_3rdm_abb_debug, "C_left"_a, "C_right"_a,
             "Compute the abb mixed-spin three-electron reduced density matrix for debugging "
             "purposes")
        .def("rdm4_aaaa_debug", &CISigmaBuilder::compute_4rdm_aaaa_debug, "C_left"_a, "C_right"_a,
             "alfa"_a,
             "Compute the four-electron same-spin reduced density matrix for debugging purposes")
        .def("rdm4_aaab_debug", &CISigmaBuilder::compute_4rdm_aaab_debug, "C_left"_a, "C_right"_a,
             "Compute the aaab mixed-spin four-electron reduced density matrix for debugging "
             "purposes")
        .def("rdm4_aabb_debug", &CISigmaBuilder::compute_4rdm_aabb_debug, "C_left"_a, "C_right"_a,
             "Compute the aabb mixed-spin four-electron reduced density matrix for debugging "
             "purposes")
        .def("rdm4_abbb_debug", &CISigmaBuilder::compute_4rdm_abbb_debug, "C_left"_a, "C_right"_a,
             "Compute the abbb mixed-spin four-electron reduced density matrix for debugging "
             "purposes")
        .def("rdm1_sf_debug", &CISigmaBuilder::compute_sf_1rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the spin-free one-electron reduced density matrix for debugging purposes")
        .def("rdm2_sf_debug", &CISigmaBuilder::compute_sf_2rdm_debug, "C_left"_a, "C_right"_a,
             "Compute the spin-free two-electron reduced density matrix for debugging purposes");
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
