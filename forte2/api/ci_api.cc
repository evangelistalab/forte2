#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
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
                      std::vector<int>>(),
             "na"_a, "nb"_a, "symmetry"_a, "orbital_symmetry"_a, "gas_min"_a, "gas_max"_a)
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
        .def("rdm1_sf", &CISigmaBuilder::compute_sf_1rdm_same_irrep, "C_left"_a, "C_right"_a)
        .def("rdm2_sf", &CISigmaBuilder::compute_sf_2rdm_same_irrep, "C_left"_a, "C_right"_a)
        .def("avg_build_time", &CISigmaBuilder::avg_build_time)
        .def("set_log_level", &CISigmaBuilder::set_log_level, "level"_a,
             "Set the logging level for the class");
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
