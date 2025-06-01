#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>

#include "ci/ci_string_lists.h"
#include "ci/ci_string_address.h"
#include "ci/ci_vector.h"
#include "ci/ci_sigma_builder.h"

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
        .def("determinant", &CIStrings::determinant, "address"_a)
        .def("determinant_index", &CIStrings::determinant_address, "d"_a);
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
        .def(nb::init<const CIStrings&, double, np_matrix&, np_tensor4&>(), "lists"_a, "E"_a, "H"_a,
             "V"_a)
        .def_static("allocate_temp_space", &CISigmaBuilder::allocate_temp_space, "lists"_a,
                    nb::rv_policy::reference_internal)
        .def_static("release_temp_space", &CISigmaBuilder::release_temp_space)
        .def("form_Hdiag_det", &CISigmaBuilder::form_Hdiag_det)
        .def(
            "Hamiltonian",
            [](const CISigmaBuilder& self, CIVector& basis, CIVector& sigma) {
                self.Hamiltonian(basis, sigma);
            },
            "basis"_a, "sigma"_a)
        .def("Hamiltonian2", &CISigmaBuilder::Hamiltonian2, "basis"_a, "sigma"_a)
        .def("avg_build_time", &CISigmaBuilder::avg_build_time);
}

} // namespace forte2