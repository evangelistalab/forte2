#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include "ci/slater_rules.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_slater_rules_api(nb::module_& m) {
    nb::class_<SlaterRules>(m, "SlaterRules")
        .def(nb::init<int, double, np_matrix, np_tensor4>(), "norb"_a, "scalar_energy"_a,
             "one_electron_integrals"_a, "two_electron_integrals"_a)
        .def("energy", &SlaterRules::energy)
        .def("slater_rules", &SlaterRules::slater_rules, "lhs"_a, "rhs"_a);
}

void export_rel_slater_rules_api(nb::module_& m) {
    nb::class_<RelSlaterRules>(m, "RelSlaterRules")
        .def(nb::init<int, double, np_matrix_complex, np_tensor4_complex>(), "nspinor"_a,
             "scalar_energy"_a, "one_electron_integrals"_a, "two_electron_integrals"_a)
        .def("energy", &RelSlaterRules::energy)
        .def("slater_rules", &RelSlaterRules::slater_rules, "lhs"_a, "rhs"_a);
}
} // namespace forte2