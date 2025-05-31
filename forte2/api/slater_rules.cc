#include <nanobind/nanobind.h>

#include "ci/slater_rules.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_determinant_api(nb::module_& m) {
    nb::class_<SlaterRules>(m, "Determinant")
        .def(nb::init<int, double, np_matrix, np_tensor4>(), "norb"_a, "scalar_energy"_a,
             "one_electron_integrals"_a, "two_electron_integrals"_a)
        .def("energy", &SlaterRules::energy);
}
} // namespace forte2