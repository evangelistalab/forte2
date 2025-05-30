#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "ci/ci_string_lists.h"
#include "ci/ci_string_address.h"

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
        .def_prop_ro("ndet", &CIStrings::ndet);
    // .def_prop_ro("orbitals", &CIStrings::orbitals)
    // .def_prop_ro("orbital_symmetry", &CIStrings::orbital_symmetry)
    // .def_prop_ro("gas_min", &CIStrings::gas_min)
    // .def_prop_ro("gas_max", &CIStrings::gas_max);
}
} // namespace forte2