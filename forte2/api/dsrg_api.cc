#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>

#include "dsrg/dsrg_utils.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_dsrg_api(nb::module_& m) {
    nb::module_ sub_m = m.def_submodule("dsrg_utils", "DSRG utilities submodule");

    sub_m.def("regularized_denominator", &regularized_denominator, "x"_a, "s"_a);
    sub_m.def("taylor_exp", &taylor_exp, "z"_a);
}
} // namespace forte2