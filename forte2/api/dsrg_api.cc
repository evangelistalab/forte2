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
    sub_m.def("compute_T1_block", &compute_T1_block, "t1"_a, "ei"_a, "ea"_a, "flow_param"_a);
    sub_m.def("compute_T2_block", &compute_T2_block, "t2"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a,
              "flow_param"_a);
    sub_m.def("renormalize_V_block", &renormalize_V_block, "v"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a,
              "flow_param"_a);
    sub_m.def("renormalize_CCVV", &renormalize_CCVV, "v"_a, "ec"_a, "ev"_a, "ei"_a, "flow_param"_a);
}
} // namespace forte2