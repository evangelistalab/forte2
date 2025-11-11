#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/complex.h>

#include "dsrg/dsrg_utils.h"
#include "helpers/ndarray.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_dsrg_api(nb::module_& m) {
    nb::module_ sub_m = m.def_submodule("dsrg_utils", "DSRG utilities submodule");

    sub_m.def("regularized_denominator", &regularized_denominator, "x"_a, "s"_a);
    sub_m.def("taylor_exp", &taylor_exp, "z"_a);
    sub_m.def(
        "compute_T1_block",
        [](np_matrix& t1, np_vector& ei, np_vector& ea, double flow_param) {
            compute_T1_block<double>(t1, ei, ea, flow_param);
        },
        "t1"_a, "ei"_a, "ea"_a, "flow_param"_a);
    sub_m.def(
        "compute_T1_block",
        [](np_matrix_complex& t1, np_vector& ei, np_vector& ea, double flow_param) {
            compute_T1_block<std::complex<double>>(t1, ei, ea, flow_param);
        },
        "t1"_a, "ei"_a, "ea"_a, "flow_param"_a);
    sub_m.def(
        "compute_T2_block",
        [](np_tensor4& t2, np_vector& ei, np_vector& ej, np_vector& ea, np_vector& eb,
           double flow_param) { compute_T2_block<double>(t2, ei, ej, ea, eb, flow_param); },
        "t2"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a);
    sub_m.def(
        "compute_T2_block",
        [](np_tensor4_complex& t2, np_vector& ei, np_vector& ej, np_vector& ea, np_vector& eb,
           double flow_param) {
            compute_T2_block<std::complex<double>>(t2, ei, ej, ea, eb, flow_param);
        },
        "t2"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a);
    sub_m.def(
        "renormalize_V_block",
        [](np_tensor4& v, np_vector& ei, np_vector& ej, np_vector& ea, np_vector& eb,
           double flow_param) { renormalize_V_block<double>(v, ei, ej, ea, eb, flow_param); },
        "v"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a);
    sub_m.def(
        "renormalize_V_block",
        [](np_tensor4_complex& v, np_vector& ei, np_vector& ej, np_vector& ea, np_vector& eb,
           double flow_param) {
            renormalize_V_block<std::complex<double>>(v, ei, ej, ea, eb, flow_param);
        },
        "v"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a);
    sub_m.def("renormalize_V_block_sf", renormalize_V_block_sf, "v"_a, "ea"_a, "eb"_a, "ei"_a,
              "ej"_a, "flow_param"_a);
    sub_m.def(
        "renormalize_3index",
        [](np_tensor3& v, double& ep, np_vector& eq, np_vector& er, np_vector& es,
           double flow_param) { renormalize_3index<double>(v, ep, eq, er, es, flow_param); },
        "v"_a, "ep"_a, "eq"_a, "er"_a, "es"_a, "flow_param"_a);
    sub_m.def(
        "renormalize_3index",
        [](np_tensor3_complex& v, double& ep, np_vector& eq, np_vector& er, np_vector& es,
           double flow_param) {
            renormalize_3index<std::complex<double>>(v, ep, eq, er, es, flow_param);
        },
        "v"_a, "ep"_a, "eq"_a, "er"_a, "es"_a, "flow_param"_a);
    sub_m.def("renormalize_F", &renormalize_F, "F"_a, "ei"_a, "ea"_a, "flow_param"_a);
    sub_m.def("renormalize_CCVV", &renormalize_CCVV, "Jvvc"_a, "ec_"_a, "ev"_a, "ec"_a,
              "flow_param"_a);
    sub_m.def("renormalize_CCAV", &renormalize_CCAV, "JKva"_a, "Jva"_a, "e_"_a, "ev"_a, "ea"_a,
              "flow_param"_a);
    sub_m.def("renormalize_CAVV", &renormalize_CAVV, "JKvva"_a, "Jvva"_a, "e_"_a, "ev"_a, "ea"_a,
              "flow_param"_a);
}
} // namespace forte2