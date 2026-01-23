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

    sub_m.def(
        "compute_T1_block",
        [](np_matrix& t1, np_vector& ei, np_vector& ea, double flow_param) {
            compute_T1_block<double>(t1, ei, ea, flow_param);
        },
        "t1"_a, "ei"_a, "ea"_a, "flow_param"_a, "Computes the renormalized T1 amplitudes for a given block: T1_renorm = T1 * (1 - exp(-s*denom^2)) / denom");
    sub_m.def(
        "compute_T1_block",
        [](np_matrix_complex& t1, np_vector& ei, np_vector& ea, double flow_param) {
            compute_T1_block<std::complex<double>>(t1, ei, ea, flow_param);
        },
        "t1"_a, "ei"_a, "ea"_a, "flow_param"_a, "Computes the renormalized T1 amplitudes for a given block: T1_renorm = T1 * (1 - exp(-s*denom^2)) / denom");
    sub_m.def(
        "compute_T2_block",
        [](np_tensor4& t2, np_vector& ei, np_vector& ej, np_vector& ea, np_vector& eb,
           double flow_param) { compute_T2_block<double>(t2, ei, ej, ea, eb, flow_param); },
        "t2"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a, "Computes the renormalized T2 amplitudes for a given block: T2_renorm = T2 * (1 - exp(-s*denom^2)) / denom");
    sub_m.def(
        "compute_T2_block",
        [](np_tensor4_complex& t2, np_vector& ei, np_vector& ej, np_vector& ea, np_vector& eb,
           double flow_param) {
            compute_T2_block<std::complex<double>>(t2, ei, ej, ea, eb, flow_param);
        },
        "t2"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a, "Computes the renormalized T2 amplitudes for a given block: T2_renorm = T2 * (1 - exp(-s*denom^2)) / denom");
    sub_m.def(
        "renormalize_V_block",
        [](np_tensor4& v, np_vector& ei, np_vector& ej, np_vector& ea, np_vector& eb,
           double flow_param) { renormalize_V_block<double>(v, ei, ej, ea, eb, flow_param); },
        "v"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a, "Renormalizes a block of two-electron integrals: V_renorm = V * (1 + exp(-s*denom^2))");
    sub_m.def(
        "renormalize_V_block",
        [](np_tensor4_complex& v, np_vector& ei, np_vector& ej, np_vector& ea, np_vector& eb,
           double flow_param) {
            renormalize_V_block<std::complex<double>>(v, ei, ej, ea, eb, flow_param);
        },
        "v"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a, "Renormalizes a block of two-electron integrals: V_renorm = V * (1 + exp(-s*denom^2))");
    sub_m.def(
        "renormalize_3index",
        [](np_tensor3& v, double& ep, np_vector& eq, np_vector& er, np_vector& es,
           double flow_param) { renormalize_3index<double>(v, ep, eq, er, es, flow_param); },
        "v"_a, "ep"_a, "eq"_a, "er"_a, "es"_a, "flow_param"_a, "Renormalizes a block of three-index intermediates: V_renorm = V * (1 + exp(-s*denom^2)) * (1 - exp(-s*denom^2)) / denom");
    sub_m.def(
        "renormalize_3index",
        [](np_tensor3_complex& v, double& ep, np_vector& eq, np_vector& er, np_vector& es,
           double flow_param) {
            renormalize_3index<std::complex<double>>(v, ep, eq, er, es, flow_param);
        },
        "v"_a, "ep"_a, "eq"_a, "er"_a, "es"_a, "flow_param"_a, "Renormalizes a block of three-index intermediates: V_renorm = V * (1 + exp(-s*denom^2)) * (1 - exp(-s*denom^2)) / denom");
}
} // namespace forte2