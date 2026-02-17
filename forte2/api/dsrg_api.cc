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
        [](ndarray<double, 2>& t1, ndarray<double, 1>& ei, ndarray<double, 1>& ea, double flow_param) {
            compute_T1_block<double>(t1, ei, ea, flow_param);
        },
        "t1"_a, "ei"_a, "ea"_a, "flow_param"_a, "Computes the renormalized T1 amplitudes for a given block: T1_renorm = T1 * (1 - exp(-s*denom^2)) / denom");
    sub_m.def(
        "compute_T1_block",
        [](ndarray<std::complex<double>, 2>& t1, ndarray<double, 1>& ei, ndarray<double, 1>& ea, double flow_param) {
            compute_T1_block<std::complex<double>>(t1, ei, ea, flow_param);
        },
        "t1"_a, "ei"_a, "ea"_a, "flow_param"_a, "Computes the renormalized T1 amplitudes for a given block: T1_renorm = T1 * (1 - exp(-s*denom^2)) / denom");
    sub_m.def(
        "compute_T2_block",
        [](ndarray<double, 4>& t2, ndarray<double, 1>& ei, ndarray<double, 1>& ej, ndarray<double, 1>& ea, ndarray<double, 1>& eb,
           double flow_param) { compute_T2_block<double>(t2, ei, ej, ea, eb, flow_param); },
        "t2"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a, "Computes the renormalized T2 amplitudes for a given block: T2_renorm = T2 * (1 - exp(-s*denom^2)) / denom");
    sub_m.def(
        "compute_T2_block",
        [](ndarray<std::complex<double>, 4>& t2, ndarray<double, 1>& ei, ndarray<double, 1>& ej, ndarray<double, 1>& ea, ndarray<double, 1>& eb,
           double flow_param) {
            compute_T2_block<std::complex<double>>(t2, ei, ej, ea, eb, flow_param);
        },
        "t2"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a, "Computes the renormalized T2 amplitudes for a given block: T2_renorm = T2 * (1 - exp(-s*denom^2)) / denom");
    sub_m.def(
        "renormalize_V_block",
        [](ndarray<double, 4>& v, ndarray<double, 1>& ei, ndarray<double, 1>& ej, ndarray<double, 1>& ea, ndarray<double, 1>& eb,
           double flow_param) { renormalize_V_block<double>(v, ei, ej, ea, eb, flow_param); },
        "v"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a, "Renormalizes a block of two-electron integrals: V_renorm = V * (1 + exp(-s*denom^2))");
    sub_m.def(
        "renormalize_V_block",
        [](ndarray<std::complex<double>, 4>& v, ndarray<double, 1>& ei, ndarray<double, 1>& ej, ndarray<double, 1>& ea, ndarray<double, 1>& eb,
           double flow_param) {
            renormalize_V_block<std::complex<double>>(v, ei, ej, ea, eb, flow_param);
        },
        "v"_a, "ei"_a, "ej"_a, "ea"_a, "eb"_a, "flow_param"_a, "Renormalizes a block of two-electron integrals: V_renorm = V * (1 + exp(-s*denom^2))");
    sub_m.def(
        "renormalize_3index",
        [](ndarray<double, 3>& v, double& ep, ndarray<double, 1>& eq, ndarray<double, 1>& er, ndarray<double, 1>& es,
           double flow_param) { renormalize_3index<double>(v, ep, eq, er, es, flow_param); },
        "v"_a, "ep"_a, "eq"_a, "er"_a, "es"_a, "flow_param"_a, "Renormalizes a block of three-index intermediates: V_renorm = V * (1 + exp(-s*denom^2)) * (1 - exp(-s*denom^2)) / denom");
    sub_m.def(
        "renormalize_3index",
        [](ndarray<std::complex<double>, 3>& v, double& ep, ndarray<double, 1>& eq, ndarray<double, 1>& er, ndarray<double, 1>& es,
           double flow_param) {
            renormalize_3index<std::complex<double>>(v, ep, eq, er, es, flow_param);
        },
        "v"_a, "ep"_a, "eq"_a, "er"_a, "es"_a, "flow_param"_a, "Renormalizes a block of three-index intermediates: V_renorm = V * (1 + exp(-s*denom^2)) * (1 - exp(-s*denom^2)) / denom");
}
} // namespace forte2