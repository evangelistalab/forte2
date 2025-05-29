#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/make_iterator.h>

#include "ci/sparse_exp.h"
#include "ci/sparse_operator.h"
#include "ci/sparse_state.h"

#include "helpers/string_algorithms.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_sparse_exp_api(nb::module_& m) {
    nb::class_<SparseExp>(m, "SparseExp", "A class to compute the exponential of a sparse operator")
        // Constructors
        .def(nb::init<int, double>(), "maxk"_a = 19, "screen_thresh"_a = 1.0e-12)
        .def("set_screen_thresh", &SparseExp::set_screen_thresh, "screen_thresh"_a)
        .def("set_maxk", &SparseExp::set_maxk, "maxk"_a)
        .def("apply_op",
             nb::overload_cast<const SparseOperator&, const SparseState&, double>(
                 &SparseExp::apply_op),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0)
        .def("apply_op",
             nb::overload_cast<const SparseOperatorList&, const SparseState&, double>(
                 &SparseExp::apply_op),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0)
        .def("apply_antiherm",
             nb::overload_cast<const SparseOperator&, const SparseState&, double>(
                 &SparseExp::apply_antiherm),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0)
        .def("apply_antiherm",
             nb::overload_cast<const SparseOperatorList&, const SparseState&, double>(
                 &SparseExp::apply_antiherm),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0);
}

void export_sparse_fact_exp_api(nb::module_& m) {
    nb::class_<SparseFactExp>(
        m, "SparseFactExp",
        "A class to compute the product exponential of a sparse operator using factorization")
        .def(nb::init<double>(), "screen_thresh"_a = 1.0e-12)
        .def("apply_op", &SparseFactExp::apply_op, "sop"_a, "state"_a, "inverse"_a = false)
        .def("apply_antiherm", &SparseFactExp::apply_antiherm, "sop"_a, "state"_a,
             "inverse"_a = false);
}

} // namespace forte2