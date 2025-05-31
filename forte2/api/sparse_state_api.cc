#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h> 
#include <nanobind/make_iterator.h>

#include "ci/sparse_state.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_sparse_state_api(nb::module_& m) {
    nb::class_<SparseState>(m, "SparseState", "A class to represent a vector of determinants")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<const SparseState&>(), "Copy constructor")
        .def(nb::init<const SparseState::container&>(),
             "Create a SparseState from a container of Determinants")
        .def(nb::init<const Determinant&, sparse_scalar_t>(), "det"_a, "val"_a = 1,
             "Create a SparseState with a single determinant")
        .def(
            "items",
            [](const SparseState& v) {
                return nb::make_iterator(nb::type<SparseState>(), "item_iterator", v.begin(),
                                         v.end());
            },
            nb::keep_alive<0, 1>()) // Essential: keep object alive while iterator exists
        .def("str", &SparseState::str)
        .def("size", &SparseState::size)
        .def("norm", &SparseState::norm, "p"_a = 2,
             "Calculate the p-norm of the SparseState (default p = 2, p = -1 for infinity norm)")
        .def("add", &SparseState::add)
        .def("__add__", &SparseState::operator+, "Add two SparseStates")
        .def(
            "__sub__", [](const SparseState& a, const SparseState& b) { return a - b; },
            "Subtract two SparseStates")
        .def("__iadd__", &SparseState::operator+=, "Add a SparseState to this SparseState")
        .def("__isub__", &SparseState::operator-=, "Subtract a SparseState from this SparseState")
        .def("__imul__", &SparseState::operator*=, "Multiply this SparseState by a scalar")
        .def("__len__", &SparseState::size)
        .def("__eq__", &SparseState::operator==)
        .def("__repr__", [](const SparseState& v) { return v.str(); })
        .def("__str__", [](const SparseState& v) { return v.str(); })
        .def("map", [](const SparseState& v) { return v.elements(); })
        .def("elements", [](const SparseState& v) { return v.elements(); })
        .def("__getitem__", [](SparseState& v, const Determinant& d) { return v[d]; })
        .def("__setitem__",
             [](SparseState& v, const Determinant& d, const sparse_scalar_t val) { v[d] = val; })
        .def("__contains__", [](SparseState& v, const Determinant& d) { return v.count(d); })
        .def(
            "apply",
            [](const SparseState& v, const SparseOperator& op) {
                return apply_operator_lin(op, v);
            },
            "Apply an operator to this SparseState and return a new SparseState")
        .def(
            "apply_antiherm",
            [](const SparseState& v, const SparseOperator& op) {
                return apply_operator_antiherm(op, v);
            },
            "Apply the antihermitian combination of the operator (op - op^dagger) to this "
            "SparseState and return a new SparseState")
        .def("number_project",
             [](const SparseState& v, int na, int nb) { return apply_number_projector(na, nb, v); })
        .def(
            "spin2", [](const SparseState& v) { return spin2(v, v); },
            "Calculate the expectation value of S^2 for this SparseState")
        .def(
            "overlap",
            [](const SparseState& v, const SparseState& other) { return overlap(v, other); },
            "Calculate the overlap between this SparseState and another SparseState");

    m.def("apply_op", &apply_operator_lin, "sop"_a, "state0"_a, "screen_thresh"_a = 1.0e-12);

    m.def("apply_antiherm", &apply_operator_antiherm, "sop"_a, "state0"_a,
          "screen_thresh"_a = 1.0e-12);

    m.def("apply_number_projector", &apply_number_projector);

    // m.def("get_projection", &get_projection);

    // there's already a function called spin2, overload the spin2 function
    m.def(
        "spin2",
        [](const SparseState& left_state, const SparseState& right_state) {
            return spin2(left_state, right_state);
        },
        "Calculate the <left_state|S^2|right_state> expectation value");

    m.def("overlap", &overlap);

    m.def("normalize", &normalize, "Returns a normalized version of the input SparseState");
}
} // namespace forte2