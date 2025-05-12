#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/unordered_map.h>

#include "ci/state.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_sparsestate_api(nb::module_& m) {
    nb::class_<State>(m, "SparseState")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<const State&>(), "Copy constructor")
        .def(nb::init<const State::container&>(), "Create a State from a container of Determinants")
        .def(nb::init<const Determinant&, sparse_scalar_t>(), "det"_a, "val"_a = 1,
             "Create a State with a single determinant")
        .def("__len__", &State::size)
        .def("__getitem__", [](State& v, const Determinant& d) { return v[d]; })
        .def("__setitem__",
             [](State& v, const Determinant& d, const sparse_scalar_t val) { v[d] = val; })
        .def("__contains__", [](State& v, const Determinant& d) { return v.count(d); })
        //    .def(
        //        "items", [](const State& v) { return nb::make_iterator(v.begin(),
        //   v.end());
        //        }, nb::keep_alive<0, 1>()) // Essential: keep object alive while iterator exists
        .def("norm", &State::norm, "p"_a = 2,
             "Calculate the p-norm of the State (default p = 2, p = -1 for infinity norm)")
        .def("add", &State::add)
        .def("items", [](const State& v) { return v.elements(); })
        .def("__add__", &State::operator+, "Add two States")
        .def(
            "__sub__", [](const State& a, const State& b) { return a - b; }, "Subtract two States")
        .def("__iadd__", &State::operator+=, "Add a State to this State")
        .def("__isub__", &State::operator-=, "Subtract a State from this State")
        .def("__imul__", &State::operator*=, "Multiply this State by a scalar")
        .def("__eq__", &State::operator==);
    //    .def("__repr__", [](const State& v) { return v.str(); })
    //    .def("__str__", [](const State& v) { return v.str(); })
}
} // namespace forte2