#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/make_iterator.h>

#include "sparse/sq_operator_string.h"
#include "sparse/sparse_operator.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {
void export_sq_operator_string_api(nb::module_& m) {
    nb::class_<SQOperatorString>(m, "SQOperatorString",
                                 "A class to represent a string of creation/annihilation operators")
        .def(nb::init<const Determinant&, const Determinant&>())
        .def("cre", &SQOperatorString::cre, "Get the creation operator string")
        .def("ann", &SQOperatorString::ann, "Get the annihilation operator string")
        .def("str", &SQOperatorString::str, "Get the string representation of the operator string")
        .def("count", &SQOperatorString::count, "Get the number of operators")
        .def("adjoint", &SQOperatorString::adjoint, "Get the adjoint operator string")
        .def("spin_flip", &SQOperatorString::spin_flip, "Get the spin-flipped operator string")
        .def("number_component", &SQOperatorString::number_component,
             "Get the number component of the operator string")
        .def("non_number_component", &SQOperatorString::non_number_component,
             "Get the non-number component of the operator string")
        .def("__str__", &SQOperatorString::str,
             "Get the string representation of the operator string")
        .def("__repr__", &SQOperatorString::str,
             "Get the string representation of the operator string")
        .def("latex", &SQOperatorString::latex,
             "Get the LaTeX representation of the operator string")
        .def("latex_compact", &SQOperatorString::latex_compact,
             "Get the compact LaTeX representation of the operator string")
        .def("is_identity", &SQOperatorString::is_identity,
             "Check if the operator string is the identity operator")
        .def("is_nilpotent", &SQOperatorString::is_nilpotent,
             "Check if the operator string is nilpotent")
        .def("op_tuple", &SQOperatorString::op_tuple, "Get the operator tuple")
        .def("__eq__", &SQOperatorString::operator==, "Check if two operator strings are equal")
        .def("__lt__", &SQOperatorString::operator<,
             "Check if an operator string is less than another")
        .def(
            "__mul__",
            [](const SQOperatorString& sqop, const sparse_scalar_t& scalar) {
                SparseOperator sop;
                sop.add(sqop, scalar);
                return sop;
            },
            nb::is_operator(), "Multiply an operator string by a scalar")
        .def(
            "__rmul__",
            [](const SQOperatorString& sqop, const sparse_scalar_t& scalar) {
                SparseOperator sop;
                sop.add(sqop, scalar);
                return sop;
            },
            nb::is_operator(), "Multiply an operator string by a scalar");

    m.def(
        "sqop",
        [](const std::string& s, bool allow_reordering) {
            return make_sq_operator_string(s, allow_reordering);
        },
        "s"_a, "allow_reordering"_a = false,
        "Create an operator string from a string representation (default: no not allow "
        "reordering)");

    nb::enum_<CommutatorType>(m, "CommutatorType")
        .value("commute", CommutatorType::Commute)
        .value("anticommute", CommutatorType::AntiCommute)
        .value("may_not_commute", CommutatorType::MayNotCommute);

    m.def("commutator_type", &commutator_type, "lhs"_a, "rhs"_a,
          "Get the commutator type of two operator strings");
}
} // namespace forte2
