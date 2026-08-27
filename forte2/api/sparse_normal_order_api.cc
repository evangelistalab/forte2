#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "helpers/string_algorithms.h"

#include "sparse/sparse_generalized_normal_order_product.h"
#include "sparse/sparse_normal_order.h"
#include "sparse/sparse_normal_order_product.h"
#include "sparse/sparse_state.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_sparse_normal_order_api(nb::module_& m) {
    nb::class_<NormalOrderedString>(m, "NormalOrderedString",
                                    "A normal-ordered string represented by compact creator and "
                                    "annihilator determinants")
        .def(nb::init<const Determinant&, const Determinant&>(), "cre"_a, "ann"_a)
        .def("cre", &NormalOrderedString::cre, "Get the normal creator string")
        .def("ann", &NormalOrderedString::ann, "Get the normal annihilator string")
        .def("sign_mask", &NormalOrderedString::sign_mask, "Get the normal-order sign mask")
        .def("op_tuple", &NormalOrderedString::op_tuple, "reference"_a,
             "Get the physical operator tuple for a determinant reference")
        .def("count", &NormalOrderedString::count, "Get the number of operators")
        .def("many_body_rank", &NormalOrderedString::many_body_rank,
             "Get the many-body rank, defined as ceil(count / 2)")
        .def("is_identity", &NormalOrderedString::is_identity,
             "Check if the normal-ordered string is the identity")
        .def("str", nb::overload_cast<>(&NormalOrderedString::str, nb::const_),
             "Get the string representation assuming the particle vacuum")
        .def("str", nb::overload_cast<const Determinant&>(&NormalOrderedString::str, nb::const_),
             "reference"_a, "Get the physical string representation for a determinant reference")
        .def("latex", nb::overload_cast<>(&NormalOrderedString::latex, nb::const_),
             "Get the LaTeX representation assuming the particle vacuum")
        .def("latex",
             nb::overload_cast<const Determinant&>(&NormalOrderedString::latex, nb::const_),
             "reference"_a, "Get the LaTeX representation for a determinant reference")
        .def("__str__", nb::overload_cast<>(&NormalOrderedString::str, nb::const_),
             "Get the string representation assuming the particle vacuum")
        .def("__repr__", nb::overload_cast<>(&NormalOrderedString::str, nb::const_),
             "Get the string representation assuming the particle vacuum")
        .def("__eq__", &NormalOrderedString::operator==,
             "Check if two normal-ordered strings are equal")
        .def("__lt__", &NormalOrderedString::operator<,
             "Check if a normal-ordered string is less than another")
        .def("__hash__",
             [](const NormalOrderedString& str) { return NormalOrderedString::Hash{}(str); });

    nb::class_<NormalOrderedSparseOperator>(m, "NormalOrderedSparseOperator",
                                            "A sparse operator in determinant-normal-ordered form")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<const Determinant&>(), "reference"_a,
             "Create an empty normal-ordered operator with a determinant reference")
        .def(nb::init<const Determinant&, const NormalOrderedString&, sparse_scalar_t>(),
             "reference"_a, "term"_a, "coefficient"_a = sparse_scalar_t(1),
             "Create a normal-ordered operator with a single term")
        .def(
            "add",
            [](NormalOrderedSparseOperator& op, const NormalOrderedString& term,
               sparse_scalar_t coefficient) { op.add(term, coefficient); },
            "term"_a, "coefficient"_a = sparse_scalar_t(1),
            "Add a normal-ordered term to the operator")
        .def("coefficient", &NormalOrderedSparseOperator::coefficient, "term"_a,
             "Get the coefficient of a normal-ordered term")
        .def("reference", &NormalOrderedSparseOperator::reference, "Get the reference determinant")
        .def("__len__", &NormalOrderedSparseOperator::size,
             "Get the number of normal-ordered terms")
        .def(
            "__iter__",
            [](const NormalOrderedSparseOperator& op) {
                return nb::make_iterator(nb::type<NormalOrderedSparseOperator>(), "item_iterator",
                                         op.elements().begin(), op.elements().end());
            },
            nb::keep_alive<0, 1>())
        .def("str", &NormalOrderedSparseOperator::str,
             "Get a string representation of the normal-ordered operator")
        .def("latex", &NormalOrderedSparseOperator::latex,
             "Get a LaTeX representation of the normal-ordered operator")
        .def("truncate", &NormalOrderedSparseOperator::truncate, "max_rank"_a,
             "screen_thresh"_a = 1.0e-12, "Return a copy with terms above max_rank removed")
        .def("__repr__", [](const NormalOrderedSparseOperator& op) { return join(op.str(), "\n"); })
        .def("__str__", [](const NormalOrderedSparseOperator& op) { return join(op.str(), "\n"); })
        .def(
            "__add__",
            [](const NormalOrderedSparseOperator& lhs, const NormalOrderedSparseOperator& rhs) {
                return lhs + rhs;
            },
            "Add two normal-ordered operators")
        .def(
            "__iadd__",
            [](NormalOrderedSparseOperator& lhs,
               const NormalOrderedSparseOperator& rhs) -> NormalOrderedSparseOperator& {
                lhs += rhs;
                return lhs;
            },
            "Add a normal-ordered operator in place")
        .def(
            "__sub__",
            [](const NormalOrderedSparseOperator& lhs, const NormalOrderedSparseOperator& rhs) {
                return lhs - rhs;
            },
            "Subtract two normal-ordered operators")
        .def(
            "__isub__",
            [](NormalOrderedSparseOperator& lhs,
               const NormalOrderedSparseOperator& rhs) -> NormalOrderedSparseOperator& {
                lhs -= rhs;
                return lhs;
            },
            "Subtract a normal-ordered operator in place")
        .def(
            "__mul__",
            [](const NormalOrderedSparseOperator& op, sparse_scalar_t scalar) {
                return op * scalar;
            },
            "Multiply a normal-ordered operator by a scalar")
        .def(
            "__rmul__",
            [](const NormalOrderedSparseOperator& op, sparse_scalar_t scalar) {
                return op * scalar;
            },
            "Multiply a scalar by a normal-ordered operator")
        .def(
            "__neg__", [](const NormalOrderedSparseOperator& op) { return -op; },
            "Negate the normal-ordered operator")
        .def(
            "norm", [](const NormalOrderedSparseOperator& op) { return op.norm(); },
            "Compute the norm of the normal-ordered operator")
        .def(
            "adjoint",
            [](const NormalOrderedSparseOperator& op, double screen_thresh) {
                return adjoint(op, screen_thresh);
            },
            "screen_thresh"_a = 1.0e-12, "Return the adjoint")
        .def(
            "commutator",
            [](const NormalOrderedSparseOperator& lhs, const NormalOrderedSparseOperator& rhs,
               int max_rank, double screen_thresh) {
                return normal_ordered_commutator(lhs, rhs, max_rank, screen_thresh);
            },
            "rhs"_a, "max_rank"_a, "screen_thresh"_a = 1.0e-12,
            "Compute a normal-ordered commutator truncated to max_rank")
        .def("to_sparse_operator", &NormalOrderedSparseOperator::to_sparse_operator,
             "screen_thresh"_a = 1.0e-12,
             "Convert this normal-ordered operator back to a SparseOperator")
        .def("apply_to_state", &NormalOrderedSparseOperator::apply_to_state, "state"_a,
             "screen_thresh"_a = 1.0e-12, "Apply this normal-ordered operator to a SparseState")
        .def(
            "__matmul__",
            [](const NormalOrderedSparseOperator& op, const SparseState& state) {
                return op.apply_to_state(state);
            },
            "Apply this normal-ordered operator to a SparseState")
        .def("__eq__", &NormalOrderedSparseOperator::operator==,
             "Check if two normal-ordered operators are equal");

    m.def("normal_order", &normal_order, "op"_a, "vacuum"_a, "screen_thresh"_a = 1.0e-12,
          "max_rank"_a = -1, "Normal order a SparseOperator with respect to a determinant vacuum");

    nb::class_<GeneralizedNormalOrderedSparseOperator>(
        m, "GeneralizedNormalOrderedSparseOperator",
        "A sparse operator in generalized normal-ordered form for a sparse CI vacuum")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<const SparseState&, std::size_t, int>(), "vacuum"_a, "norb"_a,
             "max_cumulant"_a = -1, "Create an empty generalized normal-ordered operator")
        .def(nb::init<const SparseState&, std::size_t, int, const SQOperatorString&,
                      sparse_scalar_t>(),
             "vacuum"_a, "norb"_a, "max_cumulant"_a, "term"_a, "coefficient"_a = sparse_scalar_t(1),
             "Create a generalized normal-ordered operator with a single term")
        .def(
            "add",
            [](GeneralizedNormalOrderedSparseOperator& op, const SQOperatorString& term,
               sparse_scalar_t coefficient) { op.add(term, coefficient); },
            "term"_a, "coefficient"_a = sparse_scalar_t(1),
            "Add a generalized normal-ordered term to the operator")
        .def("coefficient", &GeneralizedNormalOrderedSparseOperator::coefficient, "term"_a,
             "Get the coefficient of a generalized normal-ordered term")
        .def("vacuum", &GeneralizedNormalOrderedSparseOperator::vacuum, "Get the sparse CI vacuum")
        .def("norb", &GeneralizedNormalOrderedSparseOperator::norb,
             "Get the number of spatial orbitals")
        .def("max_cumulant", &GeneralizedNormalOrderedSparseOperator::max_cumulant,
             "Get the maximum contracted body rank")
        .def("__len__", &GeneralizedNormalOrderedSparseOperator::size,
             "Get the number of generalized normal-ordered terms")
        .def(
            "__iter__",
            [](const GeneralizedNormalOrderedSparseOperator& op) {
                return nb::make_iterator(nb::type<GeneralizedNormalOrderedSparseOperator>(),
                                         "item_iterator", op.elements().begin(),
                                         op.elements().end());
            },
            nb::keep_alive<0, 1>())
        .def("str", &GeneralizedNormalOrderedSparseOperator::str,
             "Get a string representation of the generalized normal-ordered operator")
        .def("latex", &GeneralizedNormalOrderedSparseOperator::latex,
             "Get a LaTeX representation of the generalized normal-ordered operator")
        .def("truncate", &GeneralizedNormalOrderedSparseOperator::truncate, "max_rank"_a,
             "screen_thresh"_a = 1.0e-12, "Return a copy with terms above max_rank removed")
        .def("__repr__",
             [](const GeneralizedNormalOrderedSparseOperator& op) { return join(op.str(), "\n"); })
        .def("__str__",
             [](const GeneralizedNormalOrderedSparseOperator& op) { return join(op.str(), "\n"); })
        .def(
            "__add__",
            [](const GeneralizedNormalOrderedSparseOperator& lhs,
               const GeneralizedNormalOrderedSparseOperator& rhs) { return lhs + rhs; },
            "Add two generalized normal-ordered operators")
        .def(
            "__iadd__",
            [](GeneralizedNormalOrderedSparseOperator& lhs,
               const GeneralizedNormalOrderedSparseOperator& rhs)
                -> GeneralizedNormalOrderedSparseOperator& {
                lhs += rhs;
                return lhs;
            },
            "Add a generalized normal-ordered operator in place")
        .def(
            "__sub__",
            [](const GeneralizedNormalOrderedSparseOperator& lhs,
               const GeneralizedNormalOrderedSparseOperator& rhs) { return lhs - rhs; },
            "Subtract two generalized normal-ordered operators")
        .def(
            "__isub__",
            [](GeneralizedNormalOrderedSparseOperator& lhs,
               const GeneralizedNormalOrderedSparseOperator& rhs)
                -> GeneralizedNormalOrderedSparseOperator& {
                lhs -= rhs;
                return lhs;
            },
            "Subtract a generalized normal-ordered operator in place")
        .def(
            "__mul__",
            [](const GeneralizedNormalOrderedSparseOperator& op, sparse_scalar_t scalar) {
                return op * scalar;
            },
            "Multiply a generalized normal-ordered operator by a scalar")
        .def(
            "__rmul__",
            [](const GeneralizedNormalOrderedSparseOperator& op, sparse_scalar_t scalar) {
                return op * scalar;
            },
            "Multiply a scalar by a generalized normal-ordered operator")
        .def(
            "__neg__", [](const GeneralizedNormalOrderedSparseOperator& op) { return -op; },
            "Negate the generalized normal-ordered operator")
        .def(
            "norm", [](const GeneralizedNormalOrderedSparseOperator& op) { return op.norm(); },
            "Compute the norm of the generalized normal-ordered operator")
        .def(
            "commutator",
            [](const GeneralizedNormalOrderedSparseOperator& lhs,
               const GeneralizedNormalOrderedSparseOperator& rhs, int max_rank,
               double screen_thresh) {
                return generalized_normal_ordered_commutator(lhs, rhs, max_rank, screen_thresh);
            },
            "rhs"_a, "max_rank"_a, "screen_thresh"_a = 1.0e-12,
            "Compute a generalized normal-ordered commutator truncated to max_rank")
        .def("to_sparse_operator", &GeneralizedNormalOrderedSparseOperator::to_sparse_operator,
             "screen_thresh"_a = 1.0e-12,
             "Convert this generalized normal-ordered operator back to a SparseOperator")
        .def("apply_to_state", &GeneralizedNormalOrderedSparseOperator::apply_to_state, "state"_a,
             "screen_thresh"_a = 1.0e-12,
             "Apply this generalized normal-ordered operator to a SparseState")
        .def(
            "__matmul__",
            [](const GeneralizedNormalOrderedSparseOperator& op, const SparseState& state) {
                return op.apply_to_state(state);
            },
            "Apply this generalized normal-ordered operator to a SparseState")
        .def("__eq__", &GeneralizedNormalOrderedSparseOperator::operator==,
             "Check if two generalized normal-ordered operators are equal");

    m.def("generalized_normal_order", &generalized_normal_order, "op"_a, "vacuum"_a, "norb"_a,
          "max_cumulant"_a = -1, "screen_thresh"_a = 1.0e-12, "max_rank"_a = -1,
          "Generalized normal order a SparseOperator with respect to a sparse CI vacuum");
}

} // namespace forte2
