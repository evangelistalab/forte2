#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/make_iterator.h>

#include "ci/sparse_operator.h"
#include "ci/sparse_state.h"

#include "helpers/string_algorithms.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_sparse_operator_api(nb::module_& m) {
    nb::class_<SparseOperator>(m, "SparseOperator", "A class to represent a sparse operator")
        // Constructors
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<SparseOperator>(), "Copy constructor")
        .def(nb::init<const SparseOperator::container&>(),
             "Create a SparseOperator from a container of terms")
        .def(nb::init<const SQOperatorString&, sparse_scalar_t>(), "sqop"_a,
             "coefficient"_a = sparse_scalar_t(1), "Create a SparseOperator with a single term")

        // Add/Remove terms
        .def("add",
             nb::overload_cast<const SQOperatorString&, sparse_scalar_t>(&SparseOperator::add),
             "sqop"_a, "coefficient"_a = sparse_scalar_t(1), "Add a term to the operator")
        .def("add",
             nb::overload_cast<const std::string&, sparse_scalar_t, bool>(
                 &SparseOperator::add_term_from_str),
             "str"_a, "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false,
             "Add a term to the operator from a string representation")
        .def(
            "add",
            [](SparseOperator& op, const std::vector<size_t>& acre, const std::vector<size_t>& bcre,
               const std::vector<size_t>& aann, const std::vector<size_t>& bann,
               sparse_scalar_t coeff) {
                op.add(SQOperatorString({acre.begin(), acre.end()}, {bcre.begin(), bcre.end()},
                                        {aann.begin(), aann.end()}, {bann.begin(), bann.end()}),
                       coeff);
            },
            "acre"_a, "bcre"_a, "aann"_a, "bann"_a, "coeff"_a = sparse_scalar_t(1),
            "Add a term to the operator by passing lists of creation and annihilation indices. "
            "This version is faster than the string version and does not check for reordering")
        .def(
            "remove",
            [](SparseOperator& op, const std::string& s) {
                const auto [sqop, _] = make_sq_operator_string(s, false);
                op.remove(sqop);
            },
            "Remove a term")

        // Accessors
        .def(
            "__iter__",
            [](const SparseOperator& v) {
                return nb::make_iterator(nb::type<SparseOperator>(), "item_iterator",
                                         v.elements().begin(), v.elements().end());
            },
            nb::keep_alive<0, 1>()) // Essential: keep object alive while iterator exists
        .def(
            "__getitem__",
            [](const SparseOperator& op, const std::string& s) {
                const auto [sqop, factor] = make_sq_operator_string(s, false);
                return factor * op[sqop];
            },
            "Get the coefficient of a term")
        .def("__len__", &SparseOperator::size, "Get the number of terms in the operator")
        .def(
            "coefficient",
            [](const SparseOperator& op, const std::string& s) {
                const auto [sqop, factor] = make_sq_operator_string(s, false);
                return factor * op[sqop];
            },
            "Get the coefficient of a term")
        .def(
            "set_coefficient",
            [](SparseOperator& op, const std::string& s, sparse_scalar_t value) {
                const auto [sqop, factor] = make_sq_operator_string(s, false);
                op[sqop] = factor * value;
            },
            "Set the coefficient of a term")

        // Arithmetic Operations
        .def("__add__", &SparseOperator::operator+, "Add two SparseOperators")
        .def(
            "__sub__", [](const SparseOperator& a, const SparseOperator& b) { return a - b; },
            "Subtract two SparseOperators")
        .def("__iadd__", &SparseOperator::operator+=, "Add a SparseOperator to this SparseOperator")
        .def("__isub__", &SparseOperator::operator-=,
             "Subtract a SparseOperator from this SparseOperator")
        .def(
            "__imul__",
            [](const SparseOperator self, sparse_scalar_t scalar) {
                return self * scalar; // Call the multiplication operator
            },
            "Multiply this SparseOperator by a scalar")
        // .def(
        //     "__imul__",
        //     [](SparseOperator self, const SparseOperator& other) {
        //         SparseOperator C;
        //         for (const auto& [op, coeff] : self.elements()) {
        //             for (const auto& [op2, coeff2] : other.elements()) {
        //                 new_product2(C, op, op2, coeff * coeff2);
        //             }
        //         }
        //         self = C;
        //         return self;
        //     },
        //     "Multiply this SparseOperator by another SparseOperator")
        .def(
            "__matmul__",
            [](const SparseOperator& lhs, const SparseOperator& rhs) { return lhs * rhs; },
            "Multiply two SparseOperator objects")
        .def(
            "commutator",
            [](const SparseOperator& lhs, const SparseOperator& rhs) {
                return commutator(lhs, rhs);
            },
            "Compute the commutator of two SparseOperator objects")
        .def(
            "__itruediv__",
            [](SparseOperator& self, sparse_scalar_t scalar) {
                return self /= scalar; // Call the in-place division operator
            },
            nb::is_operator(), "Divide this SparseOperator by a scalar")
        .def(
            "__truediv__",
            [](const SparseOperator& self, sparse_scalar_t scalar) {
                return self / scalar; // Call the division operator
            },
            nb::is_operator(), "Divide this SparseOperator by a scalar")
        .def(
            "__mul__",
            [](const SparseOperator& self, sparse_scalar_t scalar) {
                return self * scalar; // This uses the operator* we defined
            },
            "Multiply a SparseOperator by a scalar")
        .def(
            "__rmul__",
            [](const SparseOperator& self, sparse_scalar_t scalar) {
                // This enables the reversed operation: scalar * SparseOperator
                return self * scalar; // Reuse the __mul__ logic
            },
            "Multiply a scalar by a SparseOperator")
        .def(
            "__rdiv__",
            [](const SparseOperator& self, sparse_scalar_t scalar) {
                return self * (1.0 / scalar); // This uses the operator* we defined
            },
            "Divide a scalar by a SparseOperator")
        // .def(
        //     "__mul__",
        //     [](const SparseOperator& self, const SparseOperator& other) {
        //         SparseOperator C;
        //         for (const auto& [op, coeff] : self.elements()) {
        //             for (const auto& [op2, coeff2] : other.elements()) {
        //                 new_product2(C, op, op2, coeff * coeff2);
        //             }
        //         }
        //         return C;
        //     },
        //     "Multiply two SparseOperators")
        // .def(nb::self - nb::self, "Subtract two SparseOperators")
        .def(
            "__neg__", [](const SparseOperator& self) { return -self; }, "Negate the operator")
        .def("copy", &SparseOperator::copy, "Create a copy of this SparseOperator")
        .def(
            "norm", [](const SparseOperator& op) { return op.norm(); },
            "Compute the norm of the operator")
        .def("str", &SparseOperator::str, "Get a string representation of the operator")
        .def("latex", &SparseOperator::latex, "Get a LaTeX representation of the operator")
        .def(
            "adjoint", [](const SparseOperator& op) { return op.adjoint(); }, "Get the adjoint")
        .def("__eq__", &SparseOperator::operator==, "Check if two SparseOperators are equal")
        .def(
            "__repr__", [](const SparseOperator& op) { return join(op.str(), "\n"); },
            "Get a string representation of the operator")
        .def(
            "__str__", [](const SparseOperator& op) { return join(op.str(), "\n"); },
            "Get a string representation of the operator")
        .def(
            "apply_to_state",
            [](const SparseOperator& op, const SparseState& state, double screen_thresh) {
                return apply_operator_lin(op, state, screen_thresh);
            },
            "state"_a, "screen_thresh"_a = 1.0e-12, "Apply the operator to a state")
        // .def(
        //     "fact_trans_lin",
        //     [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double
        //     screen_thresh) {
        //         auto O_copy = O;
        //         fact_trans_lin(O_copy, T, reverse, screen_thresh);
        //         return O_copy;
        //     },
        //     "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
        //     "Evaluate ... (1 - T1) O (1 + T1) ...")

        // .def(
        //     "fact_unitary_trans_antiherm",
        //     [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double
        //     screen_thresh) {
        //         auto O_copy = O;
        //         fact_unitary_trans_antiherm(O_copy, T, reverse, screen_thresh);
        //         return O_copy;
        //     },
        //     "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
        //     "Evaluate ... exp(T1^dagger - T1) O exp(T1 - T1^dagger) ...")

        // .def(
        //     "fact_unitary_trans_antiherm_grad",
        //     [](SparseOperator& O, const SparseOperatorList& T, size_t n, bool reverse,
        //        double screen_thresh) {
        //         auto O_copy = O;
        //         fact_unitary_trans_antiherm_grad(O_copy, T, n, reverse, screen_thresh);
        //         return O_copy;
        //     },
        //     "T"_a, "n"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
        //     "Evaluate the gradient of ... exp(T1^dagger - T1) O exp(T1 - T1^dagger) ...")

        // .def(
        //     "fact_unitary_trans_imagherm",
        //     [](SparseOperator& O, const SparseOperatorList& T, bool reverse, double
        //     screen_thresh) {
        //         auto O_copy = O;
        //         fact_unitary_trans_imagherm(O_copy, T, reverse, screen_thresh);
        //         return O_copy;
        //     },
        //     "T"_a, "reverse"_a = false, "screen_thresh"_a = 1.0e-12,
        //     "Evaluate ... exp(i (T1^dagger + T1)) O exp(-i(T1 + T1^dagger)) ...")
        .def(
            "__matmul__",
            [](const SparseOperator& op, const SparseState& st) {
                return apply_operator_lin(op, st);
            },
            "Multiply a SparseOperator and a SparseState")
        .def(
            "matrix",
            [](const SparseOperator& sop, const std::vector<Determinant>& dets,
               double screen_thresh) {
                std::vector<sparse_scalar_t> elements;
                for (const auto& deti : dets) {
                    SparseState deti_state;
                    deti_state.add(deti, 1.0);
                    auto op_deti = apply_operator_lin(sop, deti_state, screen_thresh);
                    for (const auto& detj : dets) {
                        elements.push_back(op_deti[detj]);
                    }
                }
                return elements;
            },
            "dets"_a, "screen_thresh"_a = 1.0e-12,
            "Compute the matrix elements of the operator between a list of determinants");
    m.def(
        "sparse_operator",
        [](const std::string& s, sparse_scalar_t coefficient, bool allow_reordering) {
            SparseOperator sop;
            sop.add_term_from_str(s, coefficient, allow_reordering);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false,
        "Create a SparseOperator object from a string and a complex");

    m.def(
        "sparse_operator",
        [](const std::vector<std::pair<std::string, sparse_scalar_t>>& list,
           bool allow_reordering) {
            SparseOperator sop;
            for (const auto& [s, coefficient] : list) {
                sop.add_term_from_str(s, coefficient, allow_reordering);
            }
            return sop;
        },
        "list"_a, "allow_reordering"_a = false,
        "Create a SparseOperator object from a list of Tuple[str, complex]");

    m.def(
        "sparse_operator",
        [](const SQOperatorString& sqop, sparse_scalar_t coefficient) {
            SparseOperator sop;
            sop.add(sqop, coefficient);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1),
        "Create a SparseOperator object from a SQOperatorString and a complex");

    m.def(
        "sparse_operator",
        [](const std::vector<std::pair<SQOperatorString, sparse_scalar_t>>& list) {
            SparseOperator sop;
            for (const auto& [sqop, coefficient] : list) {
                sop.add(sqop, coefficient);
            }
            return sop;
        },
        "list"_a, "Create a SparseOperator object from a list of Tuple[SQOperatorString, complex]");

    m.def("new_product", [](const SparseOperator A, const SparseOperator B) {
        SparseOperator C;
        SQOperatorProductComputer computer;
        for (const auto& [op, coeff] : A.elements()) {
            for (const auto& [op2, coeff2] : B.elements()) {
                computer.product(op, op2, coeff * coeff2,
                                 [&C](const SQOperatorString& sqop, const sparse_scalar_t c) {
                                     C.add(sqop, c);
                                 });
            }
        }
        return C;
    });

    // m.def("new_product2", [](const SparseOperator A, const SparseOperator B) {
    //     SparseOperator C;
    //     for (const auto& [op, coeff] : A.elements()) {
    //         for (const auto& [op2, coeff2] : B.elements()) {
    //             new_product2(C, op, op2, coeff * coeff2);
    //         }
    //     }
    //     return C;
    // });

    // m.def("sparse_operator_hamiltonian", &sparse_operator_hamiltonian,
    //       "Create a SparseOperator object from an ActiveSpaceIntegrals object", "as_ints"_a,
    //       "screen_thresh"_a = 1.0e-12);

    m.def("sparse_operator_hamiltonian", &sparse_operator_hamiltonian,
          "Create a SparseOperator object from integrals", "scalar"_a, "oei_a"_a, "oei_b"_a,
          "tei_aa"_a, "tei_ab"_a, "tei_bb"_a, "screen_thresh"_a = 1.0e-12,
          "Create a SparseOperator object from one-electron and two-electron integrals");
}

void export_sparse_operator_list_api(nb::module_& m) {
    nb::class_<SparseOperatorList>(m, "SparseOperatorList",
                                   "A class to represent a list of sparse operators")
        .def(nb::init<>())
        .def(nb::init<SparseOperatorList>())
        .def("add", &SparseOperatorList::add)
        .def("add", &SparseOperatorList::add_term_from_str, "str"_a,
             "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false)
        // .def("add",
        //      [](SparseOperatorList& op, const, sparse_scalar_t value, bool allow_reordering) {
        //          make_sq_operator_string_from_list op.add(sqop, value);
        //      })
        .def("add_term",
             nb::overload_cast<const std::vector<std::tuple<bool, bool, int>>&, double, bool>(
                 &SparseOperatorList::add_term),
             "op_list"_a, "value"_a = 0.0, "allow_reordering"_a = false)
        .def(
            "add",
            [](SparseOperatorList& op, const std::vector<size_t>& acre,
               const std::vector<size_t>& bcre, const std::vector<size_t>& aann,
               const std::vector<size_t>& bann, sparse_scalar_t coeff) {
                op.add(SQOperatorString({acre.begin(), acre.end()}, {bcre.begin(), bcre.end()},
                                        {aann.begin(), aann.end()}, {bann.begin(), bann.end()}),
                       coeff);
            },
            "acre"_a, "bcre"_a, "aann"_a, "bann"_a, "coeff"_a = sparse_scalar_t(1),
            "Add a term to the operator by passing lists of creation and annihilation indices. "
            "This version is faster than the string version and does not check for reordering")
        .def("to_operator", &SparseOperatorList::to_operator)
        .def(
            "remove",
            [](SparseOperatorList& op, const std::string& s) {
                const auto [sqop, _] = make_sq_operator_string(s, false);
                op.remove(sqop);
            },
            "Remove a specific element from the vector space")
        .def("__len__", &SparseOperatorList::size)
        .def(
            "__iter__",
            [](const SparseOperatorList& v) {
                return nb::make_iterator(nb::type<SparseOperatorList>(), "item_iterator",
                                         v.elements().begin(), v.elements().end());
            },
            nb::keep_alive<0, 1>())
        .def("__repr__", [](const SparseOperatorList& op) { return join(op.str(), "\n"); })
        .def("__str__", [](const SparseOperatorList& op) { return join(op.str(), "\n"); })
        .def(
            "__getitem__", [](const SparseOperatorList& op, const size_t n) { return op[n]; },
            "Get the coefficient of a term")
        .def(
            "__getitem__",
            [](const SparseOperatorList& op, const std::string& s) {
                const auto [sqop, factor] = make_sq_operator_string(s, false);
                return factor * op[sqop];
            },
            "Get the coefficient of a term")
        .def(
            "__setitem__",
            [](SparseOperatorList& op, const size_t n, sparse_scalar_t value) { op[n] = value; },
            "Set the coefficient of a term")
        .def("coefficients",
             [](SparseOperatorList& op) {
                 std::vector<sparse_scalar_t> values(op.size());
                 for (size_t i = 0, max = op.size(); i < max; ++i) {
                     values[i] = op[i];
                 }
                 return values;
             })
        .def("set_coefficients",
             [](SparseOperatorList& op, const std::vector<sparse_scalar_t>& values) {
                 if (op.size() != values.size()) {
                     throw std::invalid_argument(
                         "The size of the list of coefficients must match the "
                         "size of the operator list");
                 }
                 for (size_t i = 0; i < op.size(); ++i) {
                     op[i] = values[i];
                 }
             })
        .def("reverse", &SparseOperatorList::reverse, "Reverse the order of the operators")
        .def(
            "__call__",
            [](const SparseOperatorList& op, const size_t n) {
                if (n >= op.size()) {
                    throw std::out_of_range("Index out of range");
                }
                return op(n);
            },
            "Get the nth operator")
        .def(
            "__matmul__",
            [](const SparseOperatorList& op, const SparseState& st) {
                // form a temporary SparseOperator from the list of operators
                auto sop = op.to_operator();
                return apply_operator_lin(sop, st);
            },
            "Multiply a SparseOperator and a SparseState")
        .def(
            "__add__",
            [](const SparseOperatorList& op1, const SparseOperatorList& op2) {
                SparseOperatorList result = op1;
                result += op2;
                return result;
            },
            "Add (concatenate) two SparseOperatorList objects")
        .def(
            "__iadd__",
            [](SparseOperatorList& op1, const SparseOperatorList& op2) {
                op1 += op2;
                return op1;
            },
            "Add (concatenate) a SparseOperatorList object to this SparseOperatorList object")
        .def(
            "apply_to_state",
            [](const SparseOperatorList& op, const SparseState& state, double screen_thresh) {
                auto sop = op.to_operator();
                return apply_operator_lin(sop, state, screen_thresh);
            },
            "state"_a, "screen_thresh"_a = 1.0e-12, "Apply the operator to a state");
}

} // namespace forte2