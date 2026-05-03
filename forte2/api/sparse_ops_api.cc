#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>

#include "helpers/ndarray.h"
#include "helpers/string_algorithms.h"

#include "sparse/sparse_operator.h"
#include "sparse/sparse_operator_product.h"
#include "sparse/sparse_state.h"
#include "sparse/sparse_operator_hamiltonian.h"
#include "sparse/sparse_exp.h"
#include "sparse/sq_operator_string.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_sparse_normal_order_api(nb::module_& m);

namespace {
void export_sq_operator_string_api(nb::module_& m);
void export_sparse_state_api(nb::module_& m);
void export_sparse_operator_api(nb::module_& m);
void export_sparse_operator_list_api(nb::module_& m);
void export_sparse_exp_api(nb::module_& m);
void export_sparse_fact_exp_api(nb::module_& m);
} // namespace

void export_sparse_ops_api(nb::module_& m) {
    nb::module_ sub_m =
        m.def_submodule("sparse_ops", "Sparse operators, states, and their operations");

    export_sq_operator_string_api(sub_m);

    export_sparse_state_api(sub_m);

    export_sparse_operator_api(sub_m);

    export_sparse_operator_list_api(sub_m);

    export_sparse_exp_api(sub_m);

    export_sparse_fact_exp_api(sub_m);

    export_sparse_normal_order_api(sub_m);
}

namespace {
void export_sparse_operator_api(nb::module_& sub_m) {

    nb::class_<SparseOperator>(sub_m, "SparseOperator", "A class to represent a sparse operator")
        // Constructors
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<SparseOperator>(), "Copy constructor")
        .def(nb::init<const SparseOperator::old_container&>(),
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
            "rank_screened_commutator",
            [](const SparseOperator& lhs, const SparseOperator& rhs, int max_rank,
               double screen_thresh) {
                return rank_screened_commutator(lhs, rhs, max_rank, screen_thresh);
            },
            "rhs"_a, "max_rank"_a, "screen_thresh"_a = 1.0e-12,
            "Compute a commutator while skipping term pairs that cannot contribute to max_rank")
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
                auto elements = make_zeros<nb::numpy, std::complex<double>, 2>(
                    std::array<size_t, 2>({dets.size(), dets.size()}));
                for (size_t i = 0; const auto& deti : dets) {
                    SparseState deti_state;
                    deti_state.add(deti, 1.0);
                    auto op_deti = apply_operator_lin(sop, deti_state, screen_thresh);
                    for (size_t j = 0; const auto& detj : dets) {
                        elements(i, j) = op_deti[detj];
                        ++j;
                    }
                    ++i;
                }
                return elements;
            },
            "dets"_a, "screen_thresh"_a = 1.0e-12,
            "Compute the matrix elements of the operator between a list of determinants");

    sub_m.def(
        "sparse_operator",
        [](const std::string& s, sparse_scalar_t coefficient, bool allow_reordering) {
            SparseOperator sop;
            sop.add_term_from_str(s, coefficient, allow_reordering);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false,
        "Create a SparseOperator object from a string and a complex");

    sub_m.def(
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

    sub_m.def(
        "sparse_operator",
        [](const SQOperatorString& sqop, sparse_scalar_t coefficient) {
            SparseOperator sop;
            sop.add(sqop, coefficient);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1),
        "Create a SparseOperator object from a SQOperatorString and a complex");

    sub_m.def(
        "sparse_operator",
        [](const std::vector<std::pair<SQOperatorString, sparse_scalar_t>>& list) {
            SparseOperator sop;
            for (const auto& [sqop, coefficient] : list) {
                sop.add(sqop, coefficient);
            }
            return sop;
        },
        "list"_a, "Create a SparseOperator object from a list of Tuple[SQOperatorString, complex]");

    sub_m.def("new_product", [](const SparseOperator A, const SparseOperator B) {
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

    // sub_m.def("new_product2", [](const SparseOperator A, const SparseOperator B) {
    //     SparseOperator C;
    //     for (const auto& [op, coeff] : A.elements()) {
    //         for (const auto& [op2, coeff2] : B.elements()) {
    //             new_product2(C, op, op2, coeff * coeff2);
    //         }
    //     }
    //     return C;
    // });
    // overloaded: real Hamiltonian
    sub_m.def(
        "sparse_operator_hamiltonian",
        [](double scalar_energy, np_matrix one_electron_integrals,
           np_tensor4 two_electron_integrals, double screen_thresh) {
            return sparse_operator_hamiltonian(scalar_energy, one_electron_integrals,
                                               two_electron_integrals, screen_thresh);
        },
        "scalar_energy"_a, "one_electron_integrals"_a, "two_electron_integrals"_a,
        "screen_thresh"_a = 1e-12,
        "Create a SparseOperator object representing the second quantized Hamiltonian.");
    // overloaded: complex Hamiltonian
    sub_m.def(
        "sparse_operator_hamiltonian",
        [](double scalar_energy, np_matrix_complex one_electron_integrals,
           np_tensor4_complex two_electron_integrals, double screen_thresh) {
            return sparse_operator_hamiltonian(scalar_energy, one_electron_integrals,
                                               two_electron_integrals, screen_thresh);
        },
        "scalar_energy"_a, "one_electron_integrals"_a, "two_electron_integrals"_a,
        "screen_thresh"_a = 1e-12,
        "Create a SparseOperator object representing the second quantized Hamiltonian.");
}

void export_sparse_operator_list_api(nb::module_& sub_m) {
    nb::class_<SparseOperatorList>(sub_m, "SparseOperatorList",
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
        .def("pop_left", &SparseOperatorList::pop_left, "Remove the leftmost operator")
        .def("pop_right", &SparseOperatorList::pop_right, "Remove the rightmost operator")
        .def("slice", &SparseOperatorList::slice, "start"_a, "end"_a,
             "Return a slice of the operator")
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

    sub_m.def(
        "operator_list",
        [](const std::string& s, sparse_scalar_t coefficient, bool allow_reordering) {
            SparseOperatorList sop;
            sop.add_term_from_str(s, coefficient, allow_reordering);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1), "allow_reordering"_a = false,
        "Create a SparseOperatorList object from a string and a complex");

    sub_m.def(
        "operator_list",
        [](const std::vector<std::pair<std::string, sparse_scalar_t>>& list,
           bool allow_reordering) {
            SparseOperatorList sop;
            for (const auto& [s, coefficient] : list) {
                sop.add_term_from_str(s, coefficient, allow_reordering);
            }
            return sop;
        },
        "list"_a, "allow_reordering"_a = false,
        "Create a SparseOperatorList object from a list of Tuple[str, complex]");

    sub_m.def(
        "operator_list",
        [](const SQOperatorString& sqop, sparse_scalar_t coefficient) {
            SparseOperatorList sop;
            sop.add(sqop, coefficient);
            return sop;
        },
        "s"_a, "coefficient"_a = sparse_scalar_t(1),
        "Create a SparseOperatorList object from a SQOperatorString and a complex");

    sub_m.def(
        "operator_list",
        [](const std::vector<std::pair<SQOperatorString, sparse_scalar_t>>& list) {
            SparseOperatorList sop;
            for (const auto& [sqop, coefficient] : list) {
                sop.add(sqop, coefficient);
            }
            return sop;
        },
        "list"_a,
        "Create a SparseOperatorList object from a list of Tuple[SQOperatorString, complex]");
}

void export_sq_operator_string_api(nb::module_& sub_m) {
    nb::class_<SQOperatorString>(sub_m, "SQOperatorString",
                                 "A class to represent a string of creation/annihilation operators")
        .def(nb::init<const Determinant&, const Determinant&>())
        .def(
            "cre", [](const SQOperatorString& sqop) { return sqop.cre(); },
            "Get the creation operator string")
        .def(
            "ann", [](const SQOperatorString& sqop) { return sqop.ann(); },
            "Get the annihilation operator string")
        .def(
            "sign_mask", [](const SQOperatorString& sqop) { return sqop.sign_mask(); },
            "Get the precomputed sign mask")
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

    nb::enum_<CommutatorType>(sub_m, "CommutatorType")
        .value("commute", CommutatorType::Commute)
        .value("anticommute", CommutatorType::AntiCommute)
        .value("may_not_commute", CommutatorType::MayNotCommute);

    sub_m.def(
        "sqop",
        [](const std::string& s, bool allow_reordering) {
            return make_sq_operator_string(s, allow_reordering);
        },
        "s"_a, "allow_reordering"_a = false,
        "Create an operator string from a string representation (default: no not allow "
        "reordering)");

    sub_m.def(
        "compute_sign_mask",
        [](const Determinant& cre, const Determinant& ann) {
            Determinant sign_mask = Determinant::zero();
            compute_sign_mask(cre, ann, sign_mask);
            return sign_mask;
        },
        "cre"_a, "ann"_a,
        "Compute the sign mask associated with a set of creation and annihilation operators");

    sub_m.def("commutator_type", &commutator_type, "lhs"_a, "rhs"_a,
              "Get the commutator type of two operator strings");
}

void export_sparse_state_api(nb::module_& sub_m) {
    nb::class_<SparseState>(sub_m, "SparseState", "A class to represent a vector of determinants")
        .def(nb::init<>(), "Default constructor")
        .def(nb::init<const SparseState&>(), "Copy constructor")
        .def(nb::init<const SparseState::old_container&>(),
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
        .def("__mul__", &SparseState::operator*, "Multiply this SparseState by a scalar")
        .def("__rmul__", &SparseState::operator*, "Multiply a scalar by this SparseState")
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

    sub_m.def("apply_op", &apply_operator_lin, "sop"_a, "state0"_a, "screen_thresh"_a = 1.0e-12);

    sub_m.def("apply_antiherm", &apply_operator_antiherm, "sop"_a, "state0"_a,
              "screen_thresh"_a = 1.0e-12);

    sub_m.def("apply_number_projector", &apply_number_projector);

    sub_m.def("get_projection", &get_projection);

    sub_m.def(
        "spin2",
        [](const SparseState& left_state, const SparseState& right_state) {
            return spin2(left_state, right_state);
        },
        "Calculate the <left_state|S^2|right_state> expectation value");

    sub_m.def("overlap", &overlap);

    sub_m.def("normalize", &normalize, "Returns a normalized version of the input SparseState");
}

void export_sparse_exp_api(nb::module_& sub_m) {
    nb::class_<SparseExp>(sub_m, "SparseExp",
                          "A class to compute the exponential of a sparse operator")
        .def(nb::init<int, double>(), "maxk"_a = 19, "screen_thresh"_a = 1.0e-12)
        .def("apply_op",
             nb::overload_cast<const SparseOperator&, const SparseState&, double>(
                 &SparseExp::apply_op),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0,
             "Apply the exponential of a SparseOperator to a state: exp(scaling_factor * sop) "
             "|state>")
        .def("apply_op",
             nb::overload_cast<const SparseOperatorList&, const SparseState&, double>(
                 &SparseExp::apply_op),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0,
             "Apply the exponential of a SparseOperatorList to a state: exp(scaling_factor * sop) "
             "|state>")
        .def("apply_antiherm",
             nb::overload_cast<const SparseOperator&, const SparseState&, double>(
                 &SparseExp::apply_antiherm),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0,
             "Apply the antihermitian "
             "exponential of a SparseOperator to a state: exp(scaling_factor * (sop - sop^dagger)) "
             "|state>")
        .def("apply_antiherm",
             nb::overload_cast<const SparseOperatorList&, const SparseState&, double>(
                 &SparseExp::apply_antiherm),
             "sop"_a, "state"_a, "scaling_factor"_a = 1.0,
             "Apply the antihermitian "
             "exponential of a SparseOperatorList to a state: exp(scaling_factor * (sop - "
             "sop^dagger)) "
             "|state");
}

void export_sparse_fact_exp_api(nb::module_& sub_m) {
    nb::class_<SparseFactExp>(
        sub_m, "SparseFactExp",
        "A class to compute the product exponential of a sparse operator using factorization")
        .def(nb::init<double>(), "screen_thresh"_a = 1.0e-12)
        .def("apply_op", &SparseFactExp::apply_op, "sop"_a, "state"_a, "inverse"_a = false,
             "reverse"_a = false,
             "Apply the factorized exponential of a SparseOperator to a state: "
             "... exp(op2) exp(op1) |state>. inverse=True computes the inverse, and reverse=True"
             "applies the operators in reverse order")
        .def("apply_antiherm", &SparseFactExp::apply_antiherm, "sop"_a, "state"_a,
             "inverse"_a = false, "reverse"_a = false,
             "Apply the factorized antihermitian "
             "exponential of a SparseOperator to a state: "
             "... exp(op2 - op2^dagger) exp(op1 - op1^dagger) |state>. inverse=True computes the "
             "inverse, and reverse=True applies the operators in reverse order")
        .def("apply_antiherm_deriv", &SparseFactExp::apply_antiherm_deriv, "sqop"_a, "t"_a,
             "state"_a);
}
} // namespace

} // namespace forte2
