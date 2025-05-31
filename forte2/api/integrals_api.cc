#if __cplusplus >= 202002L
// C++20 or later
#include <format>
#endif

#include <libint2.hpp>
#include <libint2/shell.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include "ints/basis.h"
#include "ints/fock_builder.h"
#include "ints/nuclear_repulsion.h"
#include "ints/one_electron.h"
#include "ints/two_electron.h"
#include "ints/value_at_points.h"

namespace nb = nanobind;
using namespace nb::literals;

/// @brief This file contains the Python bindings for the integrals API.
///        It includes the bindings for the basis set and shell classes and functions to compute
///        integrals and evaluate basis functions at points.
namespace forte2 {

void export_shell_api(nb::module_& m);
void export_basis_api(nb::module_& m);
void export_scalar_api(nb::module_& m);
void export_one_electron_api(nb::module_& m);
void export_two_electron_api(nb::module_& m);
void export_value_at_points_api(nb::module_& m);

void export_integrals_api(nb::module_& m) {
    nb::module_ sub_m = m.def_submodule("ints", "Integrals submodule");

    export_shell_api(sub_m);

    export_basis_api(sub_m);

    export_scalar_api(sub_m);

    export_one_electron_api(sub_m);

    export_two_electron_api(sub_m);

    export_value_at_points_api(sub_m);
}

void export_shell_api(nb::module_& sub_m) {
    /// @brief Shell class bindings
    /// @details The Shell class is a wrapper around libint2::Shell and provides
    ///          a Python interface to create and manipulate Gaussian shells.
    ///          The Shell class represents a collection of Gaussian primitives (contractions)
    ///          with the same angular momentum and center. In forte2, we assume
    ///          that a shell is a single contraction with multiple primitives.
    ///          The general form of a shell is:
    ///          \f[\chi_\mu(r) = SH(x,y,z,l,m) \sum_{i=1}^{n} c_{i} e^{-\alpha_i (r - O)^2} \f]
    ///          where SH is a solid harmonic, \f$ c_i \f$ are the contraction coefficients,
    ///          \f$ \alpha_i \f$ are the Gaussian exponents, \f$ O \f$ is center of the shell.
    ///          \f$ l \f$ is the angular momentum of the shell and \f$ m \f$ is the magnetic
    ///          quantum number, which can take value of m = -l, -l+1, ..., l.
    nb::class_<libint2::Shell>(sub_m, "Shell")
        .def(nb::init<>())
        .def(
            "__init__",
            [](libint2::Shell* t, int l, const std::vector<double>& exponents,
               const std::vector<double>& coeffs, const std::array<double, 3>& origin, bool is_pure,
               bool embed_normalization_into_coefficients) {
                // Convert std::vector to libint2::svector to match libint2's constructor
                auto l2_exponents = libint2::svector<double>(exponents.begin(), exponents.end());
                auto l2_coeffs = libint2::svector<double>(coeffs.begin(), coeffs.end());
                new (t) libint2::Shell{l2_exponents,
                                       {{l, is_pure, l2_coeffs}},
                                       origin,
                                       embed_normalization_into_coefficients};
            },
            "l"_a, "exponents"_a, "coeffs"_a, "center"_a, "is_pure"_a = true,
            "embed_normalization_into_coefficients"_a = true,
            "Construct a shell from the angular momentum (l) and a list of exponents and "
            "coefficients.")
#if __cplusplus >= 202002L
        // C++20 or later
        .def("__repr__",
             [](const libint2::Shell& s) {
                 std::string str;
                 str = "l = " + std::to_string(s.contr[0].l) +
                       " nprim = " + std::to_string(s.nprim());
                 for (std::size_t i = 0; i < s.nprim(); ++i) {
                     str += std::format("\n  {0:10.6f} {1:10.6f}", s.alpha[i], s.contr[0].coeff[i]);
                 }
                 return str;
             })
#endif
        .def_prop_ro(
            "size", [](libint2::Shell& s) { return s.size(); },
            "The number of basis functions in the shell (e.g., for l = 2, size = 5).")
        .def_prop_ro(
            "ncontr", [](libint2::Shell& s) { return s.ncontr(); },
            "The number of contractions in the shell.")
        .def_prop_ro(
            "nprim", [](libint2::Shell& s) { return s.nprim(); },
            "The number of primitives Gaussians in the shell.")
        .def_prop_ro(
            "l", [](libint2::Shell& s) { return s.contr[0].l; },
            "The angular momentum of the shell.")
        .def_prop_ro(
            "coeff",
            [](libint2::Shell& s) {
                return std::vector<double>(s.contr[0].coeff.begin(), s.contr[0].coeff.end());
            },
            "The coefficients of the primitives in the shell.")
        .def_prop_ro(
            "exponents",
            [](libint2::Shell& s) { return std::vector<double>(s.alpha.begin(), s.alpha.end()); },
            "The exponents of the primitives in the shell.")
        .def_prop_ro(
            "is_pure", [](libint2::Shell& s) { return s.contr[0].pure; },
            "Is the shell pure? (i.e., we have 5d and 7f functions)")
        .def_prop_ro(
            "center", [](libint2::Shell& s) { return s.O; },
            "The center of the shell (x, y, z) in bohr.");
}

void export_basis_api(nb::module_& sub_m) {
    nb::class_<Basis>(sub_m, "Basis")
        .def(nb::init<>())
        .def("add", &Basis::add, "shell"_a)
        .def("set_name", &Basis::set_name, "name"_a)
        .def("__getitem__", &Basis::operator[], "i"_a)
        .def("__len__", &Basis::size)
        .def_prop_ro("shell_first_and_size", &Basis::shell_first_and_size)
        .def_prop_ro("center_first_and_last", &Basis::center_first_and_last)
        .def_prop_ro("size", &Basis::size)
        .def_prop_ro("max_l", &Basis::max_l)
        .def_prop_ro("name", &Basis::name)
        .def_prop_ro("max_nprim", &Basis::max_nprim)
        .def_prop_ro("nprim", &Basis::max_nprim)
        .def_prop_ro("nshells", &Basis::nshells)
        .def("__repr__", [](const Basis& b) {
            std::ostringstream oss;
            oss << "<Basis '" << b.name() << "' with " << b.size() << " basis functions>";
            return oss.str();
        });

    sub_m.def("shell_label", shell_label, "l"_a, "idx"_a,
              "Returns a label for a given angular momentum (l) and index (idx).");

    sub_m.def(
        "evaluate_shell",
        [](const libint2::Shell& shell, const std::array<double, 3>& point) {
            // Allocate a buffer for the result
            std::vector<double> buffer(shell.size());
            // Evaluate the shell at the given point
            evaluate_shell(shell, point, buffer.data());
            return buffer;
        },
        "shell"_a, "point"_a,
        "Evaluate the shell at a given point. Returns a list of values for each basis function.");
}

void export_value_at_points_api(nb::module_& sub_m) {
    sub_m.def("basis_at_points", &basis_at_points, "basis"_a, "points"_a);

    sub_m.def(
        "orbitals_at_points", &orbitals_at_points, "basis"_a, "points"_a, "C"_a,
        "Evaluate the orbitals on a set of points. Returns a 2D array of shape (npoints, norb).");

    sub_m.def("orbitals_on_grid", &orbitals_on_grid, "basis"_a, "C"_a, "min"_a, "npoints"_a,
              "axis"_a);
}

void export_scalar_api(nb::module_& sub_m) {
    sub_m.def(
        "nuclear_repulsion",
        [](std::vector<std::pair<double, std::array<double, 3>>> charges) {
            return nuclear_repulsion(charges);
        },
        "charges"_a);
}

void export_one_electron_api(nb::module_& sub_m) {
    sub_m.def("overlap", &overlap, "basis1"_a, "basis2"_a,
              R"pbdoc(
Compute the overlap integral matrix.

Parameters
----------
b1 : forte2.Basis
    First basis set.
b2 : forte2.Basis
    Second basis set.

Returns
-------
ndarray, shape = (nb1, nb2)
    Overlap integrals matrix.
)pbdoc");
    sub_m.def(
        "overlap", [](const Basis& basis) { return overlap(basis, basis); }, "basis"_a);

    sub_m.def("kinetic", &kinetic, "basis1"_a, "basis2"_a);
    sub_m.def(
        "kinetic", [](const Basis& basis) { return kinetic(basis, basis); }, "basis"_a);

    sub_m.def(
        "nuclear",
        [](const Basis& basis1, const Basis& basis2,
           std::vector<std::pair<double, std::array<double, 3>>> charges) {
            return nuclear(basis1, basis2, charges);
        },
        "basis1"_a, "basis2"_a, "charges"_a);
    sub_m.def(
        "nuclear",
        [](const Basis& basis, std::vector<std::pair<double, std::array<double, 3>>> charges) {
            return nuclear(basis, basis, charges);
        },
        "basis"_a, "charges"_a);

    sub_m.def(
        "emultipole1",
        [](const Basis& basis1, const Basis& basis2, std::array<double, 3> origin) {
            return emultipole1(basis1, basis2, origin);
        },
        "basis1"_a, "basis2"_a, "origin"_a = std::array<double, 3>{0.0, 0.0, 0.0});
    sub_m.def(
        "emultipole1",
        [](const Basis& basis, std::array<double, 3> origin) {
            return emultipole1(basis, basis, origin);
        },
        "basis"_a, "origin"_a = std::array<double, 3>{0.0, 0.0, 0.0});

    sub_m.def(
        "emultipole2",
        [](const Basis& basis1, const Basis& basis2, std::array<double, 3> origin) {
            return emultipole2(basis1, basis2, origin);
        },
        "basis1"_a, "basis2"_a, "origin"_a = std::array<double, 3>{0.0, 0.0, 0.0});
    sub_m.def(
        "emultipole2",
        [](const Basis& basis, std::array<double, 3> origin) {
            return emultipole2(basis, basis, origin);
        },
        "basis"_a, "origin"_a = std::array<double, 3>{0.0, 0.0, 0.0});

    sub_m.def(
        "emultipole3",
        [](const Basis& basis1, const Basis& basis2, std::array<double, 3> origin) {
            return emultipole3(basis1, basis2, origin);
        },
        "basis1"_a, "basis2"_a, "origin"_a = std::array<double, 3>{0.0, 0.0, 0.0});
    sub_m.def(
        "emultipole3",
        [](const Basis& basis, std::array<double, 3> origin) {
            return emultipole3(basis, basis, origin);
        },
        "basis"_a, "origin"_a = std::array<double, 3>{0.0, 0.0, 0.0});

    sub_m.def(
        "opVop",
        [](const Basis& basis1, const Basis& basis2,
           std::vector<std::pair<double, std::array<double, 3>>> charges) {
            return opVop(basis1, basis2, charges);
        },
        "basis1"_a, "basis2"_a, "charges"_a);
    sub_m.def(
        "opVop",
        [](const Basis& basis, std::vector<std::pair<double, std::array<double, 3>>> charges) {
            return opVop(basis, basis, charges);
        },
        "basis"_a, "charges"_a);

    sub_m.def(
        "erf_nuclear",
        [](const Basis& basis1, const Basis& basis2,
           std::tuple<double, std::vector<std::pair<double, std::array<double, 3>>>>
               omega_charges) { return erf_nuclear(basis1, basis2, omega_charges); },
        "basis1"_a, "basis2"_a, "omega_charges"_a);

    sub_m.def(
        "erfc_nuclear",
        [](const Basis& basis1, const Basis& basis2,
           std::tuple<double, std::vector<std::pair<double, std::array<double, 3>>>>
               omega_charges) { return erfc_nuclear(basis1, basis2, omega_charges); },
        "basis1"_a, "basis2"_a, "omega_charges"_a);
}

void export_two_electron_api(nb::module_& sub_m) {
    sub_m.def(
        "coulomb_4c",
        [](const Basis& basis1, const Basis& basis2, const Basis& basis3, const Basis& basis4) {
            return coulomb_4c(basis1, basis2, basis3, basis4);
        },
        "basis1"_a, "basis2"_a, "basis3"_a, "basis4"_a);

    sub_m.def(
        "coulomb_4c", [](const Basis& basis) { return coulomb_4c(basis, basis, basis, basis); },
        "basis"_a);

    sub_m.def(
        "coulomb_3c",
        [](const Basis& basis1, const Basis& basis2, const Basis& basis3) {
            return coulomb_3c(basis1, basis2, basis3);
        },
        "basis1"_a, "basis2"_a, "basis3"_a);

    sub_m.def(
        "coulomb_2c",
        [](const Basis& basis1, const Basis& basis2) { return coulomb_2c(basis1, basis2); },
        "basis1"_a, "basis2"_a);
    sub_m.def(
        "coulomb_2c", [](const Basis& basis) { return coulomb_2c(basis, basis); }, "basis"_a);

    sub_m.def(
        "erf_coulomb_3c",
        [](const Basis& basis1, const Basis& basis2, const Basis& basis3, double omega) {
            return erf_coulomb_3c(basis1, basis2, basis3, omega);
        },
        "basis1"_a, "basis2"_a, "basis3"_a, "omega"_a);

    sub_m.def(
        "erf_coulomb_2c",
        [](const Basis& basis1, const Basis& basis2, double omega) {
            return erf_coulomb_2c(basis1, basis2, omega);
        },
        "basis1"_a, "basis2"_a, "omega"_a);

    sub_m.def(
        "erfc_coulomb_3c",
        [](const Basis& basis1, const Basis& basis2, const Basis& basis3, double omega) {
            return erfc_coulomb_3c(basis1, basis2, basis3, omega);
        },
        "basis1"_a, "basis2"_a, "basis3"_a, "omega"_a);

    sub_m.def(
        "erfc_coulomb_2c",
        [](const Basis& basis1, const Basis& basis2, double omega) {
            return erfc_coulomb_2c(basis1, basis2, omega);
        },
        "basis1"_a, "basis2"_a, "omega"_a);
}

} // namespace forte2
