#include <libint2.hpp>
#include <libint2/shell.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include "ints/basis.h"
#include "ints/fock_builder.h"
#include "ints/nuclear_repulsion.h"
#include "ints/one_electron.h"
#include "ints/two_electron.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_integrals_api(nb::module_& m) {
    nb::module_ sub_m = m.def_submodule("ints", "Integrals submodule");

    nb::class_<libint2::Shell>(sub_m, "Shell")
        .def(nb::init<>())
        .def(
            "__init__",
            [](libint2::Shell* t, int l, const std::vector<double>& exponents,
               const std::vector<double>& coeffs, const std::vector<double>& origin, bool is_pure) {
                auto l2_exponents = libint2::svector<double>(exponents.begin(), exponents.end());
                auto l2_coeffs = libint2::svector<double>(coeffs.begin(), coeffs.end());
                new (t) libint2::Shell{
                    l2_exponents, {{l, is_pure, l2_coeffs}}, {origin[0], origin[1], origin[2]}};
            },
            "l"_a, "exponents"_a, "coeffs"_a, "centers"_a, "is_pure"_a = true)
        .def_prop_ro("size", [](libint2::Shell& s) { return s.size(); })
        .def_prop_ro("ncontr", [](libint2::Shell& s) { return s.ncontr(); })
        .def_prop_ro("nprim", [](libint2::Shell& s) { return s.nprim(); });

    nb::class_<Basis>(sub_m, "Basis")
        .def(nb::init<>())
        .def("add", &Basis::add, "shell"_a)
        .def("__getitem__", &Basis::operator[], "i"_a)
        .def_prop_ro("shell_first_and_size", &Basis::shell_first_and_size)
        .def_prop_ro("size", &Basis::size)
        .def_prop_ro("max_l", &Basis::max_l)
        .def_prop_ro("max_nprim", &Basis::max_nprim)
        .def_prop_ro("nprim", &Basis::max_nprim)
        .def_prop_ro("nshells", &Basis::nshells);

    nb::class_<FockBuilder>(sub_m, "FockBuilder")
        .def(nb::init<const Basis&, const Basis&>(), "basis"_a, "auxiliary_basis"_a = Basis())
        .def("build", &FockBuilder::build);

    sub_m.def(
        "nuclear_repulsion",
        [](std::vector<std::pair<double, std::array<double, 3>>> charges) {
            return nuclear_repulsion(charges);
        },
        "charges"_a);

    sub_m.def(
        "overlap", [](const Basis& basis1, const Basis& basis2) { return overlap(basis1, basis2); },
        "basis1"_a, "basis2"_a);
    sub_m.def(
        "overlap", [](const Basis& basis) { return overlap(basis, basis); }, "basis"_a);

    sub_m.def(
        "kinetic", [](const Basis& basis1, const Basis& basis2) { return kinetic(basis1, basis2); },
        "basis1"_a, "basis2"_a);
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
}
} // namespace forte2