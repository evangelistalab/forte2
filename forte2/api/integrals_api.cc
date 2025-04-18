#include <libint2.hpp>
#include <libint2/shell.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
// #include <nanobind/ndarray.h>

#include "ints/basis.h"
#include "ints/overlap.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_integrals_api(nb::module_& m) {
    nb::class_<libint2::Shell>(m, "Shell")
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

    nb::class_<Basis>(m, "Basis")
        .def(nb::init<>())
        .def("add", &Basis::add, "shell"_a)
        .def_prop_ro("nshells", &Basis::nshells);

    m.def(
        "overlap", [](const Basis& basis1, const Basis& basis2) { return overlap(basis1, basis2); },
        "basis1"_a, "basis2"_a);
    m.def(
        "overlap", [](const Basis& basis) { return overlap(basis, basis); }, "basis"_a);
}
} // namespace forte2