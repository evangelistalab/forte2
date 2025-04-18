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

void libint2_example() {
    libint2::initialize();

    //                       Exp     L  Cart.  Coeff       (x0,y0,z0)
    auto s = libint2::Shell{{1.0}, {{0, false, {1.0}}}, {{0.0, 0.0, 0.0}}};

    libint2::Engine engine(libint2::Operator::kinetic, // will compute overlap ints
                           3, // max # of primitives in shells this engine will accept
                           1  // max angular momentum of shells this engine will accept
    );

    const auto& buff = engine.results(); // get the buffer for the results

    // <s | K | s> = -1/2 * <s | del^2 | s>
    engine.compute(s, s); // compute the integrals

    // print the results
    std::cout << "Overlap integrals:" << std::endl;
    auto ints_shellset = buff[0]; // location of the computed integrals
    if (ints_shellset != nullptr) {
        auto n1 = s.size(); // number of basis functions in first shell
        auto n2 = s.size(); // number of basis functions in second shell

        // integrals are packed into ints_shellset in row-major (C) form
        // this iterates over integrals in this order
        for (auto f1 = 0; f1 != n1; ++f1)
            for (auto f2 = 0; f2 != n2; ++f2)
                std::cout << "  " << f1 << " " << f2 << " " << double(ints_shellset[f1 * n2 + f2])
                          << std::endl;
    }

    libint2::finalize();
}

void export_integrals_api(nb::module_& m) {
    m.def("libint2_example", &libint2_example, "This is a test function.");

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