#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "helpers/determinant_helpers.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_determinant_helpers_api(nb::module_& m) {
    m.def(
        "hilbert_space",
        [](size_t nmo, size_t na, size_t nb, size_t nirrep, std::vector<int> mo_symmetry,
           int symmetry) { return make_hilbert_space(nmo, na, nb, nirrep, mo_symmetry, symmetry); },
        "nmo"_a, "na"_a, "nb"_a, "nirrep"_a = 1, "mo_symmetry"_a = std::vector<int>(),
        "symmetry"_a = 0,
        "Generate the Hilbert space for a given number of electrons and orbitals."
        "If information about the symmetry of the MOs is not provided, it assumes that all MOs "
        "have symmetry 0.");
    m.def(
        "hilbert_space",
        [](size_t nmo, size_t na, size_t nb, Determinant ref, int truncation, size_t nirrep,
           std::vector<int> mo_symmetry, int symmetry) {
            return make_hilbert_space(nmo, na, nb, ref, truncation, nirrep, mo_symmetry, symmetry);
        },
        "nmo"_a, "na"_a, "nb"_a, "ref"_a, "truncation"_a, "nirrep"_a = 1,
        "mo_symmetry"_a = std::vector<int>(), "symmetry"_a = 0,
        "Generate the Hilbert space for a given number of electrons, orbitals, and the truncation "
        "level."
        "If information about the symmetry of the MOs is not provided, it assumes that all MOs "
        "have symmetry 0."
        "A reference determinant must be provided to establish the excitation rank.");
}
} // namespace forte2