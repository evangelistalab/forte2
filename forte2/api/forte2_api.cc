#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_determinant_api(nb::module_& m);
void export_determinant_helpers_api(nb::module_& m);
void export_integrals_api(nb::module_& m);
void export_sparsestate_api(nb::module_& m);
void export_sparseoperator_api(nb::module_& m);

NB_MODULE(_forte2, m) {
    export_integrals_api(m);
    export_determinant_api(m);
    export_determinant_helpers_api(m);
    export_sparsestate_api(m);
    export_sparseoperator_api(m);
    m.attr("__version__") = "0.1.0";
    m.attr("__author__") = "Forte2 Developers";
}
} // namespace forte2