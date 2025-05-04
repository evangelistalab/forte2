#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_determinant_api(nb::module_& m);
void export_integrals_api(nb::module_& m);
void export_sparsestate_api(nb::module_& m);

NB_MODULE(_forte2, m) {
    export_integrals_api(m);
    export_determinant_api(m);
    export_sparsestate_api(m);
}
} // namespace forte2