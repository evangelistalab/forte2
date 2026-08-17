#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_cpp_helpers_api(nb::module_& m);
void export_ints_api(nb::module_& m);
void export_ci_helpers_api(nb::module_& m);
void export_det_api(nb::module_& m);
void export_dsrg_utils_api(nb::module_& m);
void export_sparse_ops_api(nb::module_& m);
void export_rdms_api(nb::module_& m);

NB_MODULE(lib, m) {
    m.doc() = "Bindings for C++ functions and classes for forte2";

    export_cpp_helpers_api(m);
    export_ints_api(m);
    export_ci_helpers_api(m);
    export_det_api(m);
    export_dsrg_utils_api(m);
    export_sparse_ops_api(m);
    export_rdms_api(m);
    m.attr("__version__") = "2026.6.4";
    m.attr("__author__") = "Forte2 Developers";
}
} // namespace forte2
