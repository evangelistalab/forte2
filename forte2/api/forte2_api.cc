#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_helpers_api(nb::module_& m);
void export_ci_strings_api(nb::module_& m);
void export_ci_sigma_builder_api(nb::module_& m);
void export_rel_ci_sigma_builder_api(nb::module_& m);
void export_ci_spin_adapter_api(nb::module_& m);
void export_slater_rules_api(nb::module_& m);
void export_rel_slater_rules_api(nb::module_& m);
void export_determinant_api(nb::module_& m);
void export_determinant_helpers_api(nb::module_& m);
void export_configuration_api(nb::module_& m);
void export_integrals_api(nb::module_& m);
void export_logging_api(nb::module_& m);
void export_sparse_state_api(nb::module_& m);
void export_sparse_operator_api(nb::module_& m);
void export_sparse_operator_list_api(nb::module_& m);
void export_sparse_exp_api(nb::module_& m);
void export_sparse_fact_exp_api(nb::module_& m);
void export_sq_operator_string_api(nb::module_& m);

NB_MODULE(_forte2, m) {
    export_helpers_api(m);
    export_integrals_api(m);
    export_ci_strings_api(m);
    export_ci_sigma_builder_api(m);
    export_rel_ci_sigma_builder_api(m);
    export_ci_spin_adapter_api(m);
    export_determinant_api(m);
    export_determinant_helpers_api(m);
    export_configuration_api(m);
    export_logging_api(m);
    export_slater_rules_api(m);
    export_rel_slater_rules_api(m);
    export_sparse_state_api(m);
    export_sparse_operator_api(m);
    export_sparse_operator_list_api(m);
    export_sparse_exp_api(m);
    export_sparse_fact_exp_api(m);
    export_sq_operator_string_api(m);
    m.attr("__version__") = "0.2.0";
    m.attr("__author__") = "Forte2 Developers";
}
} // namespace forte2