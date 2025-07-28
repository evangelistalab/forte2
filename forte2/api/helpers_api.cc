#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>

#include "helpers/indexing.hpp"
#include "helpers/np_matrix_functions.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_helpers_api(nb::module_& m) {
    nb::module_ sub_m = m.def_submodule("cpp_helpers", "Helpers submodule");

    sub_m.def("pair_index_geq", &pair_index_geq<size_t>);
    sub_m.def("pair_index_gt", &pair_index_gt<size_t>);
    sub_m.def("inv_pair_index_gt", &inv_pair_index_gt<size_t>);
    sub_m.def("triplet_index_gt", &triplet_index_gt<size_t>);
    sub_m.def("triplet_index_aab", &triplet_index_aab<size_t>);
    sub_m.def("triplet_index_abb", &triplet_index_abb<size_t>);
    sub_m.def("packed_tensor4_to_tensor4", &matrix::packed_tensor4_to_tensor4, "m"_a,
              "Expand a packed 4D tensor stored as a 2D matrix into a full 4D tensor");
}
} // namespace forte2