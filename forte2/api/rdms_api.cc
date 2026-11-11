#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/make_iterator.h>

#include "sparse/sparse_rdms.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

void export_rdms_api(nb::module_& m) {
    nb::module_ sub_m = m.def_submodule("rdms", "Reduced density matrices between SparseStates");

    sub_m.def("compute_a_1rdm", &compute_a_1rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the alpha 1-RDM between two SparseStates");
    sub_m.def("compute_b_1rdm", &compute_b_1rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the beta 1-RDM between two SparseStates");
    sub_m.def("compute_aa_2rdm", &compute_aa_2rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the alpha-alpha 2-RDM between two SparseStates");
    sub_m.def("compute_ab_2rdm", &compute_ab_2rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the alpha-beta 2-RDM between two SparseStates");
    sub_m.def("compute_bb_2rdm", &compute_bb_2rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the beta-beta 2-RDM between two SparseStates");
    sub_m.def("compute_aaa_3rdm", &compute_aaa_3rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the alpha-alpha-alpha 3-RDM between two SparseStates");
    sub_m.def("compute_aab_3rdm", &compute_aab_3rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the alpha-alpha-beta 3-RDM between two SparseStates");
    sub_m.def("compute_abb_3rdm", &compute_abb_3rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the alpha-beta-beta 3-RDM between two SparseStates");
    sub_m.def("compute_bbb_3rdm", &compute_bbb_3rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the beta-beta-beta 3-RDM between two SparseStates");
    sub_m.def("compute_aaaa_4rdm", &compute_aaaa_4rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the alpha-alpha-alpha-alpha 4-RDM between two SparseStates");
    sub_m.def("compute_aaab_4rdm", &compute_aaab_4rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the alpha-alpha-alpha-beta 4-RDM between two SparseStates");
    sub_m.def("compute_aabb_4rdm", &compute_aabb_4rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the alpha-alpha-beta-beta 4-RDM between two SparseStates");
    sub_m.def("compute_abbb_4rdm", &compute_abbb_4rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the alpha-beta-beta-beta 4-RDM between two SparseStates");
    sub_m.def("compute_bbbb_4rdm", &compute_bbbb_4rdm, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the beta-beta-beta-beta 4-RDM between two SparseStates");
    sub_m.def("compute_1rdm_2c", &compute_1rdm_2c, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the two-component 1-RDM between two SparseStates (conjugates the bra)");
    sub_m.def("compute_2rdm_2c", &compute_2rdm_2c, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the two-component 2-RDM between two SparseStates (conjugates the bra)");
    sub_m.def("compute_3rdm_2c", &compute_3rdm_2c, "state_left"_a, "state_right"_a, "norb"_a,
              "Compute the two-component 3-RDM between two SparseStates (conjugates the bra)");
}

} // namespace forte2
