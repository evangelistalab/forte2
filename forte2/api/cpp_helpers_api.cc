#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>

#include "helpers/ndarray.h"
#include "helpers/indexing.hpp"
#include "helpers/np_matrix_functions.h"
#include "helpers/logger.h"
#include "helpers/parallel.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace forte2 {

namespace {
void export_indexing_api(nb::module_& m);
void export_logging_api(nb::module_& m);
void export_parallel_api(nb::module_& m);
template <typename T> void export_packing(nb::module_& m);
} // namespace

void export_cpp_helpers_api(nb::module_& m) {
    nb::module_ sub_m = m.def_submodule("cpp_helpers", "C++ utilities");

    export_indexing_api(sub_m);

    export_logging_api(sub_m);

    export_parallel_api(sub_m);
}

namespace {
void export_indexing_api(nb::module_& sub_m) {
    sub_m.def("pair_index_geq", &pair_index_geq<size_t>);
    sub_m.def("pair_index_gt", &pair_index_gt<size_t>);
    sub_m.def("inv_pair_index_gt", &inv_pair_index_gt<size_t>);
    sub_m.def("triplet_index_gt", &triplet_index_gt<size_t>);
    sub_m.def("triplet_index_aab", &triplet_index_aab<size_t>);
    sub_m.def("triplet_index_abb", &triplet_index_abb<size_t>);
    sub_m.def(
        "packed_tensor4_to_tensor4",
        [](np_matrix m) { return matrix::packed_tensor4_to_tensor4<double>(m); }, "m"_a,
        "Expand a packed 4D tensor stored as a 2D matrix into a full 4D tensor");
    sub_m.def(
        "packed_tensor4_to_tensor4",
        [](np_matrix_complex m) {
            return matrix::packed_tensor4_to_tensor4<std::complex<double>>(m);
        },
        "m"_a, "Expand a packed 4D tensor stored as a 2D matrix into a full 4D tensor");
    export_packing<double>(sub_m);
    export_packing<std::complex<double>>(sub_m);
}

template <typename T> void export_packing(nb::module_& sub_m) {
    using matrix_t = nb::ndarray<nb::numpy, T, nb::ndim<2>>;
    using tensor4_t = nb::ndarray<nb::numpy, T, nb::ndim<4>>;
    using tensor6_t = nb::ndarray<nb::numpy, T, nb::ndim<6>>;
    sub_m.def(
        "tensor4_to_packed_tensor4",
        [](tensor4_t t) { return matrix::tensor4_to_packed_tensor4<T>(t); }, "t"_a,
        "Pack a 4D tensor antisymmetric in its first and last index pairs into a 2D matrix "
        "over the pairs p > q and r > s");
    sub_m.def(
        "packed_tensor6_to_tensor6",
        [](matrix_t m, size_t n) { return matrix::packed_tensor6_to_tensor6<T>(m, n); }, "m"_a,
        "n"_a,
        "Expand a 6D tensor packed over the triplets p > q > r and s > t > u into a full 6D "
        "tensor");
    sub_m.def(
        "tensor6_to_packed_tensor6",
        [](tensor6_t t) { return matrix::tensor6_to_packed_tensor6<T>(t); }, "t"_a,
        "Pack a 6D tensor antisymmetric in its first and last index triplets into a 2D matrix "
        "over the triplets p > q > r and s > t > u");
    sub_m.def(
        "packed_aab_tensor6_to_tensor6",
        [](tensor4_t m) { return matrix::packed_aab_tensor6_to_tensor6<T>(m); }, "m"_a,
        "Expand an alpha-alpha-beta 6D tensor packed over the pairs p > q and s > t into a full "
        "6D tensor");
    sub_m.def(
        "tensor6_to_packed_aab_tensor6",
        [](tensor6_t t) { return matrix::tensor6_to_packed_aab_tensor6<T>(t); }, "t"_a,
        "Pack an alpha-alpha-beta 6D tensor over the pairs p > q and s > t, giving shape "
        "(npair, n, npair, n)");
    sub_m.def(
        "packed_abb_tensor6_to_tensor6",
        [](tensor4_t m) { return matrix::packed_abb_tensor6_to_tensor6<T>(m); }, "m"_a,
        "Expand an alpha-beta-beta 6D tensor packed over the pairs q > r and t > u into a full "
        "6D tensor");
    sub_m.def(
        "tensor6_to_packed_abb_tensor6",
        [](tensor6_t t) { return matrix::tensor6_to_packed_abb_tensor6<T>(t); }, "t"_a,
        "Pack an alpha-beta-beta 6D tensor over the pairs q > r and t > u, giving shape "
        "(n, npair, n, npair)");
}

void export_logging_api(nb::module_& sub_m) {
    sub_m.def(
        "set_log_level",
        [](int level) { Logger::getInstance().setLevel(static_cast<Logger::Level>(level)); },
        "Set the logging verbosity level (0=NONE, 1=ERROR, 2=WARNING, 3=INFO, 4=DEBUG)");
    sub_m.def(
        "get_log_level", []() { return static_cast<int>(Logger::getInstance().getLevel()); },
        "Get the current logging verbosity level");
}

void export_parallel_api(nb::module_& sub_m) { sub_m.def("get_num_threads", &get_num_threads); }
} // namespace
} // namespace forte2
