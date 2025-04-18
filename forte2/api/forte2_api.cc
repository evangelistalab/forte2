#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

int add(int a, int b) { return a + b; }

#include "integrals_api.h"

NB_MODULE(_forte2, m) {
    m.def("add", &add, "a"_a, "b"_a, "This is a test function.");
    forte2::export_integrals_api(m);
}