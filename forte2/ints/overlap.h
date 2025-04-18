#pragma once

namespace nb = nanobind;
#include <nanobind/ndarray.h>

namespace forte2 {
nb::ndarray<nb::numpy, double, nb::ndim<2>> overlap(const Basis& basis1, const Basis& basis2);
}