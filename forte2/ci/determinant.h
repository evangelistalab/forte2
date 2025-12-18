#pragma once

#include "helpers/unordered_dense.h"

#include "determinant.hpp"
#include "configuration.hpp"

namespace forte2 {

size_t constexpr Norb = 64;
size_t constexpr Norb2 = 2 * Norb;

using String = BitArray<Norb>;
using Determinant = DeterminantImpl<Norb2>;
using Configuration = ConfigurationImpl<Norb2>;

using det_vec = std::vector<Determinant>;

template <typename T = double>
using det_hash = ankerl::unordered_dense::map<Determinant, T, Determinant::Hash>;
} // namespace forte2
