#pragma once

#include <unordered_map>

#include "determinant.hpp"
#include "configuration.hpp"

namespace forte2 {

size_t constexpr Norb = 64;
size_t constexpr Norb2 = 2 * Norb;

using String = BitArray<Norb>;
using Determinant = DeterminantImpl<Norb2>;
using Configuration = ConfigurationImpl<Norb2>;

using det_vec = std::vector<Determinant>;
template <typename T = double> using det_hash = std::unordered_map<Determinant, T>;
using det_hash_it = std::unordered_map<Determinant, double>::iterator;
} // namespace forte2
