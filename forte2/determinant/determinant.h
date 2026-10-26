#pragma once

#include "helpers/unordered_dense.h"

#include <type_traits>

#include "determinant/string.hpp"
#include "determinant/determinant.hpp"
#include "determinant/spinor_determinant.hpp"
#include "determinant/configuration.hpp"

namespace forte2 {

size_t constexpr Norb = 64;
size_t constexpr Norb2 = 2 * Norb;

using String = StringImpl<Norb>;
using Determinant = DeterminantImpl<Norb2>;
using SpinorDeterminant = SpinorDeterminantImpl<Norb2>;
using Configuration = ConfigurationImpl<Norb2>;

static_assert(!std::is_constructible_v<Determinant, SpinorDeterminant>);
static_assert(!std::is_constructible_v<SpinorDeterminant, Determinant>);

using det_vec = std::vector<Determinant>;

template <typename T = double>
using det_hash = ankerl::unordered_dense::map<Determinant, T, Determinant::Hash>;
} // namespace forte2
