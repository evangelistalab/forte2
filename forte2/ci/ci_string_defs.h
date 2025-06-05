#pragma once

#include <map>
#include <vector>
#include <utility>

#include "determinant.h"

namespace forte2 {

/// A structure to store how the string J is connected to the string I and the corresponding sign
/// I -> sign J
/// The uint32_t will hold up to 4,294,967,296 elements (that should be enough)
struct StringSubstitution {
    const double sign;
    const uint32_t I;
    const uint32_t J;
    StringSubstitution(const double& sign_, const uint32_t& I_, const uint32_t& J_)
        : sign(sign_), I(I_), J(J_) {}
};

/// 1-hole string substitution
struct H1StringSubstitution {
    const int16_t sign;
    const int16_t p;
    const uint32_t J;
    constexpr H1StringSubstitution(int16_t sign_, int16_t p_, uint32_t J_) noexcept
        : sign(sign_), p(p_), J(J_) {}
};

/// 2-hole string substitution
struct H2StringSubstitution {
    const int16_t sign;
    const uint8_t p;
    const uint8_t q;
    size_t J;
    H2StringSubstitution(int16_t sign_, uint8_t p_, uint8_t q_, size_t J_)
        : sign(sign_), p(p_), q(q_), J(J_) {}
};

/// 3-hole string substitution
struct H3StringSubstitution {
    const int16_t sign;
    const int16_t p;
    const int16_t q;
    const int16_t r;
    const size_t J;
    H3StringSubstitution(int16_t sign_, int16_t p_, int16_t q_, int16_t r_, size_t J_)
        : sign(sign_), p(p_), q(q_), r(r_), J(J_) {}
};

using StringList = std::vector<std::vector<String>>;

/// Maps the integers (p,q,h) to list of strings connected by a^{+}_p a_q, where the string
/// I belongs to the irrep h
using VOList = std::map<std::tuple<size_t, size_t, int>, std::vector<StringSubstitution>>;

/// Maps the integers (class_I, class_J) to a map of orbital indices (p,q) and the corresponding
/// list of strings connected by a^{+}_p a_q, where the string I belongs to class_I and J belongs to
/// class_J
using VOListElement = std::map<std::tuple<int, int>, std::vector<StringSubstitution>>;
using VOListMap = std::map<std::pair<int, int>, VOListElement>;

using HListKey = std::tuple<int, size_t, int>;

/// Maps the integers (h_J, add_J, h_I) to list of strings connected by a_p, where the string
/// I belongs to the irrep h_I and J belongs to the irrep h_J and add_J is the address of J
using H1List = std::map<HListKey, std::vector<H1StringSubstitution>>;

/// Maps the integers (h_J, add_J, h_I) to list of strings connected by a_p a_q, where the string
/// I belongs to the irrep h_I and J belongs to the irrep h_J and add_J is the address of J
using H2List = std::map<HListKey, std::vector<H2StringSubstitution>>;

/// Maps the integers (h_J, add_J, h_I) to list of strings connected by a_p a_q a_r, where the
/// string I belongs to the irrep h_I and J belongs to the irrep h_J and add_J is the address of J
using H3List = std::map<HListKey, std::vector<H3StringSubstitution>>;

} // namespace forte2
