#pragma once

#include <complex>
#include <sstream>
#include <string>
#include <vector>

namespace forte2 {

/// Split a string into a vector of strings using a delimiter
std::vector<std::string> split_string(const std::string& str, const std::string& delimiter);

/// Convert all lower case letters in a string to the upper case
void to_upper_string(std::string& s);
std::string upper_string(std::string s);

/// Convert all lower case letters in a string to the upper case
void to_lower_string(std::string& s);
std::string lower_string(std::string s);

/// Join a vector of strings using a separator
std::string join(const std::vector<std::string>& vec_str, const std::string& sep = ",");

/// Convert a number to string with a given precision
template <typename T> std::string to_string_with_precision(const T val, const int n = 6) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << val;
    auto s = out.str();
    s.erase(s.find_last_not_of('0') + 1, std::string::npos);
    if (s.back() == '.') {
        s.pop_back();
    }
    return s;
}

template <typename T> inline std::string to_string_latex(T value) {
    if (value == T(-1)) {
        return "-";
    }
    if (value == T(1)) {
        return "";
    }
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return "(" + std::to_string(std::real(value)) + " + " + std::to_string(std::imag(value)) +
               " i)";
    }
    if constexpr (std::is_same_v<T, double>) {
        return std::to_string(value);
    }
}

/// Find a string in a vector of strings in a case insensitive
std::vector<std::string>::const_iterator find_case_insensitive(const std::string& str,
                                                               const std::vector<std::string>& vec);

} // namespace forte2