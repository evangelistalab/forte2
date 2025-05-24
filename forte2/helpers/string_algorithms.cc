#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

#include "string_algorithms.h"

namespace forte2 {

std::vector<std::string> split_string(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos) {
        if (pos != prev) { // avoid repetitions
            strings.push_back(str.substr(prev, pos - prev));
        }
        prev = pos + 1;
    }
    // To get the last substring (or only, if delimiter is not found)
    if (str.substr(prev).size() > 0)
        strings.push_back(str.substr(prev));

    return strings;
}

void to_upper_string(std::string& s) { std::transform(s.begin(), s.end(), s.begin(), ::toupper); }

std::string upper_string(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}

void to_lower_string(std::string& s) { std::transform(s.begin(), s.end(), s.begin(), ::tolower); }

std::string lower_string(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

std::string join(const std::vector<std::string>& vec_str, const std::string& sep) {
    if (vec_str.size() == 0)
        return std::string();

    std::string ss;

    std::for_each(vec_str.begin(), vec_str.end() - 1, [&](const std::string& s) { ss += s + sep; });
    ss += vec_str.back();

    return ss;
}

std::vector<std::string>::const_iterator
find_case_insensitive(const std::string& str, const std::vector<std::string>& vec) {
    auto ret = std::find_if(vec.cbegin(), vec.cend(), [&str](const std::string& s) {
        if (s.size() != str.size())
            return false;
        return std::equal(s.cbegin(), s.cend(), str.cbegin(), str.cend(),
                          [](auto c1, auto c2) { return std::toupper(c1) == std::toupper(c2); });
    });
    return ret;
}
} // namespace forte2
