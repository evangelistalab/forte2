#pragma once

#include <sstream>
#include <string>

namespace forte2 {
/// @brief Print the content of a container to the output file
template <typename Container> std::string container_to_string(const Container& c) {
    if (c.empty()) {
        return "[]";
    }
    std::stringstream ss;
    ss << "[";
    for (const auto& item : c) {
        ss << item << ",";
    }
    std::string result = ss.str();
    if (!result.empty() and result.back() == ',') {
        result.pop_back(); // Remove the trailing space
    }
    result += "]";
    return result;
}
} // namespace forte2