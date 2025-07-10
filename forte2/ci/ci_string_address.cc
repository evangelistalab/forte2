#include "ci_string_address.h"
#include "helpers/sum.hpp"

namespace forte2 {

StringAddress::StringAddress(const std::vector<int>& gas_size, int ne,
                             const std::vector<std::vector<String>>& strings)
    : nclasses_(strings.size()), nstr_(0), strpcls_(strings.size(), 0), nones_(ne),
      gas_size_(gas_size) {
    for (int h = 0; h < nclasses_; h++) {
        const auto& strings_h = strings[h];
        for (const auto& s : strings_h) {
            push_back(s, h);
        }
    }
}

void StringAddress::push_back(const String& s, int string_class) {
    size_t add = strpcls_[string_class];
    address_[s] = std::pair(add, string_class);
    strpcls_[string_class] += 1;
    nstr_++;
}

int StringAddress::nones() const { return nones_; }

int StringAddress::nbits() const { return math::sum(gas_size_); }

size_t StringAddress::add(const String& s) const { return address_.at(s).first; }

int StringAddress::sym(const String& s) const { return address_.at(s).second; }

std::unordered_map<String, std::pair<uint32_t, uint32_t>, String::Hash>::const_iterator
StringAddress::find(const String& s) const {
    return address_.find(s);
}

std::unordered_map<String, std::pair<uint32_t, uint32_t>, String::Hash>::const_iterator
StringAddress::end() const {
    return address_.end();
}

const std::pair<uint32_t, uint32_t>& StringAddress::address_and_class(const String& s) const {
    return address_.at(s);
}

int StringAddress::nclasses() const { return nclasses_; }

size_t StringAddress::strpcls(int h) const { return strpcls_[h]; }

} // namespace forte2
