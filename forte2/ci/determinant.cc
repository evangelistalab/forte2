#include "determinant.h"

namespace forte2 {

std::string str(const Determinant& d, int n) {
    std::string s;
    s += "|";
    for (int p = 0; p < n; ++p) {
        if (d.get_a(p) and d.get_b(p)) {
            s += "2";
        } else if (d.get_a(p) and not d.get_b(p)) {
            s += "+";
        } else if (not d.get_a(p) and d.get_b(p)) {
            s += "-";
        } else {
            s += "0";
        }
    }
    s += ">";
    return s;
}

} // namespace forte2
