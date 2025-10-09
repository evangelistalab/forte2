#pragma once

#include <cmath>

#define MACHEPS 1e-9
#define TAYLOR_THRES 1e-3

namespace forte2 {

double taylor_exp(double z) {
    int n = static_cast<int>(0.5 * (15.0 / TAYLOR_THRES + 1)) + 1;
    if (n > 0) {
        double value = z;
        double tmp = z;
        for (int x = 0; x < n - 1; x++) {
            tmp *= -1.0 * z * z / (x + 2);
            value += tmp;
        }
        return value;
    } else {
        return 0.0;
    }
}

double regularized_denominator(double x, double s) {
    double z = std::sqrt(s) * x;
    if (fabs(z) <= MACHEPS) {
        return taylor_exp(z) * std::sqrt(s);
    } else {
        return (1. - std::exp(-s * x * x)) / x;
    }
}
} // namespace forte2