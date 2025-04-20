#include "ints/two_electron.h"
#include "ints/two_electron_compute.h"

namespace forte2 {

nb::ndarray<nb::numpy, double, nb::ndim<4>> coulomb_4c(const Basis& b1, const Basis& b2,
                                                       const Basis& b3, const Basis& b4) {
    return compute_two_electron_4c_multi<libint2::Operator::coulomb>(b1, b2, b3, b4);
}

nb::ndarray<nb::numpy, double, nb::ndim<3>> coulomb_3c(const Basis& b1, const Basis& b2,
                                                       const Basis& b3) {
    return compute_two_electron_3c_multi<libint2::Operator::coulomb>(b1, b2, b3);
}

nb::ndarray<nb::numpy, double, nb::ndim<2>> coulomb_2c(const Basis& b1, const Basis& b2) {
    return compute_two_electron_2c_multi<libint2::Operator::coulomb>(b1, b2);
}

} // namespace forte2