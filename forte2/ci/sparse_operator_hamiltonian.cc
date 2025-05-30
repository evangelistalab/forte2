#include "sparse_operator_hamiltonian.h"

namespace forte2 {

SparseOperator sparse_operator_hamiltonian(double scalar_energy, np_matrix one_electron_integrals,
                                           np_tensor4 two_electron_integrals,
                                           double screen_thresh) {
    SparseOperator H;
    size_t nmo = one_electron_integrals.shape(0);
    if (one_electron_integrals.shape(1) != nmo || two_electron_integrals.shape(0) != nmo ||
        two_electron_integrals.shape(1) != nmo || two_electron_integrals.shape(2) != nmo ||
        two_electron_integrals.shape(3) != nmo) {
        throw std::runtime_error("One-electron and two-electron integrals must be square matrices "
                                 "of the same size.");
    }
    auto oei_view = one_electron_integrals.view();
    auto tei_view = two_electron_integrals.view();

    H.add_term_from_str("[]", scalar_energy);
    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = 0; q < nmo; q++) {
            if (auto hpq = oei_view(p, q); std::fabs(hpq) > screen_thresh) {
                H.add(SQOperatorString({p}, {}, {q}, {}), hpq);
                H.add(SQOperatorString({}, {p}, {}, {q}), hpq);
            }
        }
    }
    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = p + 1; q < nmo; q++) {
            for (size_t r = 0; r < nmo; r++) {
                for (size_t s = r + 1; s < nmo; s++) {
                    if (auto vpqrs = tei_view(p, q, r, s) - tei_view(p, q, s, r);
                        std::fabs(vpqrs) > screen_thresh) {
                        H.add(SQOperatorString({p, q}, {}, {s, r}, {}), vpqrs);
                        H.add(SQOperatorString({}, {p, q}, {}, {s, r}), vpqrs);
                    }
                }
            }
        }
    }
    for (size_t p = 0; p < nmo; p++) {
        for (size_t q = 0; q < nmo; q++) {
            for (size_t r = 0; r < nmo; r++) {
                for (size_t s = 0; s < nmo; s++) {
                    if (auto vpqrs = tei_view(p, q, r, s); std::fabs(vpqrs) > screen_thresh) {
                        H.add(SQOperatorString({p}, {q}, {r}, {s}), vpqrs);
                    }
                }
            }
        }
    }
    return H;
}
} // namespace forte2
