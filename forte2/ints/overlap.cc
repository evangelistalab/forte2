#include <libint2.hpp>

#include "ints/basis.h"

namespace forte2 {

void overlap(const Basis& basis1, const Basis& basis2) {
    std::cout << "Computing overlap integrals..." << std::endl;
    // time the computation
    auto start = std::chrono::high_resolution_clock::now();

    libint2::initialize();

    auto nb1 = basis1.size();
    auto nb2 = basis2.size();
    auto vec = std::make_shared<std::vector<double>>(nb1 * nb2, 0.0);
    auto data = vec->data();

    // max # of primitives in shells this engine will accept
    auto max_nprim = std::max(basis1.max_nprim(), basis2.max_nprim());
    // max angular momentum of shells this engine will accept
    auto max_l = std::max(basis1.max_l(), basis2.max_l());

    libint2::Engine engine(libint2::Operator::overlap, max_nprim, max_l);

    const auto& buff = engine.results(); // get the buffer for the results

    size_t offset1 = 0;
    for (size_t s1 = 0, max_s1 = basis1.nshells(); s1 != max_s1; ++s1) {
        const auto& shell1 = basis1[s1];
        auto n1 = shell1.size(); // number of basis functions in first shell
        size_t offset2 = 0;
        for (size_t s2 = 0, max_s2 = basis2.nshells(); s2 != max_s2; ++s2) {
            const auto& shell2 = basis2[s2];
            auto n2 = shell2.size();        // number of basis functions in second shell
            engine.compute(shell1, shell2); // compute the integrals
            auto ints_shellset = buff[0];   // location of the computed integrals
            if (ints_shellset != nullptr) {
                // integrals are packed into ints_shellset in row-major (C) form
                // this iterates over integrals in this order
                for (auto f1 = 0; f1 != n1; ++f1)
                    for (auto f2 = 0; f2 != n2; ++f2) {
                        const auto i1 = offset1 + f1;
                        const auto i2 = offset2 + f2;
                        data[i1 * nb2 + i2] = double(ints_shellset[f1 * n2 + f2]);
                    }
            }
            offset2 += n2;
        }
        offset1 += n1;
    }

    libint2::finalize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time taken: " << duration.count() << " µs" << std::endl;
}
} // namespace forte2