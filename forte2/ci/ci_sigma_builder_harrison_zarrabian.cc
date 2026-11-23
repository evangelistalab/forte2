#include "ci_sigma_builder.h"
#include "ci_hole_contractions.h"

namespace forte2 {

void CISigmaBuilder::H1_hz(std::span<double> basis, std::span<double> sigma, Spin spin,
                           std::span<double> h) const {
    contract_same_spin_sigma<1, double>(
        lists_, spin, basis, sigma, TR, TL, *orbital_rows_, h,
        [this](size_t rows, size_t columns) { return get_Kblock_spans(rows, columns); });
}

void CISigmaBuilder::H2_hz_same_spin(std::span<double> basis, std::span<double> sigma,
                                     Spin spin) const {
    contract_same_spin_sigma<2, double>(
        lists_, spin, basis, sigma, TR, TL, *same_spin_pairs_, v_pr_qs_a,
        [this](size_t rows, size_t columns) { return get_Kblock_spans(rows, columns); });
}

void CISigmaBuilder::H2_hz_opposite_spin(std::span<double> basis, std::span<double> sigma) const {
    const auto norb = lists_.norb();
    const auto& alpha = lists_.strings(Spin::Alpha, 1);
    const auto& beta = lists_.strings(Spin::Beta, 1);
    std::vector<LocalStringAddress> work(lists_.ngas_spaces());

    for (const auto& [classes, links] : mixed_hole_connections<1, 1>(lists_)) {
        const auto [ka, kb] = classes;
        const auto& block = opposite_spin_pairs_.for_intermediate(lists_, ka, kb);
        std::vector<size_t> rows;
        for (const auto& link : links)
            for (const auto& [p] : link.alpha.orbitals())
                for (const auto& [q] : link.beta.orbitals())
                    rows.push_back(p * norb + q);
        const CIOrbitalMatrixBlock<double> matrix(opposite_spin_pairs_, block, v_pr_qs,
                                                  std::move(rows));

        const auto nrows = matrix.size();
        const auto beta_size = beta.size(kb);
        const auto columns = checked_ci_product(alpha.size(ka), beta_size);
        const auto [D, E, chunk] = get_Kblock_spans(nrows, columns);
        for (size_t start = 0; start < columns;) {
            const size_t width = std::min(chunk, columns - start);
            std::fill_n(D.begin(), nrows * width, 0.0);
            for (const auto& link : links) {
                const auto& determinant = link.determinant;
                for_each_mixed_hole_substitution(
                    link.alpha, link.beta, beta_size, start, width, work,
                    [&](size_t col, auto p, auto q, int sign, size_t a, size_t b) {
                        const auto row = matrix.row(p[0] * norb + q[0]);
                        const auto idx = determinant.offset + a * determinant.beta_size + b;
                        D[row * width + col] = sign * basis[idx];
                    });
            }

            matrix_product('N', 'N', nrows, width, nrows, 1.0, matrix.data(), nrows, D.data(),
                           width, 0.0, E.data(), width);

            for (const auto& link : links) {
                const auto& determinant = link.determinant;
                for_each_mixed_hole_substitution(
                    link.alpha, link.beta, beta_size, start, width, work,
                    [&](size_t col, auto p, auto q, int sign, size_t a, size_t b) {
                        const auto row = matrix.row(p[0] * norb + q[0]);
                        const auto idx = determinant.offset + a * determinant.beta_size + b;
                        sigma[idx] += sign * E[row * width + col];
                    });
            }
            start += width;
        }
    }
}

} // namespace forte2
