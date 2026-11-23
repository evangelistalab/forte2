#include "ci_string_address.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <limits>
#include <stdexcept>

namespace forte2 {

size_t checked_ci_product(size_t a, size_t b) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
        throw std::overflow_error("CI product dimension exceeds the address range.");
    return a * b;
}

size_t checked_ci_sum(size_t a, size_t b) {
    if (a > std::numeric_limits<size_t>::max() - b)
        throw std::overflow_error("CI address exceeds the address range.");
    return a + b;
}

GASStringBasis::GASStringBasis(int nirrep, const std::vector<std::vector<int>>& orbital_symmetry)
    : nirrep_(nirrep) {
    if (nirrep <= 0 || !std::has_single_bit(static_cast<unsigned>(nirrep)))
        throw std::invalid_argument("CIStrings: nirrep must be a positive power of two.");

    const auto add_space = [&](std::span<const int> labels) {
        if (labels.size() > Norb - norb())
            throw std::invalid_argument("CIStrings: Too many active orbitals for a determinant.");
        auto mask = String::zero();
        for (const auto irrep : labels) {
            if (irrep < 0 || irrep >= nirrep_)
                throw std::invalid_argument(
                    "CIStrings: Orbital irreps must be non-negative and less than nirrep.");
            orbital_prefix_.push_back(mask);
            mask.set_bit(norb(), true);
            irreps_.push_back(irrep);
            orbital_gas_.push_back(sizes_.size());
        }
        sizes_.push_back(static_cast<int>(labels.size()));
        offsets_.push_back(norb());
        masks_.push_back(mask);
        factors_.emplace_back(labels.size() + 1);
    };
    for (const auto& labels : orbital_symmetry)
        add_space(labels);
    if (orbital_symmetry.empty())
        add_space({});
}

void GASStringBasis::prepare(std::span<const occupation_t> occupations) {
    for (const auto& occupation : occupations) {
        for (size_t g = 0; g < ngas(); ++g) {
            const int electrons = occupation[g];
            auto& entry = factors_[g][electrons];
            if (entry)
                continue;

            auto& local = entry.emplace();
            local.sectors.resize(nirrep_);
            const size_t begin = offsets_[g];
            const size_t end = offsets_[g + 1];
            auto string = String::zero();
            for (size_t p = end - electrons; p < end; ++p)
                string.set_bit(p, true);
            const bool track_order = ngas() > 1 && nirrep_ > 1;
            do {
                const auto h = static_cast<int>(string.symmetry(irreps_));
                auto& sector = local.sectors[h];
                const LocalStringAddress address{sector.strings.size(), h};
                local.addresses.emplace(string, address);
                sector.strings.push_back(string);
                if (track_order) {
                    sector.positions.push_back(local.in_order.size());
                    local.in_order.push_back(address);
                }
            } while (std::next_permutation(std::next(string.begin(), begin),
                                           std::next(string.begin(), end)));
        }
    }
}

void GASStringBasis::prepare_substitutions() {
    for (size_t g = 0; g < ngas(); ++g) {
        for (int n = 0; n < sizes_[g]; ++n) {
            auto& lower = factors_[g][n];
            auto& upper = factors_[g][n + 1];
            if (!lower || !upper)
                continue;
            const size_t vacancies = sizes_[g] - n;
            for (auto& sector : upper->sectors)
                sector.annihilations.resize(checked_ci_product(sector.strings.size(), n + 1));
            for (auto& sector : lower->sectors) {
                sector.creations.resize(checked_ci_product(sector.strings.size(), vacancies));
                size_t offset = 0;
                for (size_t rank = 0; rank < sector.strings.size(); ++rank) {
                    const auto& local = sector.strings[rank];
                    for (size_t p = offsets_[g]; p < offsets_[g + 1]; ++p) {
                        if (local.get_bit(p))
                            continue;
                        auto next = local;
                        next.set_bit(p, true);
                        const auto [target, h] = upper->addresses.at(next);
                        sector.creations[offset++] = target;
                        const size_t occupied = local.intersection_count(orbital_prefix_[p]);
                        upper->sectors[h].annihilations[target * (n + 1) + occupied] = rank;
                    }
                }
            }
        }
    }
}

int GASStringBasis::substitute(size_t gas, int electrons, LocalStringAddress& address,
                               size_t orbital, bool create) const {
    const auto& sector = factor(gas, electrons).sectors[address.irrep];
    const auto& local = sector.strings[address.rank];
    if (local.get_bit(orbital) == create)
        return 0;
    const size_t occupied = local.intersection_count(orbital_prefix_[orbital]);
    if (create) {
        const size_t vacancy = orbital - offsets_[gas] - occupied;
        address.rank = sector.creations[address.rank * (sizes_[gas] - electrons) + vacancy];
    } else {
        address.rank = sector.annihilations[address.rank * electrons + occupied];
    }
    address.irrep ^= irreps_[orbital];
    return occupied % 2 == 0 ? 1 : -1;
}

StringAddress::StringAddress(std::shared_ptr<const GASStringBasis> basis,
                             std::vector<occupation_t> occupations)
    : basis_(std::move(basis)), occupations_(std::move(occupations)) {
    const auto ngas = basis_->ngas();
    const int nirrep = basis_->nirrep();
    const size_t nclasses = checked_ci_product(occupations_.size(), nirrep);
    if (nclasses > std::numeric_limits<int>::max())
        throw std::overflow_error("Too many CI string classes.");
    nclasses_ = static_cast<int>(nclasses);

    products_.reserve(occupations_.size());
    for (const auto& occupation : occupations_) {
        occupation_indices_.emplace(occupation, static_cast<int>(products_.size()));
        auto& product = products_.emplace_back();
        product.factors.reserve(ngas);
        for (size_t g = 0; g < ngas; ++g)
            product.factors.push_back(&basis_->factor(g, occupation[g]));

        auto& counts = product.suffix_counts;
        counts.resize(checked_ci_product(ngas + 1, nirrep));
        counts[ngas * nirrep] = 1;
        for (size_t g = ngas; g-- > 0;) {
            for (int h = 0; h < nirrep; ++h) {
                const auto size = product.factors[g]->sectors[h].strings.size();
                for (int tail = 0; tail < nirrep; ++tail) {
                    auto& count = counts[g * nirrep + (h ^ tail)];
                    count = checked_ci_sum(
                        count, checked_ci_product(size, counts[(g + 1) * nirrep + tail]));
                }
            }
        }
        for (int h = 0; h < nirrep; ++h)
            total_size_ = checked_ci_sum(total_size_, counts[h]);
    }
}

const occupation_t& StringAddress::occupation(int string_class) const {
    return occupations_[string_class / basis_->nirrep()];
}

size_t StringAddress::size(int string_class) const {
    return product(string_class).suffix_counts[irrep(string_class)];
}

std::optional<int> StringAddress::find_class(const occupation_t& occupation, int h) const {
    const auto it = occupation_indices_.find(occupation);
    if (it == occupation_indices_.end())
        return std::nullopt;
    return it->second * basis_->nirrep() + h;
}

size_t StringAddress::rank(int string_class,
                           std::span<const LocalStringAddress> coordinates) const {
    const auto& [factors, counts] = product(string_class);
    const int nirrep = basis_->nirrep();
    int h = irrep(string_class);
    size_t address = 0;
    for (size_t g = 0; g < factors.size(); ++g) {
        const auto [index, local_irrep] = coordinates[g];
        const auto* suffix = counts.data() + (g + 1) * nirrep;
        const auto& factor = *factors[g];
        address += index * suffix[h ^ local_irrep];
        if (nirrep > 1 && g + 1 < factors.size()) {
            const auto position = factor.sectors[local_irrep].positions[index];
            for (int other = 0; other < nirrep; ++other) {
                if (other == local_irrep || suffix[h ^ other] == 0)
                    continue;
                const auto& positions = factor.sectors[other].positions;
                const auto before = std::lower_bound(positions.begin(), positions.end(), position);
                address += (before - positions.begin()) * suffix[h ^ other];
            }
        }
        h ^= local_irrep;
    }
    assert(h == 0);
    return address;
}

void StringAddress::unrank(int string_class, size_t address,
                           std::span<LocalStringAddress> coordinates) const {
    const auto& [factors, counts] = product(string_class);
    const int nirrep = basis_->nirrep();
    int h = irrep(string_class);
    assert(address < counts[h]);
    for (size_t g = 0; g < factors.size(); ++g) {
        const auto& factor = *factors[g];
        const auto* suffix = counts.data() + (g + 1) * nirrep;
        if (nirrep == 1 || g + 1 == factors.size()) {
            const auto completions = suffix[0];
            coordinates[g] = {address / completions, h};
            address %= completions;
            continue;
        }

        // Most factors have a single viable irrep. Then the rank is a mixed-radix digit.
        int only_irrep = -1;
        for (int local = 0; local < nirrep; ++local) {
            if (suffix[h ^ local] == 0 || factor.sectors[local].strings.empty())
                continue;
            if (only_irrep != -1) {
                only_irrep = -1;
                break;
            }
            only_irrep = local;
        }
        if (only_irrep != -1) {
            const auto completions = suffix[h ^ only_irrep];
            coordinates[g] = {address / completions, only_irrep};
            address %= completions;
            h ^= only_irrep;
            continue;
        }

        // Count completions for local strings before a position in the original order.
        // Invert this monotone count without constructing the product.
        const auto before = [&](size_t position) {
            size_t count = 0;
            for (int local = 0; local < nirrep; ++local) {
                if (suffix[h ^ local] == 0)
                    continue;
                const auto& positions = factor.sectors[local].positions;
                const auto end = std::lower_bound(positions.begin(), positions.end(), position);
                count += (end - positions.begin()) * suffix[h ^ local];
            }
            return count;
        };
        size_t first = 0, last = factor.in_order.size();
        while (first + 1 < last) {
            const size_t mid = first + (last - first) / 2;
            if (before(mid) <= address)
                first = mid;
            else
                last = mid;
        }
        coordinates[g] = factor.in_order[first];
        address -= before(first);
        h ^= coordinates[g].irrep;
    }
}

String StringAddress::string_at(int string_class, size_t address,
                                std::span<LocalStringAddress> work) const {
    unrank(string_class, address, work);
    const auto& factors = product(string_class).factors;
    auto string = String::zero();
    for (size_t g = 0; g < factors.size(); ++g) {
        const auto [index, h] = work[g];
        string.union_with(factors[g]->sectors[h].strings[index]);
    }
    return string;
}

std::optional<StringAddress::Index> StringAddress::find(const String& string,
                                                        std::span<LocalStringAddress> work,
                                                        occupation_t& occupation) const {
    size_t electrons = 0;
    for (size_t g = 0; g < basis_->ngas(); ++g) {
        occupation[g] = static_cast<int>(string.intersection_count(basis_->mask(g)));
        electrons += occupation[g];
    }
    if (electrons != string.count())
        return std::nullopt;
    const auto string_class =
        find_class(occupation, static_cast<int>(string.symmetry(basis_->irreps())));
    if (!string_class)
        return std::nullopt;
    const auto& factors = product(*string_class).factors;
    for (size_t g = 0; g < factors.size(); ++g)
        work[g] = factors[g]->addresses.at(String(string & basis_->mask(g)));
    return Index{rank(*string_class, work), *string_class};
}

} // namespace forte2
