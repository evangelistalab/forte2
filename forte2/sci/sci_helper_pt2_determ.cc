#include <atomic>
#include <future>

#include "helpers/logger.h"
#include "helpers/timer.hpp"

#include "sci_helper.h"
#include "sci_helper_contributions.hpp"

namespace forte2 {

void SelectedCIHelper::pt2_determ_batch(Pt2Scratch& s, double eps2, size_t num_batches,
                                        size_t batch_id, const DetSet& existing_dets,
                                        std::span<double> out, size_t& num_dets) const {
    auto& map = s.map;
    auto& slots = s.slots;
    map.clear();
    slots.clear();

    generate_contributions(
        s.conn, eps2, num_batches, batch_id, std::span<const uint64_t>{},
        [&](const Determinant& det, const double* c_parent, size_t /*det_index*/, double coupling,
            double criterion) {
            if (criterion <= eps2)
                return;
            if (existing_dets.count(det))
                return;
            // every contribution to <det|H|Psi> lands in the same entry, so the sum is complete
            // before it is squared below
            auto [it, emplaced] = map.try_emplace(det, slots.size());
            if (emplaced)
                slots.resize(slots.size() + nroots_, 0.0);
            const size_t idx = it->second;
            for (size_t r{0}; r < nroots_; ++r)
                slots[idx + r] += coupling * c_parent[r];
        });

    for (const auto& [det, idx] : map) {
        const double energy = slater_rules_.energy(det);
        for (size_t r{0}; r < nroots_; ++r)
            out[r] += compute_delta_ept2(root_energies_[r] - energy, slots[idx + r]);
    }
    num_dets += map.size();
}

void SelectedCIHelper::compute_pt2_determ(double eps2, size_t num_batches) {
    if (num_batches == 0)
        throw std::invalid_argument("compute_pt2_determ requires num_batches > 0");

    local_timer pt2_timer;

    compute_det_energies();
    prepare_strings();
    update_hbci_ints();

    // The correction runs over determinants outside the variational space, so every determinant
    // already in it has to be excluded. This set is built once and shared by all batches.
    const DetSet existing_dets(dets_.begin(), dets_.end());

    ept2_.assign(nroots_, 0.0);
    ept2_stddev_.assign(nroots_, 0.0);
    num_pt2_dets_ = 0;

    const auto num_threads = get_num_threads();
    std::atomic<size_t> next_batch(0);
    std::vector<std::vector<double>> local_ept2(num_threads, std::vector<double>(nroots_, 0.0));
    std::vector<size_t> local_num_dets(num_threads, 0);

    auto worker = [&](size_t thread_id) {
        // reused across batches; the underlying memory is kept and grown as needed
        Pt2Scratch s;
        while (true) {
            const size_t batch_id = next_batch.fetch_add(1);
            if (batch_id >= num_batches)
                break;
            pt2_determ_batch(s, eps2, num_batches, batch_id, existing_dets, local_ept2[thread_id],
                             local_num_dets[thread_id]);
        }
    };

    std::vector<std::future<void>> workers;
    for (size_t t{0}; t < num_threads; ++t)
        workers.push_back(std::async(std::launch::async, worker, t));
    for (auto& w : workers)
        w.get();

    for (size_t t{0}; t < num_threads; ++t) {
        for (size_t r{0}; r < nroots_; ++r)
            ept2_[r] += local_ept2[t][r];
        num_pt2_dets_ += local_num_dets[t];
    }

    ept2_determ_ = ept2_;
    ept2_pseudostoch_.assign(nroots_, 0.0);
    ept2_stoch_.assign(nroots_, 0.0);
    ept2_pseudostoch_stddev_.assign(nroots_, 0.0);
    ept2_stoch_stddev_.assign(nroots_, 0.0);
    num_pseudostoch_batches_ = num_batches;

    pt2_time_ = pt2_timer.elapsed_seconds();
    LOG(log_level_) << "Deterministic PT2 with eps2 = " << eps2 << " over " << num_batches
                    << " batches: " << num_pt2_dets_ << " external determinants in " << pt2_time_
                    << " s";
}

} // namespace forte2
