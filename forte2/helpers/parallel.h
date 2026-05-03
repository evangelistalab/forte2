#pragma once

#include <algorithm>
#include <cstddef>
#include <future>
#include <thread>
#include <vector>

namespace forte2 {
static std::size_t get_num_threads() {
    return static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
}

/// @brief Run a loop in parallel over a given number of indices
template <typename WorkFn>
void run_parallel_indices(std::size_t count, std::size_t num_threads, WorkFn&& work) {
    if (num_threads <= 1 || count == 0) {
        for (std::size_t i{0}; i < count; ++i)
            work(i);
        return;
    }

    std::vector<std::future<void>> workers;
    workers.reserve(num_threads);
    for (std::size_t t{0}; t < num_threads; ++t) {
        workers.push_back(std::async(std::launch::async, [count, num_threads, t, &work]() {
            for (std::size_t i{t}; i < count; i += num_threads)
                work(i);
        }));
    }

    for (auto& w : workers)
        w.get();
}
} // namespace forte2
