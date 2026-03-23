#pragma once
#include <thread>
#include <version>
#include <execution>
#include <algorithm>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

namespace forte2 {
static std::size_t get_num_threads() {
    return static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
}

// If std::execution::par_unseq is defined, use that
// else, if Apple, then use dispatch_apply for parallelism
// finally, fall back to std::thread
#if defined(__cpp_lib_execution)
template <typename F> void parallel_for(std::size_t begin, std::size_t end, F&& func) {
    std::vector<std::size_t> indices(end - begin);
    std::iota(indices.begin(), indices.end(), begin);
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), func);
}
#elif defined(__APPLE__)
template <typename F> void parallel_for(std::size_t begin, std::size_t end, F&& func) {
    std::size_t count = end - begin;

    dispatch_apply(count, DISPATCH_APPLY_AUTO, ^(size_t i) {
      func(begin + i);
    });
}
#else
template <typename F> void parallel_for(std::size_t begin, std::size_t end, F&& func) {
    std::size_t count = end - begin;
    std::vector<std::thread> threads;
    const auto num_threads = get_num_threads();
    const std::size_t block_size = (count + num_threads - 1) / num_threads;

    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t block_begin = begin + t * block_size;
        std::size_t block_end = std::min(block_begin + block_size, end);
        if (block_begin < block_end) {
            threads.emplace_back([=] {
                for (std::size_t i = block_begin; i < block_end; ++i) {
                    func(i);
                }
            });
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }
}
// template <typename F> void parallel_for(std::size_t begin, std::size_t end, F&& func) {
//     std::size_t count = end - begin;
//     std::vector<std::future<void>> tasks;
//     const auto num_threads = get_num_threads();
//     const std::size_t block_size = (count + num_threads - 1) / num_threads;

//     for (std::size_t t = 0; t < num_threads; ++t) {
//         std::size_t block_begin = begin + t * block_size;
//         std::size_t block_end = std::min(block_begin + block_size, end);
//         if (block_begin < block_end) {
//             tasks.emplace_back(std::async(std::launch::async, [=] {
//                 for (std::size_t i = block_begin; i < block_end; ++i) {
//                     func(i);
//                 }
//             }));
//         }
//     }

//     for (auto& task : tasks) {
//         task.get();
//     }
// }

#endif

} // namespace forte2