#pragma once

#include <cstddef>
#include <thread>
#include <version>
#include <execution>
#include <algorithm>
#include <atomic>
#include <ranges>
#include <future>
#include <utility>
#include <vector>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

#define SERIAL_THRESHOLD 3

namespace forte2 {
static std::size_t get_num_threads() {
    return static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
}

template <typename F> void serial_for(const std::size_t begin, const std::size_t end, F&& func) {
    for (std::size_t i{begin}; i < end; ++i)
        func(i);
}

template <typename F>
void parallel_for_chunked_thread(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    std::size_t count = end - begin;
    const auto num_threads = get_num_threads();
    if (num_threads <= 1 || count < SERIAL_THRESHOLD * num_threads) {
        func(begin, end);
        return;
    }
    std::vector<std::thread> threads;
    const std::size_t block_size = (count + num_threads - 1) / num_threads;

    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t block_begin = begin + t * block_size;
        std::size_t block_end = std::min(block_begin + block_size, end);
        if (block_begin < block_end) {
            threads.emplace_back(
                [block_begin, block_end, &func]() { func(block_begin, block_end); });
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename F>
void parallel_for_chunked_async(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    std::size_t count = end - begin;
    const auto num_threads = get_num_threads();
    if (num_threads <= 1 || count < SERIAL_THRESHOLD * num_threads) {
        func(begin, end);
        return;
    }
    std::vector<std::future<void>> tasks;
    const std::size_t block_size = (count + num_threads - 1) / num_threads;

    for (std::size_t t = 0; t < num_threads; ++t) {
        std::size_t block_begin = begin + t * block_size;
        std::size_t block_end = std::min(block_begin + block_size, end);
        if (block_begin < block_end) {
            tasks.emplace_back(std::async(std::launch::async, [block_begin, block_end, &func]() {
                func(block_begin, block_end);
            }));
        }
    }

    for (auto& task : tasks) {
        task.get();
    }
}

template <typename F>
void parallel_for_interleaved_async(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    std::size_t count = end - begin;
    const auto num_threads = get_num_threads();

    if (num_threads <= 1 || count < SERIAL_THRESHOLD * num_threads) {
        serial_for(begin, end, func);
        return;
    }

    std::vector<std::future<void>> workers;
    workers.reserve(num_threads);
    for (std::size_t t{0}; t < num_threads; ++t) {
        workers.push_back(std::async(std::launch::async, [count, num_threads, t, begin, &func]() {
            for (std::size_t i{t}; i < count; i += num_threads)
                func(i + begin);
        }));
    }

    for (auto& w : workers)
        w.get();
}

template <typename F>
void parallel_for_interleaved_thread(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    std::size_t count = end - begin;
    const auto num_threads = get_num_threads();
    if (num_threads <= 1 || count < SERIAL_THRESHOLD * num_threads) {
        serial_for(begin, end, func);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (std::size_t t{0}; t < num_threads; ++t) {
        workers.emplace_back([count, num_threads, t, begin, &func]() {
            for (std::size_t i{t}; i < count; i += num_threads)
                func(i + begin);
        });
    }

    for (auto& w : workers)
        w.join();
}

template <typename F>
void parallel_for_dynamic_async(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    const auto num_threads = get_num_threads();
    if (num_threads <= 1 || end - begin < SERIAL_THRESHOLD * num_threads) {
        serial_for(begin, end, func);
        return;
    }
    std::atomic<std::size_t> next_index(begin);
    std::vector<std::future<void>> tasks;
    for (std::size_t t = 0; t < num_threads; ++t) {
        tasks.emplace_back(std::async(std::launch::async, [begin, end, &func, &next_index]() {
            while (true) {
                std::size_t index = next_index.fetch_add(1);
                if (index >= end)
                    break;
                func(index);
            }
        }));
    }
    for (auto& task : tasks) {
        task.get();
    }
}

template <typename F>
void parallel_for_dynamic_thread(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    const auto num_threads = get_num_threads();
    if (num_threads <= 1 || end - begin < SERIAL_THRESHOLD * num_threads) {
        serial_for(begin, end, func);
        return;
    }
    std::atomic<std::size_t> next_index(begin);
    std::vector<std::thread> threads;
    for (std::size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([begin, end, &func, &next_index]() {
            while (true) {
                std::size_t index = next_index.fetch_add(1);
                if (index >= end)
                    break;
                func(index);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

template <typename F>
void parallel_for_chunked(const std::size_t begin, const std::size_t end, F&& func) {
    parallel_for_chunked_thread(begin, end, std::forward<F>(func));
}

template <typename F> void parallel_for_chunked(const std::size_t count, F&& func) {
    return parallel_for_chunked(0, count, std::forward<F>(func));
}

// These are only defined if the compiler supports C++17 parallel algorithms (Apple Clang does not
// support this)
#if defined(__cpp_lib_execution)
template <typename F>
void parallel_for_each(const std::size_t begin, const std::size_t end, F&& func) {
    std::ranges::iota_view indices(begin, end);
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), func);
}

template <typename F> void parallel_for_each(const std::size_t count, F&& func) {
    return parallel_for_each(0, count, std::forward<F>(func));
}
#endif

// if Apple, then use dispatch_apply for parallelism
// finally, fall back to std::thread
#if defined(__APPLE__)
template <typename F> void parallel_for(const std::size_t begin, const std::size_t end, F&& func) {
    if (end <= begin)
        return;
    std::size_t count = end - begin;

    dispatch_apply(count, DISPATCH_APPLY_AUTO, ^(std::size_t i) {
      func(begin + i);
    });
}

template <typename F> void parallel_for(const std::size_t count, F&& func) {
    return parallel_for(0, count, std::forward<F>(func));
}
#else
template <typename F> void parallel_for(const std::size_t begin, const std::size_t end, F&& func) {
    parallel_for_interleaved_thread(begin, end, std::forward<F>(func));
}

template <typename F> void parallel_for(const std::size_t count, F&& func) {
    return parallel_for(0, count, std::forward<F>(func));
}
#endif

} // namespace forte2
