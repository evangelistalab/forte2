#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

#include "forte2/helpers/parallel.h"

namespace {

int test_pin_to_one_cpu() {
#ifdef __linux__
    cpu_set_t available;
    CPU_ZERO(&available);
    if (sched_getaffinity(0, sizeof(available), &available) != 0)
        return 77;

    int first_cpu = -1;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &available)) {
            first_cpu = cpu;
            break;
        }
    }
    if (first_cpu < 0)
        return 77; // unsupported environment - not a failure

    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(first_cpu, &affinity);
    return sched_setaffinity(0, sizeof(affinity), &affinity) == 0 ? 0 : 77;
#else
    return 77; // unsupported environment - not a failure
#endif
}

/// @brief Verify ThreadJoiner joins every already-launched worker while an exception unwinds.
///
/// This models a failure that occurs partway through launching workers: several threads are
/// already running when the next launch "fails" (simulated by a throw). We wait until every worker
/// has started, then throw; because each worker keeps working (a short sleep) after signaling that
/// it started, all workers are still running when ~ThreadJoiner fires during unwinding, so the join
/// must actually wait for them.
///
/// The post-condition relies on join's happens-before guarantee: only if every worker was joined
/// are all their `completed` increments visible here, giving completed == num_workers. A
/// ThreadJoiner that failed to join would either let the process std::terminate (a still-running
/// thread's destructor) or race ahead of the sleeping workers and observe completed < num_workers.
bool test_thread_joiner() {
    constexpr unsigned num_workers = 8;
    std::atomic<unsigned> started{0};
    std::atomic<unsigned> completed{0};

    try {
        std::vector<std::thread> workers;
        workers.reserve(num_workers);
        forte2::detail::ThreadJoiner thread_joiner(workers);
        for (unsigned w = 0; w < num_workers; ++w) {
            workers.emplace_back([&]() {
                started.fetch_add(1);
                // Keep working after signaling start, so the worker is still running (joinable)
                // when the throw unwinds through ~ThreadJoiner.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                completed.fetch_add(1);
            });
        }
        // Ensure every worker is actually running before we unwind.
        while (started.load() < num_workers)
            std::this_thread::yield();
        throw std::runtime_error("simulate failure to launch the next worker");
    } catch (const std::runtime_error&) {
        // Reachable only after ~ThreadJoiner joined all workers during stack unwinding.
        return completed.load() == num_workers;
    }
}

/// @brief Verify every index in [0, count) was visited exactly once.
///
/// Returns 0 when coverage is exact, or 1 otherwise (printing the offending statistics to stderr
/// for diagnostics). An empty range is vacuously exact.
int report_coverage(const std::vector<std::atomic<unsigned>>& visits, const std::size_t count) {
    unsigned long long total = 0;
    unsigned min_visits = std::numeric_limits<unsigned>::max();
    unsigned max_visits = 0;
    for (const auto& visit : visits) {
        const unsigned seen = visit.load();
        total += seen;
        min_visits = std::min(min_visits, seen);
        max_visits = std::max(max_visits, seen);
    }
    const bool exact = count == 0 ? true : (total == count && min_visits == 1 && max_visits == 1);
    if (!exact) {
        std::cerr << "coverage not exact for count=" << count << ": total=" << total
                  << " min=" << min_visits << " max=" << max_visits << '\n';
        return 1;
    }
    return 0;
}

/// @brief Drive an index-callback variant of parallel_for_*, e.g. func(i), and check exact-once coverage.
template <typename Invoker> int run_index_fn(const std::size_t count, Invoker invoke) {
    std::vector<std::atomic<unsigned>> visits(count);
    invoke([&visits](const std::size_t i) { visits[i].fetch_add(1, std::memory_order_relaxed); });
    return report_coverage(visits, count);
}

/// @brief Drive a range-callback variant of parallel_for_*, e.g. func(begin, end), and check exact-once coverage.
template <typename Invoker> int run_range_fn(const std::size_t count, Invoker invoke) {
    std::vector<std::atomic<unsigned>> visits(count);
    invoke([&visits](const std::size_t begin, const std::size_t end) {
        for (std::size_t i = begin; i < end; ++i)
            visits[i].fetch_add(1, std::memory_order_relaxed);
    });
    return report_coverage(visits, count);
}

int test_parallel_for_variant(const std::string_view name, const std::size_t count) {
    // Range-callback variants: func(begin, end).
    if (name == "parallel_for_chunked")
        return run_range_fn(count,
                            [count](auto&& f) { forte2::parallel_for_chunked(0, count, f); });
    if (name == "parallel_for_chunked_thread")
        return run_range_fn(
            count, [count](auto&& f) { forte2::parallel_for_chunked_thread(0, count, f); });
    if (name == "parallel_for_chunked_async")
        return run_range_fn(
            count, [count](auto&& f) { forte2::parallel_for_chunked_async(0, count, f); });

    // Index-callback variants: func(i).
    if (name == "parallel_for")
        return run_index_fn(count, [count](auto&& f) { forte2::parallel_for(0, count, f); });
    if (name == "parallel_for_interleaved_thread")
        return run_index_fn(
            count, [count](auto&& f) { forte2::parallel_for_interleaved_thread(0, count, f); });
    if (name == "parallel_for_interleaved_async")
        return run_index_fn(
            count, [count](auto&& f) { forte2::parallel_for_interleaved_async(0, count, f); });
    if (name == "parallel_for_dynamic_thread")
        return run_index_fn(
            count, [count](auto&& f) { forte2::parallel_for_dynamic_thread(0, count, f); });
    if (name == "parallel_for_dynamic_async")
        return run_index_fn(
            count, [count](auto&& f) { forte2::parallel_for_dynamic_async(0, count, f); });

    std::cerr << "unknown parallel_for variant: " << name << '\n';
    return 2;
}

/// @brief Pin the process to one CPU and verify get_num_threads() honors the affinity mask.
///
/// Once pinned to a single CPU, get_num_threads() must report 1. Returns 0 on success, 1 on
/// failure (printing the observed count for diagnostics), or 77 (skip) when affinity controls are
/// unavailable.
int test_process_affinity() {
    if (const int status = test_pin_to_one_cpu(); status != 0)
        return status;

    const std::size_t num_threads = forte2::get_num_threads();
    if (num_threads != 1) {
        std::cerr << "expected get_num_threads() == 1 when pinned to one CPU, got " << num_threads
                  << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string_view mode = argc >= 2 ? std::string_view(argv[1]) : std::string_view();

    if (argc == 2 && mode == "--test-thread-joiner") {
        if (test_thread_joiner())
            return 0;
        std::cerr << "ThreadJoiner did not join all workers during unwinding\n";
        return 1;
    }
    if (argc == 4 && mode == "--test-parallel-for-variant") {
        const std::size_t count = std::strtoull(argv[3], nullptr, 10);
        return test_parallel_for_variant(std::string_view(argv[2]), count);
    }
    if (argc == 2 && mode == "--test-process-affinity")
        return test_process_affinity();

    std::cerr << "usage: " << argv[0]
              << " {--test-thread-joiner | --test-parallel-for-variant <name> <count>"
                 " | --test-process-affinity}\n";
    return 2;
}
