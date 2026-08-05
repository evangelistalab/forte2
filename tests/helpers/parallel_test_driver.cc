#include <atomic>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

#ifdef __linux__
#include <sched.h>
#endif

#include "forte2/helpers/parallel.h"

namespace {

int pin_to_one_cpu() {
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
        return 77;

    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(first_cpu, &affinity);
    return sched_setaffinity(0, sizeof(affinity), &affinity) == 0 ? 0 : 77;
#else
    return 77;
#endif
}

bool joins_workers_during_unwinding() {
    std::atomic<bool> worker_completed{false};
    try {
        std::vector<std::thread> workers;
        workers.reserve(2);
        forte2::detail::ThreadJoiner thread_joiner(workers);
        workers.emplace_back([&worker_completed]() { worker_completed.store(true); });
        throw std::runtime_error("simulate failure to create the next worker");
    } catch (const std::runtime_error&) {
        return worker_completed.load();
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--join-on-exception") {
        const bool joined = joins_workers_during_unwinding();
        std::cout << joined << '\n';
        return joined ? 0 : 1;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--pin-one") {
        if (const int status = pin_to_one_cpu(); status != 0)
            return status;
    }

    const std::size_t num_threads = forte2::get_num_threads();
    const std::size_t count = 3 * num_threads;
    std::atomic<std::size_t> chunks{0};
    std::atomic<std::size_t> visited{0};
    forte2::parallel_for_chunked(count, [&](const std::size_t begin, const std::size_t end) {
        chunks.fetch_add(1);
        visited.fetch_add(end - begin);
    });

    std::cout << num_threads << ' ' << chunks.load() << ' ' << visited.load() << '\n';
    return visited.load() == count ? 0 : 1;
}
