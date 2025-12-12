#pragma once
#include <thread>

namespace forte2 {
std::size_t get_num_threads() {
    return static_cast<std::size_t>(std::max(1u, std::thread::hardware_concurrency()));
}
} // namespace forte2