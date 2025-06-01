#pragma once

#include <chrono>

namespace forte2 {

/**
 * @brief A simple timer that measures elapsed time.
 *
 * By default this uses std::chrono::steady_clock (monotonic, not subject to system-clock updates).
 * The timer starts at construction, and you can call elapsed_seconds() to get the elapsed duration
 * since construction or last reset.  Calling reset() resets the “start time” to now().
 *
 * Usage:
 *   {
 *     local_timer t;               // timer starts
 *     … do some work …
 *     double s = t.elapsed_seconds();  // seconds since t was constructed (or last reset)
 *     t.reset();                   // restart timer
 *     … do more work …
 *     s = t.elapsed_seconds();         // seconds since reset
 *   }
 */
class local_timer {
  public:
    using clock_type = std::chrono::steady_clock;
    using time_point = clock_type::time_point;
    using duration = clock_type::duration;
    using rep = duration::rep;
    using period = duration::period;

    /// Constructs and immediately starts the timer.
    local_timer() noexcept : start_time_{clock_type::now()} {}

    /// Resets the timestamp to now().
    void reset() noexcept { start_time_ = clock_type::now(); }

    /// Returns the elapsed time in seconds (double).
    [[nodiscard]] double elapsed_seconds() const noexcept {
        auto delta = clock_type::now() - start_time_;
        // duration_cast to duration<double> to get seconds as double
        return std::chrono::duration<double>(delta).count();
    }

    /// Returns the raw duration object (useful if you want milliseconds, etc.).
    [[nodiscard]] duration elapsed() const noexcept { return clock_type::now() - start_time_; }

  private:
    time_point start_time_;
};

} // namespace forte2