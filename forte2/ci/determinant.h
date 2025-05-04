#pragma once

#include <functional>
#include <tuple>

#include "ci/occupation_vector.h"

namespace forte2 {
struct Determinant {
    OccupationVector a_;
    OccupationVector b_;

    void clear() {
        a_.clear();
        b_.clear();
    }

    static Determinant zero() {
        Determinant d;
        d.a_.clear();
        d.b_.clear();
        return d;
    }

    static constexpr int norb = OccupationVector::N;

    bool get_a(int p) const { return a_[p]; }

    bool get_b(int p) const { return b_[p]; }

    void set_a(int p, bool value) { a_.set(p, value); }

    void set_b(int p, bool value) { b_.set(p, value); }

    int count() const noexcept { return a_.count() + b_.count(); }

    int count_a() const noexcept { return a_.count(); }

    int count_b() const noexcept { return b_.count(); }

    bool operator==(const Determinant& other) const noexcept {
        return a_ == other.a_ and b_ == other.b_;
    }

    struct Hash {
        std::size_t operator()(const Determinant& d) const noexcept {
            uint64_t a =
                d.a_.raw(); // You must expose this via a method like `raw()` or `as_uint64()`
            uint64_t b = d.b_.raw();

            // A simple 64-bit mix function (based on Boost hash_combine or splitmix64)
            uint64_t hash = a ^ (b + 0x9e3779b97f4a7c15 + (a << 6) + (a >> 2));
            return static_cast<std::size_t>(hash);
        }
    };
};

std::string str(const Determinant& d, int n = OccupationVector::N);
} // namespace forte2

namespace std {
template <> struct hash<forte2::Determinant> {
    std::size_t operator()(const forte2::Determinant& d) const noexcept {
        return forte2::hash_combine(d.a_.raw(), d.b_.raw());
    }
};
} // namespace std