#pragma once

#include <functional>
#include <tuple>

#include "ci/occupation_vector.h"

namespace forte2 {
struct Determinant {
    OccupationVector a;
    OccupationVector b;

    Determinant() = default;

    Determinant(OccupationVector a, OccupationVector b) : a(a), b(b) {}

    static Determinant zero() {
        Determinant d;
        d.a.clear();
        d.b.clear();
        return d;
    }

    static constexpr int norb = OccupationVector::N;

    void clear() {
        a.clear();
        b.clear();
    }

    bool get_a(int p) const { return a[p]; }

    bool get_b(int p) const { return b[p]; }

    void set_a(int p, bool value) { a.set(p, value); }

    void set_b(int p, bool value) { b.set(p, value); }

    int count() const noexcept { return a.count() + b.count(); }

    int count_a() const noexcept { return a.count(); }

    int count_b() const noexcept { return b.count(); }

    bool operator==(const Determinant& other) const noexcept {
        return a == other.a and b == other.b;
    }

    bool operator<(const Determinant& other) const noexcept {
        return std::tie(a, b) < std::tie(other.a, other.b);
    }

    int count_diff(const Determinant& other) const noexcept {
        return a.count_diff(other.a) + b.count_diff(other.b);
    }
    int count_same(const Determinant& other) const noexcept {
        return a.count_same(other.a) + b.count_same(other.b);
    }
};

std::string str(const Determinant& d, int n = OccupationVector::N);
} // namespace forte2

namespace std {
// specialization of std::hash for forte2::Determinant
template <> struct hash<forte2::Determinant> {
    std::size_t operator()(const forte2::Determinant& d) const noexcept {
        return forte2::hash_combine(d.a.raw(), d.b.raw());
    }
};
} // namespace std
