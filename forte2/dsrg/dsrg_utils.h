#pragma once

#include <cmath>
#include <cassert>
#include "helpers/ndarray.h"

constexpr double taylor_threshold = 3;
constexpr double taylor_epsilon = 1e-3;

namespace forte2 {

/// @brief Computes the Taylor expansion of (1 - exp(-z^2)) / z up to the order n,
/// where n is determined by the threshold TAYLOR_THRES.
double taylor_exp(double z) {
    int n = static_cast<int>(0.5 * (15.0 / taylor_threshold + 1)) + 1;
    if (n > 0) {
        double value = z;
        double tmp = z;
        for (int x = 0; x < n - 1; x++) {
            tmp *= -1.0 * z * z / (x + 2);
            value += tmp;
        }
        return value;
    } else {
        return 0.0;
    }
}

/// @brief Computes the regularized denominator (1 - exp(-s*x^2)) / x
double regularized_denominator(double x, double s) {
    double z = std::sqrt(s) * x;
    if (fabs(z) <= taylor_epsilon) {
        return taylor_exp(z) * std::sqrt(s);
    } else {
        return (1. - std::exp(-s * x * x)) / x;
    }
}

/// @brief Computes the renormalized T2 amplitudes for a given block.
/// where T2_renorm = T2 * (1 - exp(-s*denom^2)) / denom
/// @tparam T: Either double or std::complex<double>, for spin-free or two-component calculations
/// respectively
/// @param t2 The block of integrals to be renormalized
/// @param ei The orbital energies corresponding to the first dimension of t2
/// @param ej The orbital energies corresponding to the second dimension of t2
/// @param ea The orbital energies corresponding to the third dimension of t2
/// @param eb The orbital energies corresponding to the fourth dimension of t2
/// @param flow_param The flow parameter controlling the renormalization
template <typename T>
void compute_T2_block(nb::ndarray<nb::numpy, T, nb::ndim<4>>& t2, np_vector& ei, np_vector& ej,
                      np_vector& ea, np_vector& eb, double flow_param) {
    auto t2_v = t2.view();
    auto ei_v = ei.view();
    auto ej_v = ej.view();
    auto ea_v = ea.view();
    auto eb_v = eb.view();

    size_t ni = ei.shape(0);
    size_t nj = ej.shape(0);
    size_t na = ea.shape(0);
    size_t nb = eb.shape(0);

    assert(ni == t2.shape(0));
    assert(nj == t2.shape(1));
    assert(na == t2.shape(2));
    assert(nb == t2.shape(3));
    double denom;
    for (size_t i = 0; i < ni; i++) {
        for (size_t j = 0; j < nj; j++) {
            for (size_t a = 0; a < na; a++) {
                for (size_t b = 0; b < nb; b++) {
                    denom = ei_v(i) + ej_v(j) - ea_v(a) - eb_v(b);
                    t2_v(i, j, a, b) *= static_cast<T>(regularized_denominator(denom, flow_param));
                }
            }
        }
    }
}

/// @brief Computes the renormalized T1 amplitudes for a given block.
/// where T1_renorm = T1 * (1 - exp(-s*denom^2)) / denom
/// @tparam T Either double or std::complex<double>, for spin-free or two-component calculations
/// respectively
/// @param t1 The block of T1 amplitudes to be renormalized
/// @param ei The orbital energies corresponding to the first dimension of t1
/// @param ea The orbital energies corresponding to the second dimension of t1
/// @param flow_param The flow parameter controlling the renormalization
template <typename T>
void compute_T1_block(nb::ndarray<nb::numpy, T, nb::ndim<2>>& t1, np_vector& ei, np_vector& ea,
                      double flow_param) {
    auto t1_v = t1.view();
    auto ei_v = ei.view();
    auto ea_v = ea.view();

    size_t ni = ei.shape(0);
    size_t na = ea.shape(0);

    assert(ni == t1.shape(0));
    assert(na == t1.shape(1));
    double denom;
    for (size_t i = 0; i < ni; i++) {
        for (size_t a = 0; a < na; a++) {
            denom = ei_v(i) - ea_v(a);
            t1_v(i, a) *= static_cast<T>(regularized_denominator(denom, flow_param));
        }
    }
}

/// @brief Renormalizes a block of two-electron integrals.
/// where V_renorm = V * (1 + exp(-s*denom^2))
/// @tparam T: Either double or std::complex<double>, for spin-free or two-component calculations
/// respectively
/// @param v The block of integrals to be renormalized
/// @param ei The orbital energies corresponding to the first dimension of v
/// @param ej The orbital energies corresponding to the second dimension of v
/// @param ea The orbital energies corresponding to the third dimension of v
/// @param eb The orbital energies corresponding to the fourth dimension of v
/// @param flow_param The flow parameter controlling the renormalization
template <typename T>
void renormalize_V_block(nb::ndarray<nb::numpy, T, nb::ndim<4>>& v, np_vector& ei, np_vector& ej,
                         np_vector& ea, np_vector& eb, double flow_param) {
    auto v_v = v.view();
    auto ei_v = ei.view();
    auto ej_v = ej.view();
    auto ea_v = ea.view();
    auto eb_v = eb.view();

    size_t ni = ei.shape(0);
    size_t nj = ej.shape(0);
    size_t na = ea.shape(0);
    size_t nb = eb.shape(0);

    assert(ni == v.shape(0));
    assert(nj == v.shape(1));
    assert(na == v.shape(2));
    assert(nb == v.shape(3));
    double denom;
    for (size_t i = 0; i < ni; i++) {
        for (size_t j = 0; j < nj; j++) {
            for (size_t a = 0; a < na; a++) {
                for (size_t b = 0; b < nb; b++) {
                    denom = ei_v(i) + ej_v(j) - ea_v(a) - eb_v(b);
                    v_v(i, j, a, b) +=
                        v_v(i, j, a, b) * static_cast<T>(std::exp(-flow_param * denom * denom));
                }
            }
        }
    }
}

/// @brief Renormalizes a block of three-index intermediates
/// for on-the-fly computation of expensive contractions, where
/// V_renorm = V * (1 + exp(-s*denom^2)) * (1 - exp(-s*denom^2)) / denom
/// @tparam T Either double or std::complex<double>, for spin-free or two-component calculations
/// respectively
/// @param v The block of three-index integrals to be renormalized
/// @param ep The orbital energy corresponding to the batched index of v
/// @param eq The orbital energies corresponding to the first dimension of v
/// @param er The orbital energies corresponding to the second dimension of v
/// @param es The orbital energies corresponding to the third dimension of v
/// @param flow_param The flow parameter controlling the renormalization
template <typename T>
void renormalize_3index(nb::ndarray<nb::numpy, T, nb::ndim<3>>& v, double& ep, np_vector& eq,
                        np_vector& er, np_vector& es, double flow_param) {
    auto v_v = v.view();
    auto eq_v = eq.view();
    auto er_v = er.view();
    auto es_v = es.view();

    size_t nq = eq.shape(0);
    size_t nr = er.shape(0);
    size_t ns = es.shape(0);

    assert(nq == v.shape(0));
    assert(nr == v.shape(1));
    assert(ns == v.shape(2));
    double denom;
    for (size_t q = 0; q < nq; q++) {
        for (size_t r = 0; r < nr; r++) {
            for (size_t s = 0; s < ns; s++) {
                denom = ep + eq_v(q) - er_v(r) - es_v(s);
                v_v(q, r, s) *= static_cast<T>((1 + std::exp(-flow_param * denom * denom)) *
                                               regularized_denominator(denom, flow_param));
            }
        }
    }
}

} // namespace forte2