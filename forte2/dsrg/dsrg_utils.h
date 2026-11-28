#pragma once

#include <cmath>
#include <cassert>
#include "helpers/ndarray.h"
#include "helpers/parallel.h"

constexpr double taylor_threshold = 3;
constexpr double taylor_epsilon = 1e-3;

namespace forte2 {

/// @brief Computes the Taylor expansion of (1 - exp(-z^2)) / z up to the order n,
/// where n is determined by the threshold TAYLOR_THRES.
double taylor_exp(double z);

/// @brief Computes the regularized denominator (1 - exp(-s*x^2)) / x
double regularized_denominator(double x, double s);

/// @brief Computes the regularized denominator (1 - exp(-s*x^2)) / x, given exp_term =
/// exp(-s*x^2)
inline double regularized_denominator(double x, double s, double exp_term) {
    double z = std::sqrt(s) * x;
    if (std::fabs(z) <= taylor_epsilon) {
        return taylor_exp(z) * std::sqrt(s);
    }
    return (1. - exp_term) / x;
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
    parallel_for_chunked(ni * nj * na, [&](size_t begin, size_t end) {
        for (size_t ija = begin; ija < end; ija++) {
            const size_t i = ija / (nj * na), j = (ija / na) % nj, a = ija % na;
            const double d_ija = ei_v(i) + ej_v(j) - ea_v(a);
            for (size_t b = 0; b < nb; b++) {
                const double denom = d_ija - eb_v(b);
                t2_v(i, j, a, b) *= regularized_denominator(denom, flow_param,
                                                            std::exp(-flow_param * denom * denom));
            }
        }
    });
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
    parallel_for_chunked(ni * nj * na, [&](size_t begin, size_t end) {
        for (size_t ija = begin; ija < end; ija++) {
            const size_t i = ija / (nj * na), j = (ija / na) % nj, a = ija % na;
            const double d_ija = ei_v(i) + ej_v(j) - ea_v(a);
            for (size_t b = 0; b < nb; b++) {
                const double denom = d_ija - eb_v(b);
                v_v(i, j, a, b) *= 1 + std::exp(-flow_param * denom * denom);
            }
        }
    });
}

/// @brief Renormalizes a batch of three-index intermediates for on-the-fly computation of
/// expensive contractions, where
/// v_renorm = v * (1 + exp(-s*denom^2)) and t_renorm = t * (1 - exp(-s*denom^2)) / denom
/// @tparam T Either double or std::complex<double>, for spin-free or two-component calculations
/// respectively
/// @param v The C-contiguous batch of integrals to be renormalized
/// @param t The C-contiguous batch of integrals to be turned into amplitudes, of the same shape
/// as v
/// @param ep The orbital energy corresponding to the batched index
/// @param eq The orbital energies corresponding to the first dimension of v and t
/// @param er The orbital energies corresponding to the second dimension of v and t
/// @param es The orbital energies corresponding to the third dimension of v and t
/// @param flow_param The flow parameter controlling the renormalization
template <typename T>
void renormalize_3index(nb::ndarray<nb::numpy, T, nb::ndim<3>, nb::c_contig>& v,
                        nb::ndarray<nb::numpy, T, nb::ndim<3>, nb::c_contig>& t, double ep,
                        np_vector_c& eq, np_vector_c& er, np_vector_c& es, double flow_param) {
    const size_t nq = eq.shape(0);
    const size_t nr = er.shape(0);
    const size_t ns = es.shape(0);

    assert(nq == v.shape(0) && nq == t.shape(0));
    assert(nr == v.shape(1) && nr == t.shape(1));
    assert(ns == v.shape(2) && ns == t.shape(2));

    T* v_p = v.data();
    T* t_p = t.data();
    const double* eq_p = eq.data();
    const double* er_p = er.data();
    const double* es_p = es.data();
    parallel_for_chunked(nq * nr, [&](size_t begin, size_t end) {
        for (size_t qr = begin; qr < end; qr++) {
            const double d_qr = ep + eq_p[qr / nr] - er_p[qr % nr];
            T* v_row = v_p + qr * ns;
            T* t_row = t_p + qr * ns;
            for (size_t s = 0; s < ns; s++) {
                const double denom = d_qr - es_p[s];
                const double exp_term = std::exp(-flow_param * denom * denom);
                v_row[s] *= 1 + exp_term;
                t_row[s] *= regularized_denominator(denom, flow_param, exp_term);
            }
        }
    });
}

} // namespace forte2