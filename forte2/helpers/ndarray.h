#pragma once

#include <array>
#include <numeric>
#include <memory>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

using np_matrix = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

template <typename Type, typename T, int N>
nb::ndarray<Type, T, nb::ndim<N>> make_ndarray(std::unique_ptr<std::vector<T>> vec,
                                               const std::array<size_t, N>& shape) {
    // raw pointer to the data
    T* data_ptr = vec->data();

    // release ownership of the vector onto the heap
    // so we can delete it when Python is done
    auto* heap_vec = vec.release(); // a std::vector<T>*

    // capsule will call delete on the vector when the array dies
    nb::capsule deleter(heap_vec, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });

    // construct the ndarray(view) with shape and the deleter capsule
    return nb::ndarray<Type, T, nb::ndim<N>>(data_ptr, static_cast<size_t>(N), shape.data(),
                                             std::move(deleter));
}

template <typename Type, typename T, int N>
nb::ndarray<Type, T, nb::ndim<N>> make_ndarray(const std::array<size_t, N>& shape) {
    // allocate a vector of the right size
    auto vec = std::make_unique<std::vector<T>>(
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()));

    // raw pointer to the data
    T* data_ptr = vec->data();

    // release ownership of the vector onto the heap
    // so we can delete it when Python is done
    auto* heap_vec = vec.release(); // a std::vector<T>*

    // capsule will call delete on the vector when the array dies
    nb::capsule deleter(heap_vec, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });

    // construct the ndarray(view) with shape and the deleter capsule
    return nb::ndarray<Type, T, nb::ndim<N>>(data_ptr, static_cast<size_t>(N), shape.data(),
                                             std::move(deleter));
}