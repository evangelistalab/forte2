#pragma once

#include <array>
#include <numeric>
#include <memory>
#include <vector>
#include <complex>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// Aliases for ndarray types used in forte2
using np_vector = nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig>;
using np_matrix = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig>;
using np_tensor3 = nb::ndarray<nb::numpy, double, nb::ndim<3>, nb::c_contig>;
using np_tensor4 = nb::ndarray<nb::numpy, double, nb::ndim<4>, nb::c_contig>;
using np_tensor5 = nb::ndarray<nb::numpy, double, nb::ndim<5>, nb::c_contig>;
using np_tensor6 = nb::ndarray<nb::numpy, double, nb::ndim<6>, nb::c_contig>;
using np_tensor7 = nb::ndarray<nb::numpy, double, nb::ndim<7>, nb::c_contig>;
using np_tensor8 = nb::ndarray<nb::numpy, double, nb::ndim<8>, nb::c_contig>;
using np_vector_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<1>, nb::c_contig>;
using np_matrix_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<2>, nb::c_contig>;
using np_tensor3_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<3>, nb::c_contig>;
using np_tensor4_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<4>, nb::c_contig>;
using np_tensor5_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<5>, nb::c_contig>;
using np_tensor6_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<6>, nb::c_contig>;
using np_tensor7_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<7>, nb::c_contig>;
using np_tensor8_complex = nb::ndarray<nb::numpy, std::complex<double>, nb::ndim<8>, nb::c_contig>;

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

/// @brief Allocates the memory for an ndarray of the given shape and type.
/// @details The memory is allocated on the heap and will be freed when the ndarray
///          is deleted via a deleter capsule passed to the ndarray constructor.
/// @tparam Type The type of the ndarray (e.g. nb::numpy, nb::pytorch, etc.)
/// @tparam T The type of the data (e.g. double, float, etc.)
/// @tparam N The number of dimensions of the ndarray
/// @param shape The shape of the ndarray as an array of size N.
/// @details The shape is a list of size N, where N is the number of dimensions.
/// @return An ndarray of the given shape and type.
/// @note The ndarray is not initialized, so the data is not set to any value.
template <typename Type, typename T, int N, typename Order = nb::c_contig>
nb::ndarray<Type, T, nb::ndim<N>, Order> make_ndarray(const std::array<size_t, N>& shape) {
    // compute the number of elements in the array
    const auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    // allocate a vector of the right size
    T* array_ptr = new T[size];

    // define a capsule to delete the data when the ndarray dies
    nb::capsule deleter(array_ptr, [](void* p) noexcept { delete[] static_cast<T*>(p); });

    // construct the ndarray (a view on the array data) with shape and the deleter capsule
    return nb::ndarray<Type, T, nb::ndim<N>, Order>(array_ptr, static_cast<size_t>(N), shape.data(),
                                                    std::move(deleter));
}

/// @brief Creates an ndarray of a given shape and type set to zero.
/// @details The memory is allocated on the heap and will be freed when the ndarray
///          is deleted via a deleter capsule passed to the ndarray constructor.
/// @tparam Type The type of the ndarray (e.g. nb::numpy, nb::pytorch, etc.)
/// @tparam T The type of the data (e.g. double, float, etc.)
/// @tparam N The number of dimensions of the ndarray
/// @param shape The shape of the ndarray as an array of size N.
/// @details The shape is a list of size N, where N is the number of dimensions.
/// @return An ndarray of the given shape and type.
/// @note The ndarray is not initialized, so the data is not set to any value.
template <typename Type, typename T, int N, typename Order = nb::c_contig>
nb::ndarray<Type, T, nb::ndim<N>, Order> make_zeros(const std::array<size_t, N>& shape) {
    // allocate the ndarray
    auto array = make_ndarray<Type, T, N, Order>(shape);

    // get the data pointer and size
    auto data_ptr = array.data();
    auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

    // fill the data with zeros
    std::fill(data_ptr, data_ptr + size, static_cast<T>(0));

    return array;
}
