template <typename T, int N>
nb::ndarray<nb::numpy, T, nb::ndim<N>> make_ndarray(std::unique_ptr<std::vector<T>> vec,
                                                    const std::array<size_t, N>& shape) {
    // raw pointer to the data
    T* data_ptr = vec->data();

    // release ownership of the vector onto the heap
    // so we can delete it when Python is done
    auto* heap_vec = vec.release();

    // capsule will call delete on the vector when the array dies
    nb::capsule deleter(heap_vec, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });

    // construct the ndarray(view) with shape and the deleter capsule
    return nb::ndarray<nb::numpy, T, nb::ndim<N>>(data_ptr, static_cast<size_t>(N), shape.data(),
                                                  std::move(deleter));
}