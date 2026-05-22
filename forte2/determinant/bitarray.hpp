#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "determinant/bitwise_operations.hpp"

namespace forte2 {

/**
 * @brief This class represents an array of N bits. The bits are stored in
 *        groups called "words". Each word contains 64 bits stored as
 *        64-bit unsigned integers.
 *
 *        Words are store in a std::array object:
 *          std::array<word_t, nwords_> words_ = {};
 *
 *        BirArray = |--64 bits--| |--64 bits--| |--64 bits--| ...
 *                       word[0]       word[1]       word[2]
 */
template <size_t N> class BitArray {
  public:
    /// alias for the type used to represent a word (a 64 bit unsigned integer)
    using word_t = uint64_t;

    /// the number of bits in one word (8 * 8 = 64)
    static constexpr size_t bits_per_word = 8 * sizeof(word_t);

    /// this tests that a word has 64 bits
    static_assert(bits_per_word == 64, "The size of a word must be 64 bits");

    /// this tests that N is a multiple of 64
    static_assert(N % bits_per_word == 0, "The size of the BitArray (N) must be a multiple of 64");

    /// the number of words needed to store n bits
    static constexpr size_t bits_to_words(size_t n) noexcept {
        return n / bits_per_word + (n % bits_per_word != 0);
    }

    /// the number of words used to store the bits
    static constexpr size_t nwords_ = bits_to_words(N);

    /// @brief Alias for the container type
    using container_t = std::array<word_t, nwords_>;

    /// @brief Default constructor. The bits are uninitialized/indeterminate for performance.
    /// Use the static function BitArray::zero() if you need all bits cleared (set to 0).
    BitArray() = default;

    /// @brief Static method to create a BitArray object with all bits set to zero.
    /// Users should use this method instead of the default constructor.
    static BitArray zero() {
        BitArray b;
        b.clear();
        return b;
    }

    /// @brief A class to access the bits of a BitArray object as if they were a vector of bools
    class Proxy {
      public:
        constexpr Proxy(word_t& word, size_t index) noexcept : word_(word), mask_(maskbit(index)) {}

        // conversion to bool for read access
        constexpr operator bool() const noexcept { return word_ & mask_; }

        // assignment operator for write access
        Proxy& operator=(bool val) noexcept {
            word_ ^= (-val ^ word_) & mask_; // if-free implementation
            return *this;
        }

        // assignment operator for write access
        Proxy& operator=(const Proxy& other) noexcept {
            if (this != &other) { // check for self-assignment
                *this = static_cast<bool>(other);
            }
            return *this;
        }

        // swap function
        friend void swap(Proxy a, Proxy b) noexcept {
            bool temp = static_cast<bool>(a);
            a = static_cast<bool>(b);
            b = temp;
        }

      private:
        word_t& word_; // reference to the word where the bit is stored
        word_t mask_;  // mask for the bit
    };

    /// @brief Access the bits of a BitArray object as if they were a vector of bools
    /// @param index the index of the bit to access
    /// @return a Proxy object that can be used to read or write the bit
    Proxy operator[](size_t index) noexcept { return Proxy(getword(index), whichbit(index)); }

    /// @brief Access the bits of a BitArray object as if they were a vector of bools (const
    /// version)
    /// @param index the index of the bit to access
    /// @return the value of the bit
    constexpr bool operator[](size_t index) const noexcept { return get_bit(index); }

    class iterator {
      public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = Proxy;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type;

        iterator(typename container_t::iterator word_it, size_t index)
            : word_it_(word_it), index_(index) {}

        reference operator*() const noexcept { return Proxy(*word_it_, index_); }

        iterator& operator++() {
            ++index_;
            if (index_ == bits_per_word) {
                ++word_it_;
                index_ = 0;
            }
            return *this;
        }

        iterator operator++(int) {
            iterator copy = *this;
            ++(*this);
            return copy;
        }

        iterator& operator--() {
            if (index_ == 0) {
                --word_it_;
                index_ = bits_per_word - 1;
            } else {
                --index_;
            }
            return *this;
        }

        iterator operator--(int) {
            iterator copy = *this;
            --(*this);
            return copy;
        }

        bool operator==(const iterator& other) const noexcept {
            return (word_it_ == other.word_it_) and (index_ == other.index_);
        }

        bool operator!=(const iterator& other) const noexcept { return !(*this == other); }

      private:
        typename container_t::iterator word_it_;
        size_t index_;
    };

    /// @return An iterator to the first bit of the BitArray.
    iterator begin() noexcept { return iterator(words_.begin(), 0); }

    /// @return An iterator to one past the last bit of the BitArray.
    iterator end() noexcept { return iterator(words_.end(), 0); }

    /// @return The value of bit in position pos
    constexpr bool get_bit(size_t pos) const noexcept { return getword(pos) & maskbit(pos); }

    /// @brief Set the bit in position pos to the value val
    /// @param pos The position of the bit to set
    /// @param val The value to set the bit to
    void set_bit(size_t pos, bool val) noexcept {
        getword(pos) ^= (-val ^ getword(pos)) & maskbit(pos); // if-free implementation
    }

    /// @brief Get a word in position pos
    /// @param pos The position of the word to get
    /// @return The value of the word at position pos
    constexpr word_t get_word(size_t pos) const noexcept { return words_[pos]; }

    /// @brief Set a word in position pos
    /// @param pos The position of the word to set
    /// @param word The value to set the word to
    void set_word(size_t pos, word_t word) noexcept { words_[pos] = word; }

    /// @brief Return the number of bits
    /// @return The number of bits
    static constexpr size_t size() noexcept { return N; }

    /// @brief Set all bits (including unused) to zero.
    constexpr void clear() noexcept {
        if constexpr (N == 64) {
            words_[0] = word_t(0);
        } else if constexpr (N == 128) {
            words_[0] = word_t(0);
            words_[1] = word_t(0);
        } else {
            words_.fill(word_t(0));
        }
    }

    /// @brief Set the first n bits and clear the rest.
    /// This fills the half-open bit range [0, n). For example, fill_up_to(70) sets global bits
    /// 0 through 69, leaving bit 70 unset. If n == 0 all bits are cleared; if n >= nbits all bits
    /// are set.
    /// @param n the number of bits to set to 1, starting from the lowest bit index
    void fill_up_to(size_t n) noexcept {
        clear();
        if (n == 0) {
            return;
        }
        // find the index of the last word to fill completely with 1's
        const size_t end = std::min(static_cast<size_t>(n), N);
        const size_t full_words = whichword(end);
        for (size_t k = 0; k < full_words; ++k) {
            words_[k] = ~word_t(0);
        }
        const size_t bit_idx = whichbit(end);
        if (bit_idx != 0) {
            words_[full_words] = (word_t(1) << bit_idx) - word_t(1);
        }
    }

    /// @brief Flip all bits in the BitArray (including unused bits)
    constexpr void flip() noexcept {
        for (word_t& w : words_)
            w = ~w;
    }

    /// @brief Equality operator (==)
    /// @param lhs the left-hand side BitArray
    constexpr bool operator==(const BitArray<N>& lhs) const noexcept {
        return test_binary_condition(lhs, [](uint64_t a, uint64_t b) -> bool { return a == b; });
    }

    /// @brief Inequality operator (!=)
    /// @param lhs the left-hand side BitArray
    constexpr bool operator!=(const BitArray<N>& lhs) const noexcept {
        return test_binary_condition_any_of(lhs,
                                            [](uint64_t a, uint64_t b) -> bool { return a != b; });
    }

    /// @brief Negation operator (~)
    /// @return Flip all bits in the BitArray and return the result as a new BitArray
    constexpr BitArray<N> operator~() const noexcept {
        BitArray<N> res(*this);
        res.flip();
        return res;
    }

    /// @brief Less than operator (<)
    /// @param lhs the left-hand side BitArray
    constexpr bool operator<(const BitArray<N>& lhs) const noexcept {
        if constexpr (N == 64) {
            return (this->words_[0] < lhs.words_[0]);
        } else if constexpr (N == 128) {
            //  W1  W0  <
            //  >   >   F
            //  >   =   F
            //  >   <   F
            //  =   >   F
            //  =   =   F
            //  <   =   T
            //  <   >   T
            //  <   <   T
            //  =   <   T
            return (this->words_[1] < lhs.words_[1]) or
                   ((this->words_[1] == lhs.words_[1]) and (this->words_[0] < lhs.words_[0]));
        } else {
            for (size_t n = nwords_; n > 1;) {
                --n;
                if (this->words_[n] > lhs.words_[n])
                    return false;
                if (this->words_[n] < lhs.words_[n])
                    return true;
            }
            return this->words_[0] < lhs.words_[0];
        }
    }

    /// @brief Returns the bitwise OR operator (|) of this BitArray and another BitArray
    /// @param lhs the left-hand side BitArray
    /// @return a new BitArray that is the bitwise OR of this BitArray and the left-hand side
    /// BitArray
    constexpr BitArray<N> operator|(const BitArray<N>& lhs) const noexcept {
        BitArray<N> result;
        result.apply_ternary_operation(
            *this, lhs, [](uint64_t& a, uint64_t b, uint64_t c) -> uint64_t { return a = b | c; });
        return result;
    }

    /// @brief Modifies this BitArray by performing a bitwise OR operation with another BitArray
    /// @param lhs the left-hand side BitArray
    /// @return a reference to the modified BitArray
    BitArray<N>& operator|=(const BitArray<N>& lhs) noexcept {
        apply_binary_operation(lhs, [](uint64_t& a, uint64_t b) -> uint64_t { return a |= b; });
        return *this;
    }

    /// @brief Returns the bitwise XOR operator (^) of this BitArray and another BitArray
    /// @param lhs the left-hand side BitArray
    /// @return a new BitArray that is the bitwise XOR of this BitArray and the left-hand side
    /// BitArray
    constexpr BitArray<N> operator^(const BitArray<N>& lhs) const noexcept {
        BitArray<N> result;
        result.apply_ternary_operation(
            *this, lhs, [](uint64_t& a, uint64_t b, uint64_t c) -> uint64_t { return a = b ^ c; });
        return result;
    }

    /// @brief Modifies this BitArray by performing a bitwise XOR operation with another BitArray
    /// @param lhs the left-hand side BitArray
    /// @return a reference to the modified BitArray
    BitArray<N>& operator^=(const BitArray<N>& lhs) noexcept {
        apply_binary_operation(lhs, [](uint64_t& a, uint64_t b) -> uint64_t { return a ^= b; });
        return *this;
    }

    /// @brief Returns the bitwise plus without carrying operator (+) of this BitArray and another
    /// BitArray
    /// @param lhs the left-hand side BitArray
    /// @return a new BitArray that is the bitwise plus of this BitArray and the left-hand side
    /// BitArray
    constexpr BitArray<N> operator+(const BitArray<N>& lhs) const noexcept {
        BitArray<N> result;
        result.apply_ternary_operation(
            *this, lhs, [](uint64_t& a, uint64_t b, uint64_t c) -> uint64_t { return a = b ^ c; });
        return result;
    }

    /// @brief Returns the bitwise AND operator (&) of this BitArray and another BitArray
    /// @param lhs the left-hand side BitArray
    /// @return a new BitArray that is the bitwise AND of this BitArray and the left-hand side
    /// BitArray
    constexpr BitArray<N> operator&(const BitArray<N>& lhs) const noexcept {
        BitArray<N> result;
        result.apply_ternary_operation(
            *this, lhs, [](uint64_t& a, uint64_t b, uint64_t c) -> uint64_t { return a = b & c; });
        return result;
    }

    /// @brief Modifies this BitArray by performing a bitwise AND operation with another BitArray
    /// @param lhs the left-hand side BitArray
    /// @return a reference to the modified BitArray
    BitArray<N>& operator&=(const BitArray<N>& lhs) noexcept {
        apply_binary_operation(lhs, [](uint64_t& a, uint64_t b) -> uint64_t { return a &= b; });
        return *this;
    }

    /// @brief Returns the bitwise difference operator (-) of this BitArray and another BitArray
    /// @param lhs the left-hand side BitArray
    /// @return a new BitArray that is the bitwise difference of this BitArray and the left-hand
    /// side BitArray
    constexpr BitArray<N> operator-(const BitArray<N>& lhs) const noexcept {
        BitArray<N> result;
        result.apply_ternary_operation(
            *this, lhs,
            [](uint64_t& a, uint64_t b, uint64_t c) -> uint64_t { return a = b & (~c); });
        return result;
    }

    /// @brief Modifies this BitArray by performing a bitwise difference operation with another
    /// BitArray
    /// @param lhs the left-hand side BitArray
    /// @return a reference to the modified BitArray
    BitArray<N>& operator-=(const BitArray<N>& lhs) noexcept {
        apply_binary_operation(lhs, [](uint64_t& a, uint64_t b) -> uint64_t { return a &= ~b; });
        return *this;
    }

    /// @brief Count the number of bits set to 1 (true) in the word range [begin, end).
    /// begin and end are word indices, not bit indices.
    /// @return the number of set bits
    constexpr size_t count(size_t begin = 0, size_t end = nwords_) const noexcept {
        size_t c = 0;
        for (; begin < end; ++begin) {
            c += std::popcount(this->words_[begin]);
        }
        return c;
    }

    /// @brief Count the number of bits set to 1 (true) in the entire BitArray. This is an optimized
    /// version of count() that uses constexpr to compile only the relevant code for the specific
    /// size of the BitArray. For example, if N == 128, it compiles only the code that counts the
    /// bits in words_[0] and words_[1], skipping the loop that would be used for larger sizes.
    /// @return the number of set bits
    constexpr size_t count_all() const noexcept {
        // with constexpr we compile only one of these cases
        if constexpr (N == 128) {
            return std::popcount(words_[0]) + std::popcount(words_[1]);
        } else if constexpr (N == 256) {
            return std::popcount(words_[0]) + std::popcount(words_[1]) + std::popcount(words_[2]) +
                   std::popcount(words_[3]);
        } else {
            size_t c{0};
            for (const auto& w : words_) {
                c += std::popcount(w);
            }
            return c;
        }
    }

    /// @brief Find the first bit set to one in the word range [begin, end), starting from the
    /// lowest bit index. begin and end are word indices, not bit indices.
    /// @return the index of the the first bit, or if all bits are zero, returns ~0
    uint64_t find_first_one(size_t begin = 0, size_t end = nwords_) const noexcept {
        for (; begin < end; ++begin) {
            // find the first word != 0
            if (words_[begin] != word_t(0)) {
                return ui64_find_lowest_one_bit(words_[begin]) + begin * bits_per_word;
            }
        }
        return ui64_bit_not_found;
    }

    /// @brief Find the last bit set to one in the word range [begin, end).
    /// begin and end are word indices, not bit indices.
    /// @return the index of the last bit, or if all bits are zero, returns ~0
    uint64_t find_last_one(size_t begin = 0, size_t end = nwords_) const noexcept {
        for (; begin < end; --end) {
            const size_t word_idx = end - 1;
            if (words_[word_idx] != word_t(0)) {
                return ui64_find_highest_one_bit(words_[word_idx]) + word_idx * bits_per_word;
            }
        }
        return ui64_bit_not_found;
    }

    /// @brief Apply a callable to each bit set to one, in ascending bit-index order.
    /// The callable may return either void or a bool-like value. Void callables always continue.
    /// Bool callables continue when they return true and stop early when they return false.
    /// @param func a callable that accepts the bit index as a size_t
    /// @param begin the index of the first word to test
    /// @param end the index of the last word to test (not included)
    /// @return true if all selected bits were visited, false if the callback stopped the traversal
    template <typename Func>
    bool for_each_set_bit(Func&& func, size_t begin = 0, size_t end = nwords_) const {
        for (; begin < end; ++begin) {
            uint64_t x = words_[begin];
            const size_t base = begin * bits_per_word;
            while (x) {
                const size_t pos = base + std::countr_zero(x);
                // if the callback returns void, we ignore the return value and always continue
                if constexpr (std::is_void_v<std::invoke_result_t<Func, size_t>>) {
                    std::invoke(std::forward<Func>(func), pos);
                }
                // if the callback returns a bool-like value, we check it
                else {
                    if (!std::invoke(std::forward<Func>(func), pos)) {
                        return false;
                    }
                }
                x &= (x - 1); // clear lowest set bit
            }
        }
        return true;
    }

    /// @brief Find all the bits set to one and store their indices in the vector occ
    /// @param occ a vector of integers where the indices of the bits set to one are stored
    /// @param n the number of bits set to one
    /// @param begin the index of the first word to test
    /// @param end the index of the last word to test (not included)
    void find_set_bits(std::vector<size_t>& occ, size_t& n, size_t begin = 0,
                       size_t end = nwords_) const {
        n = 0;
        for_each_set_bit([&](size_t pos) { occ[n++] = pos; }, begin, end);
    }

    /// @brief Test (a & b) == b
    /// This checks if this BitArray is a superset of b, i.e. if all bits set in b are also set in
    /// this BitArray.
    /// @param b the second BitArray object
    /// @return true if this BitArray is a superset of b, false otherwise
    bool is_superset_of(const BitArray<N>& b) const {
        return test_binary_condition(b,
                                     [](uint64_t a, uint64_t b) -> bool { return (a & b) == b; });
    }

    /// @brief Test a & b == 0
    /// This checks if this BitArray is disjoint from b, i.e. if no bits set in b are also set in
    /// this BitArray.
    /// @param b the second BitArray object
    /// @return true if this BitArray is disjoint from b, false otherwise
    bool is_disjoint_from(const BitArray<N>& b) const {
        return test_binary_condition(b,
                                     [](uint64_t a, uint64_t b) -> bool { return (a & b) == 0; });
    }

    /// @brief Implements count(a & b)
    /// This counts the number of bits set to 1 in the intersection of this BitArray and b, i.e. the
    /// number of bits that are set in both this BitArray and b.
    /// @param b the second BitArray object
    /// @return the number of bits set to 1 in the intersection of this BitArray and b
    size_t intersection_count(const BitArray<N>& b) const {
        return count_binary_operation(
            b, [](uint64_t a, uint64_t b) -> size_t { return std::popcount(a & b); });
    }

    /// @brief Implements count(a ^ b)
    /// This counts the number of bits set to 1 in the symmetric difference of this BitArray and b,
    /// i.e. the number of bits that are set in either this BitArray or b, but not in both.
    /// @param b the second BitArray object
    /// @return the number of bits set to 1 in the symmetric difference of this BitArray and b
    size_t symmetric_difference_count(const BitArray<N>& b) const {
        return count_binary_operation(
            b, [](uint64_t a, uint64_t b) -> size_t { return std::popcount(a ^ b); });
    }

    /// @brief Implements a ^ b
    /// This returns a new BitArray that is the symmetric difference of this BitArray and b, i.e. a
    /// new BitArray that has bits set to 1 in the positions where this BitArray and b differ, and
    /// bits set to 0 in the positions where this BitArray and b are the same
    /// @param b the second BitArray object
    /// @return a new BitArray containing the symmetric difference between this BitArray and b
    BitArray<N> symmetric_difference(const BitArray<N>& b) const {
        BitArray<N> result{*this};
        result.apply_binary_operation(b,
                                      [](uint64_t& a, uint64_t b) -> uint64_t { return a ^= b; });
        return result;
    }

    /// @brief Computes the union of this BitArray and b, modifying this BitArray in place.
    /// @param b the second BitArray object
    /// @return a reference to this BitArray
    BitArray<N>& union_with(const BitArray<N>& b) {
        apply_binary_operation(b, [](uint64_t& a, uint64_t b) -> uint64_t { return a |= b; });
        return *this;
    }

    /// @brief Returns the intersection of this BitArray and b as a new BitArray.
    /// @param b the second BitArray object
    /// @return a new BitArray containing the intersection of this BitArray and b
    BitArray<N> intersection(const BitArray<N>& b) const {
        BitArray<N> result{*this};
        result.apply_binary_operation(b,
                                      [](uint64_t& a, uint64_t b) -> uint64_t { return a &= b; });
        return result;
    }

    /// @brief Test (a & b & ~c) | (c & ~a) == 0 used to check if an operator can be applied to a
    ///        determinant. In this context:
    ///        - a is the determinant itself
    ///        - b is the creation operator
    ///        - c is the annihilation operator
    /// @return true if the operator can be applied, false otherwise
    inline bool can_apply_operator(const BitArray<N>& b, const BitArray<N>& c) const {
        return test_ternary_condition(b, c, [](uint64_t a, uint64_t b, uint64_t c) -> bool {
            return ((a & b & (~c)) | (c & (~a))) == 0;
        });
    }

    /// @brief Test a & (b - c) == 0. This checks if this BitArray is disjoint from the difference
    /// of b and c, i.e. if no bits set in b and not set in c are also set in this BitArray.
    /// In grouped operator application this checks whether any creation target in `b`, excluding
    /// orbitals also annihilated by `c`, is already occupied in this determinant.
    bool disjoint_from_difference(const BitArray<N>& b, const BitArray<N>& c) const {
        return test_ternary_condition(
            b, c, [](uint64_t a, uint64_t b, uint64_t c) -> bool { return (a & (b & (~c))) == 0; });
    }

    /// @brief Return the fermionic sign corresponding to orbital n
    /// This function ignores if bit n is set or not
    /// @param n the orbital index
    /// @return the fermionic sign
    double slater_sign(size_t n) const {
        // Optimized version for 64 bits (skips counting the bits in the previous words)
        if (n < bits_per_word) {
            return ui64_sign(words_[0], n);
        }
        // parity for the word containing bit n x the parity of the previous words
        return parity_to_sign(count(0, whichword(n))) * ui64_sign(getword(n), whichbit(n));
    }

    /// @brief Return the fermionic sign corresponding to orbital n in reverse order
    /// This function ignores if bit n is set or not
    /// @param n the orbital index
    /// @return the fermionic sign
    double slater_sign_reverse(size_t n) const {
        if constexpr (N == bits_per_word) {
            return ui64_sign_reverse(words_[0], n);
        } else {
            size_t count = 0;
            // count all the following bits only if we are not looking at the last word
            size_t start_word =
                whichword(n) + 1; // Start from the word following the one containing bit n
            if (start_word < nwords_) {
                for (size_t k = start_word; k < nwords_; ++k) {
                    count += std::popcount(words_[k]);
                }
            }
            const double word_sign = ui64_sign_reverse(getword(n), whichbit(n));
            return parity_to_sign(count) * word_sign;
        }
    }

    /// @brief Return the fermionic sign for the orbitals in between n and m
    /// The sign depends only on the number of bits = 1 between n and m
    /// There are no restrictions on n and m
    /// @param n the first orbital index
    /// @param m the second orbital index
    /// @return the fermionic sign
    double slater_sign(size_t n, size_t m) const {
        if constexpr (N == 64) {
            return ui64_sign(words_[0], n, m);
        } else if constexpr (N == 128) {
            // XXXXXXXX YYYYYYYY
            // XmXXXXnX YYYYYYYY (case 1)
            //   cccc
            // XmXXXXnX YYYYYYYY (case 1)
            //   cccccc
            // cccccc
            // XmXXXXXX YYYYYnYY (case 2)
            //   cccccc ccccc
            // let's first order the numbers so that m <= n
            if (n < m)
                std::swap(m, n);
            size_t word_m = whichword(m);
            size_t word_n = whichword(n);
            // if both bits are in the same word use an optimized version
            if (word_n == word_m) {
                return ui64_sign(words_[word_n], whichbit(n), whichbit(m));
            }
            // count the bits after m in word[m]
            // count the bits before n in word[n]
            return ui64_sign_reverse(words_[word_m], whichbit(m)) *
                   ui64_sign(words_[word_n], whichbit(n));
        } else {
            // let's first order the numbers so that m <= n
            if (n < m)
                std::swap(m, n);
            size_t word_m = whichword(m);
            size_t word_n = whichword(n);
            // if both bits are in the same word use an optimized version
            if (word_n == word_m) {
                return ui64_sign(words_[word_n], whichbit(n), whichbit(m));
            }
            size_t count = 0;
            // count the number of bits in bitween the words of m and n
            for (size_t k = word_m + 1; k < word_n; ++k) {
                count += std::popcount(words_[k]);
            }
            // count the bits after m in word[m]
            // count the bits before n in word[n]
            double sign = ui64_sign_reverse(words_[word_m], whichbit(m)) *
                          ui64_sign(words_[word_n], whichbit(n));
            return (count % 2 == 0) ? sign : -sign;
        }
    }

    /// @brief Find the irreducible representation of a product of spin orbitals
    /// @param irrep a vector of irrep values
    /// @return the irrep
    int symmetry(const std::vector<int>& irrep) const {
        int sym = 0;
        for_each_set_bit([&](size_t pos) { sym ^= irrep[pos]; });
        return sym;
    }

    /// @brief Return a string representation of the BitArray of the form |010101...> where the bits
    /// are ordered from left to right in ascending order of their index.
    /// @param n the number of bits to display
    /// @return a string representation of the BitArray
    std::string str(size_t n = BitArray<N>::size()) const {
        std::string s;
        s += "|";
        for (size_t p = 0; p < n; ++p) {
            if (get_bit(p)) {
                s += "1";
            } else {
                s += "0";
            }
        }
        s += ">";
        return s;
    }

    struct Hash {
        /// @brief Returns a hash value for a BitArray object
        /// @param d the BitArray object to hash
        /// @return a hash value for the BitArray object
        std::size_t operator()(const BitArray<N>& b) const noexcept {
            if constexpr (N == 64) {
                return b.words_[0];
            } else if constexpr (N == 128) {
                return hash_combine(b.words_[0], b.words_[1]);
            } else {
                std::size_t seed = 0;
                for (auto& w : b.words_) {
                    seed = hash_combine(seed, w);
                }
                return seed;
            }
        }
    };

  private:
    /// @brief This templated function is used to generate tests for binary word conditions.
    /// @param b the second BitArray object
    /// @param condition a lambda function that takes two uint64_t integers and returns a boolean
    /// @return true if the condition is satisfied for all the words, false otherwise
    template <typename Condition>
    bool test_binary_condition(const BitArray<N>& b, Condition&& condition) const {
        for (size_t n = 0; n < words_.size(); n++) {
            if (!std::invoke(std::forward<Condition>(condition), words_[n], b.words_[n]))
                return false;
        }
        return true;
    }

    /// @brief This templated function is used to generate tests if a binary word condition is
    /// satisfied for any of the words.
    /// @param b the second BitArray object
    /// @param condition a lambda function that takes two uint64_t integers and returns a boolean
    /// @return true if the condition is satisfied for any the words, false otherwise
    template <typename Condition>
    bool test_binary_condition_any_of(const BitArray<N>& b, Condition&& condition) const {
        for (size_t n = 0; n < words_.size(); n++) {
            if (std::invoke(std::forward<Condition>(condition), words_[n], b.words_[n]))
                return true;
        }
        return false;
    }

    /// @brief This templated function is used to generate tests for ternary word conditions.
    /// @param b the second BitArray object
    /// @param c the third BitArray object
    /// @param condition a lambda function that takes three uint64_t integers and returns a boolean
    /// @return true if the condition is satisfied for all the words, false otherwise
    template <typename Condition>
    bool test_ternary_condition(const BitArray<N>& b, const BitArray<N>& c,
                                Condition&& condition) const {
        for (size_t n = 0; n < words_.size(); n++) {
            if (!std::invoke(std::forward<Condition>(condition), words_[n], b.words_[n],
                             c.words_[n]))
                return false;
        }
        return true;
    }

    /// @brief This templated function is used to generate counting binary word operations.
    /// @param b the second BitArray object
    /// @param operation a lambda function that takes two uint64_t integers and returns an integer
    /// @return the result of the operation for all the words
    template <typename Operation>
    size_t count_binary_operation(const BitArray<N>& b, Operation&& operation) const {
        size_t c = 0;
        for (size_t n = 0; n < nwords_; n++) {
            c += std::invoke(std::forward<Operation>(operation), words_[n], b.words_[n]);
        }
        return c;
    }

    /// @brief This templated function is used to generate binary word operations that modify the
    /// current BitArray.
    /// @param b the second BitArray object
    /// @param operation a lambda function that takes two uint64_t integers and modifies the first
    template <typename Operation>
    void apply_binary_operation(const BitArray<N>& b, Operation&& operation) {
        for (size_t n = 0; n < nwords_; n++) {
            std::invoke(std::forward<Operation>(operation), words_[n], b.words_[n]);
        }
    }

    /// @brief  This templated function is used to generate ternary word operations that modify the
    /// current BitArray.
    /// @tparam Operation
    /// @param b the second BitArray object
    /// @param c the third BitArray object
    /// @param operation a lambda function that takes three uint64_t integers and modifies the first
    template <typename Operation>
    void apply_ternary_operation(const BitArray<N>& b, const BitArray<N>& c,
                                 Operation&& operation) {
        for (size_t n = 0; n < nwords_; n++) {
            std::invoke(std::forward<Operation>(operation), words_[n], b.words_[n], c.words_[n]);
        }
    }

  protected:
    // ==> Protected Functions <==

    // These functions are used to address bits in the BitArray.
    // Derived classes use them for word-level specializations, but they are not part of the public
    // BitArray API.

    /// @return the index of the word where the bit in position pos is found
    static constexpr size_t whichword(size_t pos) noexcept { return pos / bits_per_word; }

    /// @return the index of a bit within a word
    static constexpr size_t whichbit(size_t pos) noexcept { return pos % bits_per_word; }

    /// @return a mask for the bit in position pos
    static constexpr word_t maskbit(size_t pos) noexcept {
        return (static_cast<word_t>(1)) << whichbit(pos);
    }

    /// @return a reference to the word where the bit in position pos is found
    constexpr word_t& getword(size_t pos) noexcept { return words_[whichword(pos)]; }

    /// @return a const reference to the word where the bit in position pos is found
    constexpr const word_t& getword(size_t pos) const noexcept { return words_[whichword(pos)]; }

    // ==> Protected Data <==

    /// @brief The bits stored as a vector of words uninitialized/indeterminate for performance.
    container_t words_;
};

/// @brief Print a BitArray object to an output stream
/// @param os the output stream to print to
/// @param ba the BitArray object to print
/// @return a reference to the output stream
template <size_t N> std::ostream& operator<<(std::ostream& os, const BitArray<N>& ba) {
    os << str(ba);
    return os;
}

/// @brief Check if two bit arrays differ by at most 4 bits, and if they have the same number of
/// bits set to 1. This is used to screen the Slater connections between two determinants, which can
/// only be connected if they differ by at most 4 spinors and have the same number of electrons.
/// @tparam start the index of the first word to compare
/// @tparam end the index of the last word to compare (not included)
/// @tparam N the number of bits in the BitArray
/// @param lhs the left bit array
/// @param rhs the right bit array
/// @return an optional containing the number of differences if the two bit arrays differ by at most
/// 4 bits and have the same number of bits set to 1, or an empty optional otherwise
template <size_t start, size_t end, size_t N>
/// @details We use a 32 bit integer to store the number of differences so that the returned
/// quantity fits in a register.
std::optional<std::uint32_t> screen_slater_connection_impl(const forte2::BitArray<N>& lhs,
                                                           const forte2::BitArray<N>& rhs) {
    int diff_count{0}; // number of spinors that differ between the two determinants
    for (std::size_t w = start; w < end; ++w) {
        diff_count += std::popcount(lhs.get_word(w) ^ rhs.get_word(w));
        // early exit if more than 4 differences
        if (diff_count > 4) {
            return std::optional<std::uint32_t>();
        }
    }
    int total_set_diff_count{0}; //  number of lhs - rhs set bits
    for (std::size_t w = start; w < end; ++w) {
        total_set_diff_count += std::popcount(lhs.get_word(w)) - std::popcount(rhs.get_word(w));
    }
    // signal early exit if the number of set bits is different
    if (total_set_diff_count != 0) {
        return std::optional<std::uint32_t>();
    }
    return diff_count;
}

/// @brief Find the single connection between two bit arrays, which must differ by exactly by one
/// replacement.
/// @tparam start the index of the first word to compare
/// @tparam end the index of the last word to compare (not included)
/// @tparam N the number of bits in the BitArray
/// @param lhs the left determinant
/// @param rhs the right determinant
/// @return a tuple containing the difference in bit positions: (i, a) where i is set in the lhs and
/// not in the rhs, and a is set in the rhs and not in the lhs
/// The indices are shifted by start * bits_per_word to account for the fact that we are only
/// comparing a subset of the words of the BitArrays.
template <size_t start, size_t end, size_t N>
std::tuple<std::size_t, std::size_t> find_single_connection_impl(const forte2::BitArray<N>& lhs,
                                                                 const forte2::BitArray<N>& rhs) {
    std::size_t i, a; // namespace
    for (std::size_t w = start; w < end; ++w) {
        const std::uint64_t lhs_word = lhs.get_word(w);
        const std::uint64_t rhs_word = rhs.get_word(w);
        const std::uint64_t lhs_only = lhs_word & ~rhs_word;
        const std::uint64_t rhs_only = rhs_word & ~lhs_word;
        if (lhs_only) {
            i = (w - start) * forte2::BitArray<N>::bits_per_word + std::countr_zero(lhs_only);
        }
        if (rhs_only) {
            a = (w - start) * forte2::BitArray<N>::bits_per_word + std::countr_zero(rhs_only);
        }
    }
    return {i, a};
}

/// @brief Find the double connection between two bit arrays, which must differ by exactly two
/// replacements.
/// @tparam start the index of the first word to compare
/// @tparam end the index of the last word to compare (not included)
/// @tparam N the number of bits in the BitArray
/// @param lhs the left determinant
/// @param rhs the right determinant
/// @return a tuple containing the indices of the differences in bit positions: (i, j, a, b) where i
/// and j are set in the lhs and not in the rhs, and a and b are set in the rhs and not in the lhs.
/// The indices are shifted by start * bits_per_word to account for the fact that we are only
/// comparing a subset of the words of the BitArrays.
template <size_t start, size_t end, size_t N>
std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
find_double_connection_impl(const forte2::BitArray<N>& lhs, const forte2::BitArray<N>& rhs) {
    constexpr std::size_t not_filled = std::numeric_limits<size_t>::max();
    std::size_t i{not_filled}, j, a{not_filled}, b; // mark i and j as not filled
    for (std::size_t w = start; w < end; ++w) {
        const std::uint64_t lhs_word = lhs.get_word(w);
        const std::uint64_t rhs_word = rhs.get_word(w);
        std::uint64_t lhs_only = lhs_word & ~rhs_word;
        std::uint64_t rhs_only = rhs_word & ~lhs_word;
        while (lhs_only) { // loop over the bits set in lhs_only
            if (i == not_filled) {
                i = (w - start) * forte2::BitArray<N>::bits_per_word + std::countr_zero(lhs_only);
            } else {
                j = (w - start) * forte2::BitArray<N>::bits_per_word + std::countr_zero(lhs_only);
            }
            ui64_clear_lowest_one_bit(lhs_only); // Clear the lowest set bit
        }
        while (rhs_only) { // loop over the bits set in rhs_only
            if (a == not_filled) {
                a = (w - start) * forte2::BitArray<N>::bits_per_word + std::countr_zero(rhs_only);
            } else {
                b = (w - start) * forte2::BitArray<N>::bits_per_word + std::countr_zero(rhs_only);
            }
            ui64_clear_lowest_one_bit(rhs_only); // Clear the lowest set bit
        }
    }
    return {i, j, a, b};
}

} // namespace forte2
