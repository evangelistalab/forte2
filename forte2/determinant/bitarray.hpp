#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ostream>
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

    /// the total number of bits (must be a multiple of 64)
    static constexpr size_t nbits = N;

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
    static constexpr size_t nwords_ = bits_to_words(nbits);

    using container_t = std::array<word_t, nwords_>;

    /// @brief Default constructor. The bits are uninitialized/indeterminate for performance.
    /// Use the static function BitArray::zero() if you need all bits cleared (set to 0).
    BitArray() = default;

    BitArray(const std::vector<bool>& v) {
        if (v.size() > nbits) {
            throw std::invalid_argument("BitArray input vector is larger than the bit array size.");
        }
        clear();
        for (size_t i = 0; const auto b : v) {
            set_bit(i, b);
            ++i;
        }
    }

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

    iterator begin() noexcept { return iterator(words_.begin(), 0); }
    iterator end() noexcept { return iterator(words_.end(), 0); }

    // get the value of bit in position pos
    constexpr bool get_bit(size_t pos) const noexcept { return getword(pos) & maskbit(pos); }

    /// set bit in position pos to the value val
    void set_bit(size_t pos, bool val) noexcept {
        getword(pos) ^= (-val ^ getword(pos)) & maskbit(pos); // if-free implementation
    }

    /// get a word in position pos
    constexpr word_t get_word(size_t pos) const noexcept { return words_[pos]; }

    /// set a word in position pos
    void set_word(size_t pos, word_t word) noexcept { words_[pos] = word; }

    /// return the number of bits
    constexpr size_t get_nbits() const noexcept { return nbits; }

    /// set all bits (including unused) to zero
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
    ///
    /// This fills the half-open bit range [0, n). For example, fill_up_to(70) sets global bits
    /// 0 through 69, leaving bit 70 unset. If n <= 0 all bits are cleared; if n >= nbits all bits
    /// are set.
    void fill_up_to(size_t n) noexcept {
        clear();
        if (n <= 0) {
            return;
        }
        const size_t end = static_cast<size_t>(n);
        if (end >= nbits) {
            words_.fill(~word_t(0));
            return;
        }

        const size_t full_words = whichword(end);
        for (size_t k = 0; k < full_words; ++k) {
            words_[k] = ~word_t(0);
        }

        const size_t bit_idx = whichbit(end);
        if (bit_idx != 0) {
            words_[full_words] = (word_t(1) << bit_idx) - word_t(1);
        }
    }

    /// @brief XOR the bits up to the nth bit
    /// @param n
    void xor_up_to(size_t n) noexcept {
        if (n == 0)
            return; // Nothing to XOR if n == 0

        size_t word_idx = whichword(n); // Identify the word containing the nth bit
        size_t bit_idx = whichbit(n);   // Position of the bit within the word

        // Handle full words up to the current word
        for (size_t k = 0; k < word_idx; ++k) {
            words_[k] ^= ~uint64_t(0); // XOR with all 1s
        }

        // Handle the remaining bits in the last word
        if (bit_idx != 0) {
            uint64_t mask = ~uint64_t(0);  // Start with all bits set to 1
            mask = mask >> (64 - bit_idx); // Create a mask up to the nth bit
            words_[word_idx] ^= mask;
        }
    }

    /// flip all bits
    constexpr void flip() noexcept {
        for (word_t& w : words_)
            w = ~w;
    }

    /// equal operator
    constexpr bool operator==(const BitArray<N>& lhs) const noexcept {
        if constexpr (N == 64) {
            return (this->words_[0] == lhs.words_[0]);
        } else if constexpr (N == 128) {
            return ((this->words_[0] == lhs.words_[0]) and (this->words_[1] == lhs.words_[1]));
        } else if constexpr (N == 192) {
            return ((this->words_[0] == lhs.words_[0]) and (this->words_[1] == lhs.words_[1]) and
                    (this->words_[2] == lhs.words_[2]));
        } else if constexpr (N == 256) {
            return ((this->words_[0] == lhs.words_[0]) and (this->words_[1] == lhs.words_[1]) and
                    (this->words_[2] == lhs.words_[2]) and (this->words_[3] == lhs.words_[3]));
        } else {
            for (size_t n = 0; n < nwords_; ++n) {
                if (this->words_[n] != lhs.words_[n])
                    return false;
            }
            return true;
        }
    }

    /// not equal operator
    constexpr bool operator!=(const BitArray<N>& lhs) const noexcept {
        if constexpr (N == 64) {
            return (this->words_[0] != lhs.words_[0]);
        } else if constexpr (N == 128) {
            return ((this->words_[0] != lhs.words_[0]) or (this->words_[1] != lhs.words_[1]));
        } else if constexpr (N == 192) {
            return ((this->words_[0] != lhs.words_[0]) or (this->words_[1] != lhs.words_[1]) or
                    (this->words_[2] != lhs.words_[2]));
        } else if constexpr (N == 256) {
            return ((this->words_[0] != lhs.words_[0]) or (this->words_[1] != lhs.words_[1]) or
                    (this->words_[2] != lhs.words_[2]) or (this->words_[3] != lhs.words_[3]));
        } else {
            for (size_t n = 0; n < nwords_; ++n) {
                if (this->words_[n] != lhs.words_[n])
                    return true;
            }
            return false;
        }
    }

    /// not operator
    constexpr BitArray<N> operator~() const noexcept {
        BitArray<N> res(*this);
        res.flip();
        return res;
    }

    /// Less than operator
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

    /// Bitwise OR operator (|)
    constexpr BitArray<N> operator|(const BitArray<N>& lhs) const noexcept {
        BitArray<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = words_[n] | lhs.words_[n];
        }
        return result;
    }

    /// Bitwise OR operator (|=)
    BitArray<N>& operator|=(const BitArray<N>& lhs) noexcept {
        for (size_t n = 0; n < nwords_; n++) {
            words_[n] |= lhs.words_[n];
        }
        return *this;
    }

    /// Bitwise XOR operator (^)
    constexpr BitArray<N> operator^(const BitArray<N>& lhs) const noexcept {
        BitArray<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = words_[n] ^ lhs.words_[n];
        }
        return result;
    }

    /// Bitwise XOR operator (^=)
    BitArray<N>& operator^=(const BitArray<N>& lhs) noexcept {
        for (size_t n = 0; n < nwords_; n++) {
            words_[n] ^= lhs.words_[n];
        }
        return *this;
    }

    /// Bitwise plus without carrying operator (+)
    constexpr BitArray<N> operator+(const BitArray<N>& lhs) const noexcept {
        BitArray<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = words_[n] ^ lhs.words_[n];
        }
        return result;
    }

    /// Bitwise AND operator (&)
    constexpr BitArray<N> operator&(const BitArray<N>& lhs) const noexcept {
        BitArray<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = words_[n] & lhs.words_[n];
        }
        return result;
    }

    /// Bitwise AND operator (&=)
    BitArray<N>& operator&=(const BitArray<N>& lhs) noexcept {
        for (size_t n = 0; n < nwords_; n++) {
            words_[n] &= lhs.words_[n];
        }
        return *this;
    }

    /// Bitwise difference operator (-)
    constexpr BitArray<N> operator-(const BitArray<N>& lhs) const noexcept {
        BitArray<N> result;
        for (size_t n = 0; n < nwords_; n++) {
            result.words_[n] = words_[n] & (~lhs.words_[n]);
        }
        return result;
    }

    /// Bitwise difference operator (-=)
    BitArray<N>& operator-=(const BitArray<N>& lhs) noexcept {
        for (size_t n = 0; n < nwords_; n++) {
            words_[n] &= ~lhs.words_[n];
        }
        return *this;
    }

    /// Count the number of set bits in the word range [begin, end).
    /// begin and end are word indices, not bit indices.
    constexpr size_t count(size_t begin = 0, size_t end = nwords_) const noexcept {
        size_t c = 0;
        for (; begin < end; ++begin) {
            c += std::popcount(this->words_[begin]);
        }
        return c;
    }

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

    /// Find the first bit set to one in the word range [begin, end), starting from the lowest bit
    /// index. begin and end are word indices, not bit indices.
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

    /// Find the last bit set to one in the word range [begin, end).
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

    /// Apply a callable to each bit set to one, in ascending bit-index order.
    ///
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

    /// Find all the bits set to one and store their indices in the vector occ
    /// @param occ a vector of integers where the indices of the bits set to one are stored
    /// @param n the number of bits set to one
    /// @param begin the index of the first word to test
    /// @param end the index of the last word to test (not included)
    void find_set_bits(std::vector<size_t>& occ, size_t& n, size_t begin = 0,
                       size_t end = nwords_) const {
        n = 0;
        for_each_set_bit([&](size_t pos) { occ[n++] = pos; }, begin, end);
    }

  private:
    /// @brief This templated function is used to generate tests for binary word conditions.
    /// @param b the second BitArray object
    /// @param condition a lambda function that takes two uint64_t integers and returns a boolean
    /// @return true if the condition is satisfied for all the words, false otherwise
    template <typename Condition>
    bool test_binary_condition(const BitArray<N>& b, Condition&& condition) const {
        if constexpr (N == 64) {
            return std::invoke(std::forward<Condition>(condition), words_[0], b.words_[0]);
        } else if constexpr (N == 128) {
            return std::invoke(std::forward<Condition>(condition), words_[0], b.words_[0]) &&
                   std::invoke(std::forward<Condition>(condition), words_[1], b.words_[1]);
        } else if constexpr (N == 192) {
            return std::invoke(std::forward<Condition>(condition), words_[0], b.words_[0]) &&
                   std::invoke(std::forward<Condition>(condition), words_[1], b.words_[1]) &&
                   std::invoke(std::forward<Condition>(condition), words_[2], b.words_[2]);
        } else if constexpr (N == 256) {
            return std::invoke(std::forward<Condition>(condition), words_[0], b.words_[0]) &&
                   std::invoke(std::forward<Condition>(condition), words_[1], b.words_[1]) &&
                   std::invoke(std::forward<Condition>(condition), words_[2], b.words_[2]) &&
                   std::invoke(std::forward<Condition>(condition), words_[3], b.words_[3]);
        } else {
            for (size_t n = 0; n < words_.size(); n++) {
                if (!std::invoke(std::forward<Condition>(condition), words_[n], b.words_[n]))
                    return false;
            }
            return true;
        }
    }

    /// @brief This templated function is used to generate tests for ternary word conditions.
    /// @param b the second BitArray object
    /// @param c the third BitArray object
    /// @param condition a lambda function that takes three uint64_t integers and returns a boolean
    /// @return true if the condition is satisfied for all the words, false otherwise
    template <typename Condition>
    bool test_ternary_condition(const BitArray<N>& b, const BitArray<N>& c,
                                Condition&& condition) const {
        if constexpr (N == 64) {
            return std::invoke(std::forward<Condition>(condition), words_[0], b.words_[0],
                               c.words_[0]);
        } else if constexpr (N == 128) {
            return std::invoke(std::forward<Condition>(condition), words_[0], b.words_[0],
                               c.words_[0]) &&
                   std::invoke(std::forward<Condition>(condition), words_[1], b.words_[1],
                               c.words_[1]);
        } else if constexpr (N == 192) {
            return std::invoke(std::forward<Condition>(condition), words_[0], b.words_[0],
                               c.words_[0]) &&
                   std::invoke(std::forward<Condition>(condition), words_[1], b.words_[1],
                               c.words_[1]) &&
                   std::invoke(std::forward<Condition>(condition), words_[2], b.words_[2],
                               c.words_[2]);
        } else if constexpr (N == 256) {
            return std::invoke(std::forward<Condition>(condition), words_[0], b.words_[0],
                               c.words_[0]) &&
                   std::invoke(std::forward<Condition>(condition), words_[1], b.words_[1],
                               c.words_[1]) &&
                   std::invoke(std::forward<Condition>(condition), words_[2], b.words_[2],
                               c.words_[2]) &&
                   std::invoke(std::forward<Condition>(condition), words_[3], b.words_[3],
                               c.words_[3]);
        } else {
            for (size_t n = 0; n < words_.size(); n++) {
                if (!std::invoke(std::forward<Condition>(condition), words_[n], b.words_[n],
                                 c.words_[n]))
                    return false;
            }
            return true;
        }
    }

    /// @brief This templated function is used to generate counting binary word operations.
    /// @param b the second BitArray object
    /// @param operation a lambda function that takes two uint64_t integers and returns an integer
    /// @return the result of the operation for all the words
    template <typename Operation>
    int count_binary_operation(const BitArray<N>& b, Operation&& operation) const {
        if constexpr (N == 64) {
            return std::invoke(std::forward<Operation>(operation), words_[0], b.words_[0]);
        } else if constexpr (N == 128) {
            return std::invoke(std::forward<Operation>(operation), words_[0], b.words_[0]) +
                   std::invoke(std::forward<Operation>(operation), words_[1], b.words_[1]);
        } else if constexpr (N == 192) {
            return std::invoke(std::forward<Operation>(operation), words_[0], b.words_[0]) +
                   std::invoke(std::forward<Operation>(operation), words_[1], b.words_[1]) +
                   std::invoke(std::forward<Operation>(operation), words_[2], b.words_[2]);
        } else if constexpr (N == 256) {
            return std::invoke(std::forward<Operation>(operation), words_[0], b.words_[0]) +
                   std::invoke(std::forward<Operation>(operation), words_[1], b.words_[1]) +
                   std::invoke(std::forward<Operation>(operation), words_[2], b.words_[2]) +
                   std::invoke(std::forward<Operation>(operation), words_[3], b.words_[3]);
        } else {
            int c = 0;
            for (size_t n = 0; n < nwords_; n++) {
                c += std::invoke(std::forward<Operation>(operation), words_[n], b.words_[n]);
            }
            return c;
        }
    }

  public:
    /// @brief Test (a & b) == b
    bool is_superset_of(const BitArray<N>& b) const {
        return test_binary_condition(b,
                                     [](uint64_t a, uint64_t b) -> bool { return (a & b) == b; });
    }

    /// @brief Test a & b == 0
    bool is_disjoint_from(const BitArray<N>& b) const {
        return test_binary_condition(b,
                                     [](uint64_t a, uint64_t b) -> bool { return (a & b) == 0; });
    }

    /// @brief Implements count(a & b)
    int intersection_count(const BitArray<N>& b) const {
        return count_binary_operation(
            b, [](uint64_t a, uint64_t b) -> int { return std::popcount(a & b); });
    }

    /// @brief Implements count(a ^ b)
    int symmetric_difference_count(const BitArray<N>& b) const {
        return count_binary_operation(
            b, [](uint64_t a, uint64_t b) -> int { return std::popcount(a ^ b); });
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

    /// @brief Test a & (b - c) == 0.
    ///
    /// In grouped operator application this checks whether any creation target in `b`, excluding
    /// orbitals also annihilated by `c`, is already occupied in this determinant.
    bool disjoint_from_difference(const BitArray<N>& b, const BitArray<N>& c) const {
        return test_ternary_condition(
            b, c, [](uint64_t a, uint64_t b, uint64_t c) -> bool { return (a & (b & (~c))) == 0; });
    }

    std::string str(size_t n = BitArray<N>::nbits) const {
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

    /// Returns a hash value for a BitArray object
    struct Hash {
        std::size_t operator()(const BitArray<N>& d) const noexcept {
            if constexpr (N == 64) {
                return d.words_[0];
            } else if constexpr (N == 128) {
                return hash_combine(d.words_[0], d.words_[1]);
            } else {
                std::size_t seed = 0;
                for (auto& w : d.words_) {
                    seed = hash_combine(seed, w);
                }
                return seed;
            }
        }
    };

  protected:
    // ==> Protected Functions <==

    // These functions are used to address bits in the BitArray.
    // Derived classes use them for word-level specializations, but they are not part of the public
    // BitArray API.

    /// the index of the word where the bit in position pos is found
    static constexpr size_t whichword(size_t pos) noexcept { return pos / bits_per_word; }

    /// the index of a bit within a word
    static constexpr size_t whichbit(size_t pos) noexcept { return pos % bits_per_word; }

    /// a mask for bit pos in its corresponding word
    static constexpr word_t maskbit(size_t pos) noexcept {
        return (static_cast<word_t>(1)) << whichbit(pos);
    }

    /// the word where bit in position pos is found
    constexpr word_t& getword(size_t pos) noexcept { return words_[whichword(pos)]; }

    /// the word where bit in position pos is found (const version)
    constexpr const word_t& getword(size_t pos) const noexcept { return words_[whichword(pos)]; }

    // ==> Protected Data <==

    /// The bits stored as a vector of words uninitialized/indeterminate for performance.
    container_t words_;
};

/// print a BitArray object to an output stream
template <size_t N> std::ostream& operator<<(std::ostream& os, const BitArray<N>& ba) {
    os << str(ba);
    return os;
}

} // namespace forte2

namespace std {
// specialization of std::hash for forte2::BitArray
template <size_t N> struct hash<forte2::BitArray<N>> {
    std::size_t operator()(const forte2::BitArray<N>& d) const noexcept {
        using HashT = typename forte2::BitArray<N>::Hash;
        return HashT{}(d);
    }
};

} // namespace std
