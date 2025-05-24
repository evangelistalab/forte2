#pragma once

#include <cstddef>
#include <vector>

namespace forte2 {

/// Return the parity of a permutation (0 = even, 1 = odd).
/// For example, the input vector is {1, 4, 3, 2, 0, 5, 7, 8, 6}.
/// In cycle notation, the permutation is (0 1 4)(2 3)(5)(6 7 8) and the parity is odd.
/// Only even-length cycles can change the permutation parity.
/// In the above example, only (2 3) is even-lengthed, which is 2.
/// IMPORTANT: the input vector index starts from 0!
///
/// For details, check the following
/// https://en.wikipedia.org/wiki/Parity_of_a_permutation
/// https://math.stackexchange.com/questions/65923/how-does-one-compute-the-sign-of-a-permutation
// of the integers 0,1,2,...,n - 1
int permutation_parity(const std::vector<size_t>& p);

} // namespace forte2
