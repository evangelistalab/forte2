#include "combinatorial.h"

namespace forte2 {

int permutation_parity(const std::vector<size_t>& p) {
    auto n = static_cast<int>(p.size());

    // vector of elements visited
    std::vector<bool> visited(n, false);

    int total_parity = 0;
    // loop over all the elements
    for (int i = 0; i < n; i++) {
        // if an element was not visited start following its cycle
        if (visited[i] == false) {
            int cycle_size = 0;
            int next = i;
            for (int j = 0; j < n; j++) {
                next = p[next];
                // mark the next element as visited
                visited[next] = true;
                // increase cycle size
                cycle_size += 1;
                // if the next element is the same as the one we
                // started from, we reached the end of the cycle
                if (next == i)
                    break;
            }
            total_parity += (cycle_size - 1) % 2;
        }
    }
    return total_parity % 2;
}
} // namespace forte2
