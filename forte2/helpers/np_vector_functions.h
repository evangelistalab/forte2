#pragma once

#include <string>
#include <span>

#include "ndarray.h"

namespace forte2::vector {

/// @brief Zero out a np_vector
/// @param vec The vector to zero out
void zero(np_vector vec);

/// @brief Copy the contents of one np_vector to another
/// @param src The source vector to copy from
/// @param dest The destination vector to copy to
void copy(np_vector src, np_vector dest);

/// @brief Scale a np_vector by a factor
/// @param vec The vector to scale
/// @param factor The scaling factor
void scale(np_vector vec, double factor);

/// @brief Perform the operation y = a * x + y for specific rows in two vectors
/// @param a The scalar multiplier
/// @param x The source vector to multiply
/// @param y The destination vector to add to
void daxpy(double a, np_vector x, np_vector y);

/// @brief Check np_vector for contiguity and make a std::span from it
std::span<double> as_span(np_vector vec);

/// @brief Print the contents of a np_vector to standard output
void print(np_vector vec, std::string label = "Vector");

} // namespace forte2::vector