#pragma once

#include <string>

#include "ndarray.h"

namespace forte2::matrix {

/// @brief Zero out a np_matrix
/// @param mat The matrix to zero out
void zero(np_matrix mat);

/// @brief Copy the contents of one np_matrix to another
/// @param src The source matrix to copy from
/// @param dest The destination matrix to copy to
void copy(np_matrix src, np_matrix dest);

/// @brief Scale a np_matrix by a factor
/// @param mat The matrix to scale
/// @param factor The scaling factor
void scale(np_matrix mat, double factor);

/// @brief Perform the operation y[:,:] = a * x[:,:] + y[:,:]
void daxpy(double a, np_matrix x, np_matrix y);

/// @brief Perform the operation y[row_y,:] = a * x[row_x,:] + y[row_y,:]
void daxpy_rows(double a, np_matrix x, int row_x, np_matrix y, int row_y);

/// @brief Perform the operation dot(x[row_x,:],y[row_y,:])
double dot_rows(np_matrix x, int row_x, np_matrix y, int row_y, size_t max_col = 0);

/// @brief Print the contents of a np_matrix to standard output
void print(np_matrix mat, std::string label = "Matrix");

} // namespace forte2::matrix