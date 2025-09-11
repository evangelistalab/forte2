#pragma once

#include <string>
#include <complex>

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

/// @brief Expand a packed 4D tensor T([p>q],[r>s]) stored as a 2D matrix into a full 4D tensor with
/// elements:   T[p,q,r,s] = +T([p>q],[r>s])
///             T[p,q,s,r] = -T([p>q],[r>s])
///             T[q,p,r,s] = -T([p>q],[r>s])
///             T[q,p,s,r] = +T([p>q],[r>s])
///
/// @param M The input 2D matrix.
/// @return The expanded 4D tensor.
np_tensor4 packed_tensor4_to_tensor4(np_matrix m);

} // namespace forte2::matrix

namespace forte2::matrix_complex {

/// @brief Zero out a np_matrix
/// @param mat The matrix to zero out
void zero(np_matrix_complex mat);

/// @brief Copy the contents of one np_matrix_complex to another
/// @param src The source matrix to copy from
/// @param dest The destination matrix to copy to
void copy(np_matrix_complex src, np_matrix_complex dest);

/// @brief Scale a np_matrix_complex by a factor
/// @param mat The matrix to scale
/// @param factor The scaling factor
void scale(np_matrix_complex mat, double factor);

/// @brief Perform the operation y[:,:] = a * x[:,:] + y[:,:]
void daxpy(double a, np_matrix_complex x, np_matrix_complex y);

/// @brief Perform the operation y[row_y,:] = a * x[row_x,:] + y[row_y,:]
void daxpy_rows(double a, np_matrix_complex x, int row_x, np_matrix_complex y, int row_y);

/// @brief Perform the operation dot(x[row_x,:],y[row_y,:])
std::complex<double> dot_rows(np_matrix_complex x, int row_x, np_matrix_complex y, int row_y, size_t max_col = 0);

/// @brief Print the contents of a np_matrix_complex to standard output
void print(np_matrix_complex mat, std::string label = "Matrix");

/// @brief Expand a packed 4D tensor T([p>q],[r>s]) stored as a 2D matrix into a full 4D tensor with
/// elements:   T[p,q,r,s] = +T([p>q],[r>s])
///             T[p,q,s,r] = -T([p>q],[r>s])
///             T[q,p,r,s] = -T([p>q],[r>s])
///             T[q,p,s,r] = +T([p>q],[r>s])
///
/// @param M The input 2D matrix.
/// @return The expanded 4D tensor.
np_tensor4_complex packed_tensor4_to_tensor4(np_matrix_complex m);

} // namespace forte2::matrix