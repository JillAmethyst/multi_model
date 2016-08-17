#include <cstdlib>
#include <iostream>
#include <vector>
#include "multimodal_variables.hpp"
#include "dlib/matrix/matrix_cholesky.h"

typedef Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> EMat;

DataMat LltSolveDlib(const DataMat& L, const DataMat& b) {
  DataMat X(b);
  using namespace blas_bindings;
  // Solve L*y = b;
  triangular_solver(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, L, X);
  // Solve L'*X = Y;
  triangular_solver(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, L, X);
  return X;
}

int main(int argc, char* argv[]) {
  EMat tmp = EMat::Random(5, 5);
  const Dtype rho = 10;
  EMat a = tmp.transpose() * tmp + rho * EMat::Identity(5, 5);
  EMat b = EMat::Random(5, 1);
  EMat x = a.ldlt().solve(b);
  std::cout << "Eigen_x:\n" << x << std::endl;

  DataMat a_d = mat(a.data(), 5, 5);
  DataMat b_d = mat(b.data(), 5, 1);
  cholesky_decomposition<DataMat> c_d(a_d);
  DataMat l_d = c_d.get_l();
  DataMat x_d = LltSolveDlib(l_d, b_d);
  std::cout << "Dlib_x:\n" << x_d << std::endl;
}