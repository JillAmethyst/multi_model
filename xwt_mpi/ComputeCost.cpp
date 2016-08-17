#include "ComputeCost.hpp"

// Thomas Lee
// ComputeCost returns only the mean value of each batch which differs from the
// algorithm in MATLAB
// The original algorithm further computes the mean of all batches which is
// totally unnecessary in my opinion.
Dtype ComputeCost(const DataMat& XArr, const DataMat& D, const DataMat& Atemp,
  const IntVec& n, const Dtype lambda, const int d, const int S, const int N,
  const int batchSize, const int t) {
  Dtype result = Dtype(0);

  for (int j = 0; j < batchSize; ++j) {
    // reshape is row major
    // transpose mat before reshaping
    DataMat A = reshape(trans(colm(Atemp, j)), d, S);
    Dtype costTemp = lambda / sqrt(d) * norm12_cpp(A);
    int temp = 0;
    for (int s = 0; s < S; ++s) {
      Dtype f_norm = frobenius_norm_cpp(
        subm(XArr, range(temp, temp + n(s) - 1), range(j + t, j + t)) -
        rowm(D, range(temp, temp + n(s) - 1)) * colm(A, s));
      costTemp += 0.5 * f_norm * f_norm;
      temp += n(s);
    }
    result += costTemp;
  }

  return result / batchSize;
}