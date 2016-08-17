#ifndef __MATLAB_FUNCTION_H__
#define __MATLAB_FUNCTION_H__

#include "multimodal_variables.hpp"

IntVec unique_cpp(const IntVec& vec);

IntVec find_cpp(const IntVec& vec);

IntVec randperm_cpp(const int N, const int seed = -1);

DataMat factor_cpp(const DataMat& A, const Dtype rho);

template <class T>
T repmat_cpp(const T& A, const int M, const int N);

inline Dtype frobenius_norm_cpp(const DataMat& mat) {
  return sqrt(sum(squared(mat)));
}

Dtype norm12_cpp(const DataMat& mat);

DataMat sum_cpp(const DataMat& A);

// return the first minimum index of a vector
// TODO: handle multiple minima if necessary
template <typename T>
inline size_t min_idx_cpp(matrix<T, 0, 1>& vec) {
  return std::distance(vec.begin(), std::min_element(vec.begin(), vec.end()));
}

// return the first maximum index of a vector
// TODO: handle multiple maxima if necessary
template <typename T>
inline size_t max_idx_cpp(matrix<T, 0, 1>& vec) {
  return std::distance(vec.begin(), std::max_element(vec.begin(), vec.end()));
}

IntVec binary2cardinal_cpp(const IntVec& bin_idx);

DataVec rowsum_cpp(const DataMat& data);

DataRowVec colsum_cpp(const DataMat& data);

DataVec rownorm_cpp(const DataMat& data);

std::vector<int> equally_division(const int data_length, const int division_num);

#endif