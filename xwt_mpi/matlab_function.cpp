#include "matlab_function.hpp"

using namespace dlib;
using namespace std;

IntVec unique_cpp(const IntVec& vec) {
  std::vector<int> tmp_vec(vec.begin(), vec.end());
  std::sort(tmp_vec.begin(), tmp_vec.end());
  std::vector<int>::iterator it;
  it = std::unique(tmp_vec.begin(), tmp_vec.end());
  tmp_vec.resize(std::distance(tmp_vec.begin(), it));
  IntVec result = mat(tmp_vec.data(), tmp_vec.size());
  return result;
}

IntVec find_cpp(const IntVec& vec) {
  std::vector<int> tmp_vec;
  tmp_vec.clear();
  for (int i = 0; i < vec.size(); ++i) {
    if (vec(i) > 0) {
      tmp_vec.push_back(i);
    }
  }
  IntVec result = mat(tmp_vec.data(), tmp_vec.size());
  return result;
}

// random permutation from 0 to N - 1
IntVec randperm_cpp(const int N, const int seed) {
  if (seed < 0){
    std::srand(0);
  }
  else {
    std::srand(seed);
  }
  IntVec result(N);
  for (int i = 0; i < N; ++i) {
    result(i) = i;
  }
  for (int i = N - 1; i >= 0; --i) {
    int rand_idx = std::rand() % (i + 1);
    int tmp = result(i);
    result(i) = result(rand_idx);
    result(rand_idx) = tmp;
  }

  return result;
}

DataMat factor_cpp(const DataMat& A, const Dtype rho) {
  const int m = A.nr();
  const int n = A.nc();
  DataMat result;
  if (m >= n) {
    result = chol(trans(A) * A + rho * identity_matrix<Dtype>(n));
  } else {
    result = chol(A * trans(A) / rho + identity_matrix<Dtype>(m));
  }
  return result;
}

template <>
DataMat repmat_cpp<DataMat>(const DataMat& A, const int M, const int N) {
  const int m_ = A.nr();
  const int n_ = A.nc();

  DataMat result;
  result.set_size(M * m_, N * n_);

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      set_subm(result, range(m * m_, (m + 1) * m_ - 1),
        range(n * n_, (n + 1) * n_ - 1)) = A;
    }
  }

  return result;
}

template <>
IntMat repmat_cpp<IntMat>(const IntMat& A, const int M, const int N) {
  const int m_ = A.nr();
  const int n_ = A.nc();

  IntMat result;
  result.set_size(M * m_, N * n_);

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      set_subm(result, range(m * m_, (m + 1) * m_ - 1),
        range(n * n_, (n + 1) * n_ - 1)) = A;
    }
  }

  return result;
}

Dtype norm12_cpp(const DataMat& mat) {
  Dtype result = 0;
  const int nr = mat.nr();
  for (int i = 0; i < nr; ++i) {
    result += sqrt(sum(squared(rowm(mat, i))));
  }
  return result;
}

DataMat sum_cpp(const DataMat& A) {
  const int m = A.nr();
  const int n = A.nc();
  DataMat result(1, n);

  for (int i = 0; i < n; i++) {
    result(0, i) = sum(colm(A, i));
  }

  return result;
}

IntVec binary2cardinal_cpp(const IntVec& bin_idx) {
  const int total_num = sum(bin_idx);
  const int row_num = bin_idx.nr();
  IntVec result(total_num);
  int count = 0;
  for (int i = 0; i < row_num; ++i) {
    if (bin_idx(i) > 0) {
      result(count) = i;
      count += 1;
    }
  }
  CHECK(count == total_num) << "non zero entry number mismatches.";
  return result;
}

DataVec rowsum_cpp(const DataMat& data) {
  const int row_num = data.nr();
  DataVec result(row_num);
  for (int i = 0; i < row_num; ++i) {
    result(i) = sum(rowm(data, i));
  }
  return result;
}

DataRowVec colsum_cpp(const DataMat& data) {
  const int col_num = data.nc();
  DataRowVec result(col_num);
  for (int i = 0; i < col_num; ++i) {
    result(i) = sum(colm(data, i));
  }
  return result;
}

DataVec rownorm_cpp(const DataMat& data) {
  const int row_num = data.nr();
  DataVec result(row_num);
  for (int i = 0; i < row_num; ++i) {
    result(i) = std::sqrt(sum(squared(rowm(data, i))));
  }
  return result;
}

std::vector<int> equally_division(
  const int data_length, const int division_num) {
  CHECK(data_length >= division_num) << "data_length is smaller than division_num";
  std::vector<int> result(division_num);
  if (data_length % division_num == 0) {
    for (int i = 0; i < division_num; ++i) {
      result[i] = data_length / division_num;
    }
    return result;
  }

  const int rough_estimation =
    static_cast<int>(std::floor(Dtype(data_length) / division_num));
  const int result_x = data_length - division_num * rough_estimation;
  const int result_y = division_num * (rough_estimation + 1) - data_length;
  for (int i = 0; i < division_num; ++i) {
    if (i < result_x) {
      result[i] = rough_estimation + 1;
    } else {
      result[i] = rough_estimation;
    }
  }

  return result;
}