#ifndef __COMPUTE_COST_H_
#define __COMPUTE_COST_H_

#include <glog/logging.h>
#include <algorithm>
#include <vector>
#include "dlib/matrix.h"
#include "matlab_function.hpp"
#include "multimodal_variables.hpp"

Dtype ComputeCost(const DataMat& XArr, const DataMat& D, const DataMat& Atemp,
  const IntVec& n, const Dtype lambda, const int d, const int S, const int N,
  const int batchSize, const int t);

#endif