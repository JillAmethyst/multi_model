#ifndef __ONLINEUNSUPTASKDRIVDICLEAJOINTC_H_
#define __ONLINEUNSUPTASKDRIVDICLEAJOINTC_H_

#define OnlineUnsupTaskDrivDicLeaJointC_OK 100

#include <glog/logging.h>
#include <algorithm>
#include <vector>
#include "ADMM_Dlib.hpp"
#include "ComputeCost.hpp"
#include "dlib/matrix.h"
#include "matlab_function.hpp"
#include "multimodal_variables.hpp"

int OnlineUnsupTaskDrivDicLeaJointC(const DataMat& XArr, const IntVec& trls,
  const IntVec& n, const int d, DataMat& D, const int N);

#endif