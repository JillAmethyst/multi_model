#ifndef _ONLINESUPTASKDRIVENDICVLEADECFUSJOINTQUADC_H_
#define _ONLINESUPTASKDRIVENDICVLEADECFUSJOINTQUADC_H_

#include <glog/logging.h>
#include <algorithm>
#include <vector>
#include "ADMM_Dlib.hpp"
#include "ComputeCost.hpp"
#include "matlab_function.hpp"
#include "multimodal_variables.hpp"

void OnlineSupTaskDrivDicLeaDecFusJointQuadC(const DataMat& XArr,
  const DataMat& Y, const IntVec& n, const int d, const DataMat& DUnsup,
  DataMat& DSup, std::vector<DataMat>& W, DataMat& b);
void projectionDic(const IntVec& n, DataMat& dic);

void GetBetaTemp2(const DataVec& alpha_vec, const DataVec& current_Y,
  const std::vector<DataMat>& W, const DataMat& b, const DataMat& DoTDoAll,
  const int d, const int S, const int j, const Dtype lambda,
  std::vector<DataMat>& Atemp2, std::vector<DataMat>& BetaTemp2);
#endif