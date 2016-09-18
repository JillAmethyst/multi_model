#ifndef __COMPUTEAANDB_H__
#define __COMPUTEAANDB_H__

#include "multimodal_variables.hpp"
#include "SGDMultiClassQuadC.hpp"
#include "matlab_function.hpp"

int computeWandB(const int number_classes, int S, const int N, const int N_test, const int d, const IntVec& trls, const IntVec& ttls, DataMat& Atr, DataMat& Att, std::vector<DataMat>& modelQuadUnsup_W, std::vector<DataMat>& modelQuadUnsup_b);

#endif