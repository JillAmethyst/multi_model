#ifndef __FINALADMMFEATURE_H__
#define __FINALADMMFEATURE_H__

#include "multimodal_variables.hpp"
#include "ADMM_Dlib.hpp"

int finalADMMfeature(const DataMat& XArr, const DataMat& YArr, DataMat& DUnsup, const IntVec n, const int d, const Dtype lambda, const Dtype rho, const int iterADMM, const Dtype tolCG, const int iterCG, const bool ADMMwithCG, DataMat& Atr, DataMat& Att);

#endif