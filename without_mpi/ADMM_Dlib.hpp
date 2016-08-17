#ifndef __ADMM_DLIB_H_
#define __ADMM_DLIB_H_

#include <math.h>
#include <iostream>
#include "dlib/matrix.h"
#include "matlab_function.hpp"
#include "multimodal_variables.hpp"

typedef Eigen::Map<Eigen::MatrixXd> EigenMap;

#define Dlib2Eigen(x, z) EigenMap(z)(x.begin(), x.nr(), x.nc())

/*************
L U dlib admm
*************/

void ADMM_Dlib(const DataMat& dlibD, const DataMat& dlibX, const IntVec& n,
  const Dtype lambda, const Dtype rho, const DataMat& dlibL,
  const DataMat& dlibU, const int iterMax, DataMat& alpha);

void ADMM(const DataMat& Dictionarys, const DataMat& X, const IntVec& n,
  const Dtype Lambda, const Dtype rho, const DataMat& L, const DataMat& U,
  const int iterMax, const int S, DataMat& alpha);

DataMat LltSolveDlib(const DataMat& L, const DataMat& b);

//void ADMM_CG(DataMat& DtD, DataMat& DtX, const int S,
//  const Dtype Lambda, const Dtype rho, const int iterMax, DataMat& alpha);

/*************
CG eigen admm
*************/

void ADMM_CG(DataMat& D, DataMat& XArr, const int S, const Dtype Lambda,
  const Dtype rho, const int iterMax, DataMat& alpha, const IntVec& n);

void ADMM_ConjugateGradient(const EigenMap& dtd_e, const EigenMap& dtx_e,
  const int S, const Dtype Lambda, const Dtype rho, const int iterMax,
  EigenMap& alpha_e);

/*******************
CG eigen manual admm
*******************/

// void ADMM_CG_xwt(DataMat& D, DataMat& XArr, const IntVec& n, const int S,
//   const Dtype Lambda, const Dtype rho, const int iterMax, DataMat& alpha, const Dtype& tolCG, const int iterCG);

// void ADMM_ConjugateGradient_xwt(const EigenMap& D, const EigenMap& DtD, const EigenMap& DtX, const EigenMat& n, const int S, const Dtype Lambda, const Dtype rho, const int iterMax,
//   EigenMap& alpha_e, const Dtype& tolCG, const int iterCG);

// inline void CG_cpp(const EigenMat& D, const EigenMat& Dt, const EigenVec& b, const int iter, const Dtype rho, EigenVec& x, const Dtype& tolCG);

/*******************
CG dlib manual admm
*******************/

void ADMM_CG_xwt(DataMat& D, const DataMat& XArr, const IntVec& n, const int S,
  const Dtype Lambda, const Dtype rho, const int iterMax, DataMat& alpha, const Dtype& tolCG, const int iterCG);

inline void CG_cpp(const DataMat& D, const DataMat& DT, const DataVec& b, const int iter, const Dtype rho, DataVec& x, const Dtype& tolCG);

#endif
