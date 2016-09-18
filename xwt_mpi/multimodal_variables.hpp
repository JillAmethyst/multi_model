#ifndef __MULTIMODAL_VARIABLES_H__
#define __MULTIMODAL_VARIABLES_H__

// system-wide dependency
#include <glog/logging.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>

// project-wide dependency
#include "Eigen/Dense"
#include "dlib/matrix.h"
#include "multimodal_config_parser.hpp"
#include "Eigen/IterativeLinearSolvers"


using namespace dlib;
using namespace std;
using std::vector;
using namespace Eigen;

typedef double Dtype;
// use column_major_layout to be compatible with MATLAB
typedef matrix<Dtype, 0, 0, default_memory_manager, column_major_layout>
  DataMat;
typedef matrix<Dtype, 0, 1> DataVec;
typedef matrix<Dtype, 1, 0> DataRowVec;
typedef matrix<int> IntMat;
typedef matrix<int, 0, 1> IntVec;
typedef matrix<int, 1, 0> IntRowVec;

typedef Matrix<Dtype, Dynamic, Dynamic, ColMajor> EigenMat;
typedef Matrix<Dtype, Dynamic, 1, ColMajor> EigenVec;

// const double global_lambda = 0.04;
// const double global_lambda2 = 0;
// const int global_iterADMM = 1000;
// const double global_rho = 0.1;
// const int global_iterUnsupDic = 5;
// const int global_iterSupDic = 10;
// const int global_batchSize = 100;
// const double global_ro = 5;
// const double global_nuQuad = 1e-8;
// const int global_iterQuad = 300;
// const int global_batchSizeQuad = 100;
// const double global_roQuad = 20;

// const bool global_computeCost = true;
// const bool global_intercept = false;

// // const string XArrfilename = "./XArr.dat";
// // const string YArrfilename = "./YArr.dat";

#endif
