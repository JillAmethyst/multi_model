#ifndef __SGDMULTICLASSQUADC_H_
#define __SGDMULTICLASSQUADC_H_

#include <glog/logging.h>
#include <algorithm>
#include <vector>
#include "dlib/matrix.h"
#include "matlab_function.hpp"
#include "multimodal_variables.hpp"
//#include "ADMM_Dlib.hpp"
//#include "ComputeCost.hpp"

int SGDMultiClassQuadC(const DataMat& Atr, DataMat& outputVectorTrain,
  DataMat& temp_W, DataMat& temp_b);

#endif