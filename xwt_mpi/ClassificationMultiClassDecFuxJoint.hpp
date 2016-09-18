#ifndef __CLASSIFICATIONMULTICLASSDECFUXJOINT_H__
#define __CLASSIFICATIONMULTICLASSDECFUXJOINT_H__

#include "SGDMultiClassQuadC.hpp"
#include "data_function.hpp"
#include "matlab_function.hpp"
#include "multimodal_variables.hpp"
#include "OnlineUnsupTaskDrivDicLeaJointC.hpp"
#include "finalADMMfeature.hpp"
#include "computeWandB.hpp"

void ClassificationMultiClassDecFuxJoint(const DataMat& XArr,
  const IntVec& trls, const DataMat& YArr, const IntVec& ttls, const int N,
  const int N_test, const int d, const int S, const IntVec& n, const char* DUnsupFileName);

#endif
