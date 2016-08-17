#ifndef __DATA_FUNCTION_H_
#define __DATA_FUNCTION_H_

#include "multimodal_variables.hpp"

int LoadDataFromFile(const char* dat_name, DataMat& M, int start_loc = 0);
int WriteDataToFile(const char* dat_name, DataMat& M, int start_loc = 0);

#endif
