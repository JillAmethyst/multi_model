#ifndef __MULTIMODAL_CONFIG_PARSER_H_
#define __MULTIMODAL_CONFIG_PARSER_H_

// MultimodalConfigParser Created by Thomas Lee 2016/06/15

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include "rapidjson/document.h"
#include "rapidjson/istreamwrapper.h"

template <typename T>
class MultimodalConfigParser {
 public:
  static MultimodalConfigParser& Instance() {
    static MultimodalConfigParser instance;
    return instance;
  }
  bool Initialized() const { return initialized_; }

  void ParseFromFile(const char* file_name);

  inline T global_lambda() const { return global_lambda_; }
  inline T global_lambda2() const { return global_lambda2_; }
  inline T global_ro() const { return global_ro_; }
  inline T global_rho() const { return global_rho_; }
  inline T global_roQuad() const { return global_roQuad_; }
  inline T global_nuQuad() const { return global_nuQuad_; }
  inline int global_iterUnsupDic() const { return global_iterUnsupDic_; }
  inline int global_iterSupDic() const { return global_iterSupDic_; }
  inline int global_batchSize() const { return global_batchSize_; }
  inline int global_iterADMM() const { return global_iterADMM_; }
  inline int global_iterQuad() const { return global_iterQuad_; }
  inline int global_batchSizeQuad() const { return global_batchSizeQuad_; }

  inline bool global_computeCost() const { return global_computeCost_; }
  inline bool global_intercept() const { return global_intercept_; }
  //inline bool global_ADMMwithCG() const { return global_ADMMwithCG_; }

  inline int atom_num() const { return atom_num_; }
  inline int class_num() const { return class_num_; }
  inline int N() const { return N_; };
  inline int S() const { return S_; };
  inline int N_test() const { return N_test_; }
  inline const std::vector<int>& n() const { return n_; };
  inline std::string XArr_filename() { return XArr_filename_; }
  inline std::string YArr_filename() { return YArr_filename_; }
  inline std::string trls_filename() { return trls_filename_; }
  inline std::string ttls_filename() { return ttls_filename_; }

 private:
  MultimodalConfigParser() : initialized_(false){};
  explicit MultimodalConfigParser(const char* file_name) {
    ParseFromFile(file_name);
  }

  int atom_num_;
  int class_num_;
  int N_;
  int S_;
  int N_test_;
  std::vector<int> n_;

  std::string XArr_filename_;
  std::string YArr_filename_;
  std::string trls_filename_;
  std::string ttls_filename_;

  T global_lambda_;
  T global_lambda2_;
  T global_ro_;
  T global_rho_;
  T global_roQuad_;
  T global_nuQuad_;

  int global_iterUnsupDic_;
  int global_iterSupDic_;
  int global_batchSize_;
  int global_iterADMM_;
  int global_iterQuad_;
  int global_batchSizeQuad_;

  bool global_computeCost_;
  bool global_intercept_;
  //bool global_ADMMwithCG_;

  // singleton
  bool initialized_;
};

#endif
