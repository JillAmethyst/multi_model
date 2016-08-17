#include "multimodal_config_parser.hpp"

template <typename T>
void MultimodalConfigParser<T>::ParseFromFile(const char* file_name) {
  std::ifstream config_file(file_name);
  rapidjson::IStreamWrapper stream_wrapper(config_file);

  rapidjson::Document d;
  d.ParseStream(stream_wrapper);

  global_lambda_ = static_cast<T>(d["global_lambda"].GetDouble());
  global_lambda2_ = static_cast<T>(d["global_lambda2"].GetDouble());
  global_ro_ = static_cast<T>(d["global_ro"].GetDouble());
  global_rho_ = static_cast<T>(d["global_rho"].GetDouble());
  global_roQuad_ = static_cast<T>(d["global_roQuad"].GetDouble());
  global_nuQuad_ = static_cast<T>(d["global_nuQuad"].GetDouble());

  global_iterUnsupDic_ = d["global_iterUnsupDic"].GetInt();
  global_iterSupDic_ = d["global_iterSupDic"].GetInt();
  global_batchSize_ = d["global_batchSize"].GetInt();
  global_iterADMM_ = d["global_iterADMM"].GetInt();
  global_iterQuad_ = d["global_iterQuad"].GetInt();
  global_batchSizeQuad_ = d["global_batchSizeQuad"].GetInt();

  global_computeCost_ = d["global_computeCost"].GetBool();
  global_intercept_ = d["global_intercept"].GetBool();
  //global_ADMMwithCG_ = d["global_ADMMwithCG"].GetBool();


  atom_num_ = d["atom_num"].GetInt();
  class_num_ = d["class_num"].GetInt();
  N_ = d["N"].GetInt();
  S_ = d["S"].GetInt();
  N_test_ = d["N_test"].GetInt();
  const rapidjson::Value& n_val = d["n"];
  n_.clear();
  n_.resize(S_);
  assert(S_ == static_cast<int>(n_val.Size()));
  for (rapidjson::SizeType i = 0; i < n_val.Size(); ++i) {
    n_[i] = n_val[i].GetInt();
  }

  XArr_filename_ = d["XArr_filename"].GetString();
  YArr_filename_ = d["YArr_filename"].GetString();
  trls_filename_ = d["trls_filename"].GetString();
  ttls_filename_ = d["ttls_filename"].GetString();

  initialized_ = true;
}

template class MultimodalConfigParser<float>;
template class MultimodalConfigParser<double>;
