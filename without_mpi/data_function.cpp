#include "data_function.hpp"

int LoadDataFromFile(const char* dat_name, DataMat& M, int start_loc) {
  std::ifstream file;
  file.open(dat_name, std::ifstream::in | std::ifstream::binary);
  if (file.fail()) {
    std::cerr << "Fail to open " << dat_name << std::endl;
    return -1;
  }
  file.seekg(start_loc, file.beg);
  file.read(reinterpret_cast<char*>(M.begin()), sizeof(Dtype) * M.size());
  file.close();
  return 0;
}

int WriteDataToFile(const char* dat_name, DataMat& M, int start_loc) {
  std::ofstream file;
  file.open(dat_name);
  if (file.fail()) {
    std::cerr << "Fail to open " << dat_name << std::endl;
    return -1;
  }

  file.write(reinterpret_cast<char*>(M.begin()), sizeof(Dtype) * M.size());
  file.close();
}
