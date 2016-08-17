#include <mpi.h>
#include <sys/time.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "ClassificationMultiClassDecFuxJoint.hpp"
#include "OnlineUnsupTaskDrivDicLeaJointC.hpp"
#include "data_function.hpp"
#include "matlab_function.hpp"
#include "multimodal_variables.hpp"

int main(int argc, char* argv[]) {

/*
    MPI_Status stat;
    int size, myid;
    int *argcVar = NULL;
    char ***argvVar = NULL;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   double timeStart = MPI_Wtime();
    double timeFinish;
*/

  static MultimodalConfigParser<Dtype>& config =
    MultimodalConfigParser<Dtype>::Instance();
  config.ParseFromFile(argv[1]);

  int atom_num = config.atom_num();
  int class_num = config.class_num();

  int d = class_num * atom_num;
  int N = config.N();
  int S = config.S();
  int N_test = config.N_test();
  const std::vector<int>& n_val = config.n();
  IntVec n = mat(n_val.data(), S);

  const char* XArr_filename = config.XArr_filename().c_str();
  const char* YArr_filename = config.YArr_filename().c_str();
  const char* trls_filename = config.trls_filename().c_str();
  const char* ttls_filename = config.ttls_filename().c_str();

  // init variables
  IntVec trls(N);
  IntVec ttls(N_test);
  DataMat trls_tmp(N, 1);
  DataMat ttls_tmp(N_test, 1);
  DataMat XArr(sum(n), N);
  DataMat YArr(sum(n), N_test);

  LoadDataFromFile(XArr_filename, XArr);
  LoadDataFromFile(YArr_filename, YArr);
  LoadDataFromFile(trls_filename, trls_tmp);
  LoadDataFromFile(ttls_filename, ttls_tmp);
  trls = matrix_cast<int>(trls_tmp) - 1;
  ttls = matrix_cast<int>(ttls_tmp) - 1;

  // std::cout << "trls:\n" << subm(trls, range(0,30),range(0,0)) << std::endl;
  // std::cout << "XArr:\n" << subm(XArr, range(0,9),range(0,9)) << std::endl;
  // std::cout << "YArr:\n" << subm(YArr, range(0,9),range(0,9)) << std::endl;

  ClassificationMultiClassDecFuxJoint(
    XArr, trls, YArr, ttls, N, N_test, d, S, n);
/*
    if (myid == 0)
    {
        (timeFinish = MPI_Wtime());
        printf("MPI_Wtime measured: %1.2f\n", timeFinish - timeStart);
        
    }
*/

}
