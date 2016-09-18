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
  const char* DUnsupFileName = argv[2];

  int rank, ProcSize;
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD,&ProcSize);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  static MultimodalConfigParser<Dtype>& config =
    MultimodalConfigParser<Dtype>::Instance();
  config.ParseFromFile(argv[1]);

  int atom_num = config.atom_num();
  int class_num = config.class_num();
  int modalDim = config.modalDim();

  int d = class_num * atom_num;
  int S = config.S();
  int N_config = config.N();
  int N_test_config = config.N_test();  
  const int N = N_config - N_config % ProcSize;
  const int N_test = N_test_config - N_test_config % ProcSize;

  const std::vector<int>& n_val = config.n();
  IntVec n = mat(n_val.data(), S);

  const char* XArr_filename = config.XArr_filename().c_str();
  const char* YArr_filename = config.YArr_filename().c_str();
  const char* trls_filename = config.trls_filename().c_str();
  const char* ttls_filename = config.ttls_filename().c_str();

  // init variables
  IntVec trls_config(N);
  IntVec ttls_config(N_test);
  DataMat trls_tmp(N, 1);
  DataMat ttls_tmp(N_test, 1);
  DataMat XArr_config(sum(n), N);
  DataMat YArr_config(sum(n), N_test);

  LoadDataFromFile(XArr_filename, XArr_config);
  LoadDataFromFile(YArr_filename, YArr_config);
  LoadDataFromFile(trls_filename, trls_tmp);
  LoadDataFromFile(ttls_filename, ttls_tmp);
  trls_config = matrix_cast<int>(trls_tmp) - 1;
  ttls_config = matrix_cast<int>(ttls_tmp) - 1;

  const IntVec trls  = colm(trls_config, range(0, N - 1));
  const IntVec ttls  = colm(ttls_config, range(0, N_test -1));

  DataMat XArrTmp(sum(n), N);
  DataMat YArrTmp(sum(n), N_test);
  cout<<"n(0):"<<n(0)<<endl;
  for(int i = 0; i<S; i++){
    set_rowm(XArrTmp, range(0+i*modalDim, n(0)+i*modalDim) ) = subm(XArr_config, range(0+i*modalDim, n(0)+i*modalDim), range(0, N - 1));
    set_rowm(YArrTmp, range(0+i*modalDim, n(0)+i*modalDim) ) = subm(YArr_config, range(0+i*modalDim, n(0)+i*modalDim), range(0, N_test - 1));
  }
  const DataMat XArr = XArrTmp;
  const DataMat YArr = YArrTmp;

  ClassificationMultiClassDecFuxJoint(
    XArr, trls, YArr, ttls, N, N_test, d, S, n, DUnsupFileName);
/*
    if (myid == 0)
    {
        (timeFinish = MPI_Wtime());
        printf("MPI_Wtime measured: %1.2f\n", timeFinish - timeStart);
        
    }
*/

}
