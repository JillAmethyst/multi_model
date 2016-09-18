#include "ClassificationMultiClassDecFuxJoint.hpp"

void ClassificationMultiClassDecFuxJoint(const DataMat& XArr,
  const IntVec& trls, const DataMat& YArr, const IntVec& ttls, const int N,
  const int N_test, const int d, const int S, const IntVec& n, const char* DUnsupFileName) {

  /**********************
  pass global parameters
  **********************/
  static MultimodalConfigParser<Dtype>& config =
  MultimodalConfigParser<Dtype>::Instance();

  CHECK(config.Initialized()) << "config has not been Initialzed";
  const Dtype rho = config.global_rho();
  const Dtype lambda = config.global_lambda();
  const int iterADMM = config.global_iterADMM();
  const bool ADMMwithCG = config.global_ADMMwithCG(); 

  const Dtype tolCG = 1e-5;
  const int iterCG = 20;

  IntVec uniqtrls = unique_cpp(trls); 
  const int number_classes = uniqtrls.nr();

  DataMat DUnsup(sum(n), d);

  #define root 0

  int rank, ProcSize;
  MPI_Comm_size(MPI_COMM_WORLD,&ProcSize);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  /*************************
     Unsup D computation
  *************************/

  // OnlineUnsupTaskDrivDicLeaJointC(XArr, trls, n, d, DUnsup, N);

  // const char* a1_filename = "./DUnsup/DUnsupAtom4Carc_1400People_x_160.dat";
  // const char* a1_filename = "./DUnsup/DUnsupOrigin297.dat";
  LoadDataFromFile(DUnsupFileName, DUnsup);

  // if (rank == root) {
  //   WriteDataToFile(DUnsupFileName, DUnsup);
  // } 
  /*************************
     sample admm feature
  *************************/
  DataMat Atr;
  DataMat Att;

  finalADMMfeature(XArr, YArr, DUnsup, n, d, lambda, rho, iterADMM, tolCG, iterCG, ADMMwithCG, Atr, Att);

  /*************************
       compute W and b
  *************************/
  std::vector<DataMat> modelQuadUnsup_W(S);
  std::vector<DataMat> modelQuadUnsup_b(S);

  computeWandB(number_classes, S, N, N_test, d, trls, ttls, Atr, Att, modelQuadUnsup_W, modelQuadUnsup_b);

  /*****************************************
  begin Sup learning, get DSup and new W b
  *****************************************/

//   DataMat DSup(sum(n), d);

//   OnlineSupTaskDrivDicLeaDecFusJointQuadC(XArr, outputVectorTrain, 
//       n, d, DUnsup, DSup, modelQuadUnsup_W, modelQuadUnsup_b[0], rank, ProcSize);

//   /*************************
//   train sample admm feature
//   *************************/

//   DataMat Atr_sup;

//   Atr_sup = zeros_matrix<Dtype>(d * S, N);

//   ADMM_CG_xwt(DSup, XArr, n, S, lambda, rho, iterADMM, Atr_sup, tolCG, iterCG);

//   /*************************
//   test sample admm feature
//   *************************/
//   DataMat Att_sup;

//   Att_sup = zeros_matrix<Dtype>(d * S, N_test);

//   ADMM_CG_xwt(DSup, YArr, n, S, lambda, rho, iterADMM, Att_sup, tolCG, iterCG);

//   /*************************************
//   let admm feature multiple with W and b
//   *************************************/

//   DataMat modelOutTrainSup = zeros_matrix<Dtype>(number_classes, N);
//   DataMat modelOutTestSup = zeros_matrix<Dtype>(number_classes, N_test);

// #ifdef USE_OMP
// #pragma omp parallel for
// #endif
//   for (int s = 0; s < S; ++s) {
//     std::cout << "s:" << s << std::endl;

//     int temp = d*s;

//       // Atr -> modelOutTrainSup
//     DataMat temp_Atr(d, N);
//     temp_Atr = rowm(Atr_sup, range(temp, temp + d - 1));

//     DataMat modelOutTrainTemp(number_classes, N);
//     modelOutTrainTemp = trans(modelQuadUnsup_W[s]) * temp_Atr + repmat_cpp(modelQuadUnsup_b[s], 1, N);

//     for (int j = 0; j < N; j++) {
//       DataMat moTrj = colm(modelOutTrainTemp, j);

//       DataMat temp_j =
//       repmat_cpp(moTrj, 1, number_classes) -
//       identity_matrix<Dtype>(number_classes);

//       temp_j = pointwise_multiply(temp_j, temp_j);

//       set_colm(modelOutTrainSup, j) =
//       colm(modelOutTrainSup, j) + trans(sum_cpp(temp_j));
//     } // for j

//       // Att -> modelOutTestSup
//     DataMat temp_Att(d, N_test);
//     temp_Att = rowm(Att_sup, range(temp, temp + d - 1));

//     DataMat modelOutTestTemp(number_classes, N_test);
//     modelOutTestTemp = trans(modelQuadUnsup_W[s]) * temp_Att + repmat_cpp(modelQuadUnsup_b[s], 1, N);

//     for (int j = 0; j < N_test; j++) {
//       DataMat moTej = colm(modelOutTestTemp, j);
//       DataMat temp_j =
//       repmat_cpp(moTej, 1, number_classes) -
//       identity_matrix<Dtype>(number_classes);
//       temp_j = pointwise_multiply(temp_j, temp_j);
//       set_colm(modelOutTestSup, j) =
//       colm(modelOutTestSup, j) + trans(sum_cpp(temp_j));
//     }//for j

//   }//for s->S

//   /*********************************************************************************
//   modelOutTrainSup modelOutTestSup => predictLableTrainSup predictLableTestSup
//   *********************************************************************************/
//     // classify train samples
//   IntVec predictedLableTrainSup(N);
//   predictedLableTrainSup = 0;
//   sum_classify = 0;

//   for (int i = 0; i < N; i++) {
//     matrix<double, 0, 1> colm_temp;
//     colm_temp = colm(modelOutTrainSup, i);
//     predictedLableTrainSup(i) = min_idx_cpp(colm_temp);
//     sum_classify += predictedLableTrainSup(i) == trls(i);
//   }
//   double CCRQuadTrainSup = sum_classify / N * 100;
//   cout << "train_Sup:" << CCRQuadTrainSup << endl;

//     // classify testing samples
//   IntVec predictedLableTestSup(N_test);
//   predictedLableTestSup = 0;
//   sum_classify = 0;

//   for (int i = 0; i < N_test; i++) {
//     matrix<double, 0, 1> colm_temp;
//     colm_temp = colm(modelOutTestSup, i);
//     predictedLableTestSup(i) = min_idx_cpp(colm_temp);
//     sum_classify += predictedLableTestSup(i) == ttls(i);
//   }
//   double CCRQuadTestSup = sum_classify / N_test * 100;
//   cout << "test_Sup:" << CCRQuadTestSup << endl;


  MPI_Finalize();

} // classification
