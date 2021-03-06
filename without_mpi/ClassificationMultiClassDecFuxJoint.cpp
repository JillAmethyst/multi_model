#include "ClassificationMultiClassDecFuxJoint.hpp"

void ClassificationMultiClassDecFuxJoint(const DataMat& XArr,
  const IntVec& trls, const DataMat& YArr, const IntVec& ttls, const int N,
  const int N_test, const int d, const int S, const IntVec& n) {

  /**********************
  pass global parameters
  **********************/

  static MultimodalConfigParser<Dtype>& config =
    MultimodalConfigParser<Dtype>::Instance();

  CHECK(config.Initialized()) << "config has not been Initialzed";
  const Dtype rho = config.global_rho();
  const Dtype lambda = config.global_lambda();
  const int iterADMM = config.global_iterADMM();

  const Dtype tolCG = 1e-5;
  const int iterCG = 20;

  IntVec uniqtrls = unique_cpp(trls); // cout<<"uniqtrls = "<<uniqtrls<<endl;
  int number_classes = uniqtrls.nr();

  int rank, ProcSize;

  /*************************
     Unsup D computation
  *************************/

  DataMat DUnsup(sum(n), d);

  OnlineUnsupTaskDrivDicLeaJointC(XArr, trls, n, d, DUnsup, rank, ProcSize);

  /*************************
  train sample admm feature
  *************************/

  DataMat Atr;

  Atr = zeros_matrix<Dtype>(d * S, N);

  ADMM_CG_xwt(DUnsup, XArr, n, S, lambda, rho, iterADMM, Atr, tolCG, iterCG);
  //  ADMM_Dlib(DUnsup, XArr, n, lambda, rho, L, U, iterMax, Atr);//LU-no-mpi  

  /*************************
  test sample admm feature
  *************************/

  DataMat Att;

  Att = zeros_matrix<Dtype>(d * S, N_test);

  ADMM_CG_xwt(DUnsup, YArr, n, S, lambda, rho, iterADMM, Att, tolCG, iterCG);//no-mpi
  //  ADMM_Dlib(DUnsup, YArr, n, lambda, rho, L, U, iterMax, Att);//LU-no-mpi

  /*************************
       compute W and b
  *************************/
  DataMat outputVectorTrain = zeros_matrix<Dtype>(number_classes, N);

  for (int j = 0; j < N; ++j) {
    outputVectorTrain(trls(j), j) = 1;
  }

  DataMat modelOutTrainUnsup = zeros_matrix<Dtype>(number_classes, N);
  DataMat modelOutTestUnsup = zeros_matrix<Dtype>(number_classes, N_test);

  std::vector<DataMat> modelQuadUnsup_W(S);
  std::vector<DataMat> modelQuadUnsup_b(S);

  for(int s = 0; s<S; s++){
    modelQuadUnsup_W[s] = zeros_matrix<Dtype>(d, number_classes);
    modelQuadUnsup_b[s] = zeros_matrix<Dtype>(number_classes,1);
  }

    
#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (int s = 0; s < S; ++s) {
    int temp = d*s;
    std::cout << "s:" << s << std::endl;
    DataMat temp_Atr(d, N);

    temp_Atr = rowm(Atr, range(temp, temp + d - 1));

    DataMat temp_W = zeros_matrix<Dtype>(d, number_classes);
    DataMat temp_b = zeros_matrix<Dtype>(number_classes, 1);

    SGDMultiClassQuadC(temp_Atr, outputVectorTrain, temp_W, temp_b);

    modelQuadUnsup_W[s] = temp_W;
    modelQuadUnsup_b[s] = temp_b;


    /*************************
    compute results use W*A+b
    *************************/

    // Atr
    DataMat modelOutTrainTemp(number_classes, N);
    modelOutTrainTemp = trans(temp_W) * temp_Atr + repmat_cpp(temp_b, 1, N);

    for (int j = 0; j < N; j++) {
      DataMat moTrj = colm(modelOutTrainTemp, j);
      DataMat temp_j =
      repmat_cpp(moTrj, 1, number_classes) -
      identity_matrix<Dtype>(number_classes);
      temp_j = pointwise_multiply(temp_j, temp_j);
      set_colm(modelOutTrainUnsup, j) =
      colm(modelOutTrainUnsup, j) + trans(sum_cpp(temp_j));
    }

    // Att
    DataMat temp_Att(d, N_test);
    temp_Att = rowm(Att, range(temp, temp + d - 1));

    DataMat modelOutTestTemp(number_classes, N_test);
    modelOutTestTemp = trans(temp_W) * temp_Att + repmat_cpp(temp_b, 1, N);

    for (int j = 0; j < N_test; j++) {
      DataMat moTej = colm(modelOutTestTemp, j);
      DataMat temp_j =
      repmat_cpp(moTej, 1, number_classes) -
      identity_matrix<Dtype>(number_classes);
      temp_j = pointwise_multiply(temp_j, temp_j);
      set_colm(modelOutTestUnsup, j) =
      colm(modelOutTestUnsup, j) + trans(sum_cpp(temp_j));
    }
      
  }// for s->S

  /***********************************************
  compute predictedLable Unsup and compute results
  ***********************************************/

  ///////// classify training samples
  IntVec predictedLableTrainUnsup(N);
  predictedLableTrainUnsup = 0;
  double sum_classify = 0;

  for (int i = 0; i < N; i++) {
    matrix<double, 0, 1> colm_temp;
    colm_temp = colm(modelOutTrainUnsup, i);
    predictedLableTrainUnsup(i) = min_idx_cpp(colm_temp);
    sum_classify += predictedLableTrainUnsup(i) == trls(i);
  }
  double CCRQuadTrainUnsup = sum_classify / N * 100;
  cout << "train_unsup:" << CCRQuadTrainUnsup << endl;

  ///////// classify testing samples
  IntVec predictedLableTestUnsup(N_test);
  predictedLableTestUnsup = 0;
  sum_classify = 0;

   for (int i = 0; i < N_test; i++) {
    matrix<double, 0, 1> colm_temp;
    colm_temp = colm(modelOutTestUnsup, i);
    predictedLableTestUnsup(i) = min_idx_cpp(colm_temp);
    sum_classify += predictedLableTestUnsup(i) == ttls(i);
  }
  double CCRQuadTestUnsup = sum_classify / N_test * 100;
  cout << "test_unsup:" << CCRQuadTestUnsup << endl;


  /*****************************************
  begin Sup learning, get DSup and new W b
  *****************************************/

  DataMat DSup(sum(n), d);

  OnlineSupTaskDrivDicLeaDecFusJointQuadC(XArr, outputVectorTrain, 
      n, d, DUnsup, DSup, modelQuadUnsup_W, modelQuadUnsup_b[0]);

  /*************************
  train sample admm feature
  *************************/

  DataMat Atr_sup;

  Atr_sup = zeros_matrix<Dtype>(d * S, N);

  ADMM_CG_xwt(DSup, XArr, n, S, lambda, rho, iterADMM, Atr_sup, tolCG, iterCG);

  /*************************
  test sample admm feature
  *************************/
  DataMat Att_sup;

  Att_sup = zeros_matrix<Dtype>(d * S, N_test);

  ADMM_CG_xwt(DSup, YArr, n, S, lambda, rho, iterADMM, Att_sup, tolCG, iterCG);

  /*************************************
  let admm feature multiple with W and b
  *************************************/

  DataMat modelOutTrainSup = zeros_matrix<Dtype>(number_classes, N);
  DataMat modelOutTestSup = zeros_matrix<Dtype>(number_classes, N_test);

#ifdef USE_OMP
#pragma omp parallel for
#endif
  for (int s = 0; s < S; ++s) {
    std::cout << "s:" << s << std::endl;

    int temp = d*s;

      // Atr -> modelOutTrainSup
    DataMat temp_Atr(d, N);
    temp_Atr = rowm(Atr_sup, range(temp, temp + d - 1));

    DataMat modelOutTrainTemp(number_classes, N);
    modelOutTrainTemp = trans(modelQuadUnsup_W[s]) * temp_Atr + repmat_cpp(modelQuadUnsup_b[s], 1, N);

    for (int j = 0; j < N; j++) {
      DataMat moTrj = colm(modelOutTrainTemp, j);

      DataMat temp_j =
      repmat_cpp(moTrj, 1, number_classes) -
      identity_matrix<Dtype>(number_classes);

      temp_j = pointwise_multiply(temp_j, temp_j);

      set_colm(modelOutTrainSup, j) =
      colm(modelOutTrainSup, j) + trans(sum_cpp(temp_j));
    } // for j

      // Att -> modelOutTestSup
    DataMat temp_Att(d, N_test);
    temp_Att = rowm(Att_sup, range(temp, temp + d - 1));

    DataMat modelOutTestTemp(number_classes, N_test);
    modelOutTestTemp = trans(modelQuadUnsup_W[s]) * temp_Att + repmat_cpp(modelQuadUnsup_b[s], 1, N);

    for (int j = 0; j < N_test; j++) {
      DataMat moTej = colm(modelOutTestTemp, j);
      DataMat temp_j =
      repmat_cpp(moTej, 1, number_classes) -
      identity_matrix<Dtype>(number_classes);
      temp_j = pointwise_multiply(temp_j, temp_j);
      set_colm(modelOutTestSup, j) =
      colm(modelOutTestSup, j) + trans(sum_cpp(temp_j));
    }//for j

  }//for s->S

  /*********************************************************************************
  modelOutTrainSup modelOutTestSup => predictLableTrainSup predictLableTestSup
  *********************************************************************************/
    // classify train samples
  IntVec predictedLableTrainSup(N);
  predictedLableTrainSup = 0;
  sum_classify = 0;

  for (int i = 0; i < N; i++) {
    matrix<double, 0, 1> colm_temp;
    colm_temp = colm(modelOutTrainSup, i);
    predictedLableTrainSup(i) = min_idx_cpp(colm_temp);
    sum_classify += predictedLableTrainSup(i) == trls(i);
  }
  double CCRQuadTrainSup = sum_classify / N * 100;
  cout << "train_Sup:" << CCRQuadTrainSup << endl;

    // classify testing samples
  IntVec predictedLableTestSup(N_test);
  predictedLableTestSup = 0;
  sum_classify = 0;

  for (int i = 0; i < N_test; i++) {
    matrix<double, 0, 1> colm_temp;
    colm_temp = colm(modelOutTestSup, i);
    predictedLableTestSup(i) = min_idx_cpp(colm_temp);
    sum_classify += predictedLableTestSup(i) == ttls(i);
  }
  double CCRQuadTestSup = sum_classify / N_test * 100;
  cout << "test_Sup:" << CCRQuadTestSup << endl;


} // Classification
