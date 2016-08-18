#include "ClassificationMultiClassDecFuxJoint.hpp"
#include <mpi.h>

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
  const bool ADMMwithCG = config.global_ADMMwithCG();

  const Dtype tolCG = 1e-5;
  const int iterCG = 20;

  IntVec uniqtrls = unique_cpp(trls); 
  int number_classes = uniqtrls.nr();

  DataMat DUnsup(sum(n), d);

  /*************************
          start MPI
  *************************/

  #define root 0

  int rank, ProcSize;
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD,&ProcSize);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  /*************************
     Unsup D computation
  *************************/

  OnlineUnsupTaskDrivDicLeaJointC(XArr, trls, n, d, DUnsup, rank, ProcSize);

  // if (rank == root) {
  //   WriteDataToFile("DUnsupAtom4Carc_500People_x_3200_500.dat", DUnsup);
  // }

  /*************************
     L U computation
  *************************/

  DataMat L(d * S, d);
  DataMat U(d * S, d);

  if ( ! ADMMwithCG ){
    int temp = 0;
    for (int s = 0; s < S; s++) {
      DataMat temp_L = factor_cpp(rowm(DUnsup, range(temp, temp + n(s) - 1)), rho);
      set_rowm(L, range(s * d, (s + 1) * d - 1)) = temp_L;
      set_rowm(U, range(s * d, (s + 1) * d - 1)) = trans(temp_L);
      temp += n(s);
    }
  }


  /*************************
  train sample admm feature
  *************************/
  DataMat Atr;

  if (rank == root) {
    Atr = zeros_matrix<Dtype>(d * S, N);
  }

  DataMat Xtr =zeros_matrix<Dtype>(DUnsup.nr(),N/ProcSize);
  DataMat Alpha_tr = zeros_matrix<Dtype>(S * d,N/ProcSize);

  MPI_Scatter(XArr.begin(),XArr.nr()*N/ProcSize,MPI_DOUBLE,Xtr.begin(),Xtr.nr()*N/ProcSize,MPI_DOUBLE,root,MPI_COMM_WORLD);

  if ( ADMMwithCG )
    ADMM_CG_xwt(DUnsup, Xtr, n, S, lambda, rho, iterADMM, Alpha_tr, tolCG, iterCG);
  else 
    ADMM_Dlib(DUnsup, Xtr, n, lambda, rho, L, U, iterADMM, Alpha_tr);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Gather(Alpha_tr.begin(),Alpha_tr.nr()*N/ProcSize,MPI_DOUBLE,Atr.begin(),S*d*N/ProcSize,MPI_DOUBLE,root,MPI_COMM_WORLD);


  /*************************
  test sample admm feature
  *************************/
  DataMat Att;

  if (rank == root) {
    Att = zeros_matrix<Dtype>(d * S, N_test);
  }

  DataMat Xtt =zeros_matrix<Dtype>(DUnsup.nr(),N_test/ProcSize);
  DataMat Alpha_tt = zeros_matrix<Dtype>(S * d,N_test/ProcSize);

  MPI_Scatter(YArr.begin(),YArr.nr()*N_test/ProcSize,MPI_DOUBLE,Xtt.begin(),Xtt.nr()*N_test/ProcSize,MPI_DOUBLE,root,MPI_COMM_WORLD);

  if ( ADMMwithCG )
    ADMM_CG_xwt(DUnsup, Xtt, n, S, lambda, rho, iterADMM, Alpha_tt, tolCG, iterCG);
  else 
    ADMM_Dlib(DUnsup, Xtt, n, lambda, rho, L, U, iterADMM, Alpha_tt);

  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Gather(Alpha_tt.begin(),Alpha_tt.nr()*N_test/ProcSize,MPI_DOUBLE,Att.begin(),S*d*N_test/ProcSize,MPI_DOUBLE,root,MPI_COMM_WORLD);


  /*************************
       compute W and b
  *************************/
  if ( rank == root ) {
    DataMat outputVectorTrain = zeros_matrix<Dtype>(number_classes, N);

    for (int j = 0; j < N; ++j) {
      outputVectorTrain(trls(j), j) = 1;
    }

    DataMat modelOutTrainUnsup = zeros_matrix<Dtype>(number_classes, N);
    DataMat modelOutTestUnsup = zeros_matrix<Dtype>(number_classes, N_test);

    DataMat modelQuadUnsup_W = zeros_matrix<Dtype>(d * S, number_classes);
    DataMat modelQuadUnsup_b = zeros_matrix<Dtype>(number_classes, S);


#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int s = 0; s < S; ++s) {
      int temp = d*s;
      std::cout << "s:" << s << std::endl;
      DataMat temp_Atr(d, N);
      DataMat temp_W = zeros_matrix<Dtype>(d, number_classes);
      DataMat temp_b = zeros_matrix<Dtype>(number_classes, 1);

      temp_Atr = rowm(Atr, range(temp, temp + d - 1));

      SGDMultiClassQuadC(temp_Atr, outputVectorTrain, temp_W, temp_b);

      set_rowm(modelQuadUnsup_W, range(temp, temp + d - 1)) = temp_W;
      set_colm(modelQuadUnsup_b, s) = temp_b;


      /*******************************
      compute modelOut Unsup use W*A+b
      *******************************/

      /////////////// Atr
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

      /////////////// Att
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
    double sum = 0;

    for (int i = 0; i < N; i++) {
      matrix<double, 0, 1> colm_temp;
      colm_temp = colm(modelOutTrainUnsup, i);
      predictedLableTrainUnsup(i) = min_idx_cpp(colm_temp);
      sum += predictedLableTrainUnsup(i) == trls(i);
    }
    double CCRQuadTrainUnsup = sum / N * 100;
    cout << "train_unsup:" << CCRQuadTrainUnsup << endl;

    ///////// classify testing samples
    IntVec predictedLableTestUnsup(N_test);
    predictedLableTestUnsup = 0;
    sum = 0;

    for (int i = 0; i < N_test; i++) {
      matrix<double, 0, 1> colm_temp;
      colm_temp = colm(modelOutTestUnsup, i);
      predictedLableTestUnsup(i) = min_idx_cpp(colm_temp);
      sum += predictedLableTestUnsup(i) == ttls(i);
    }
    double CCRQuadTestUnsup = sum / N_test * 100;
    cout << "test_unsup:" << CCRQuadTestUnsup << endl;

  } // if root

  MPI_Finalize();

} // classification
