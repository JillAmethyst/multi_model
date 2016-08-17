#include "ClassificationMultiClassDecFuxJoint.hpp"

void ClassificationMultiClassDecFuxJoint(const DataMat& XArr,
  const IntVec& trls, const DataMat& YArr, const IntVec& ttls, const int N,
  const int N_test, const int d, const int S, const IntVec& n) {
  // pass global parameters
  static MultimodalConfigParser<Dtype>& config =
    MultimodalConfigParser<Dtype>::Instance();
  CHECK(config.Initialized()) << "config has not been Initialzed";
  const Dtype rho = config.global_rho();
  const Dtype lambda = config.global_lambda();
  const int iterADMM = config.global_iterADMM();

  const Dtype tolCG = 1e-5;
  const int iterCG = 20;

  IntVec uniqtrls = unique_cpp(trls);
  // cout<<"uniqtrls = "<<uniqtrls<<endl;
  int number_classes = uniqtrls.nr();

  DataMat DUnsup(sum(n), d);
  // DataMat L(d * S, d);
  // DataMat U(d * S, d);
  // const char* a1_filename = "DUnsup.dat";
  // LoadDataFromFile(a1_filename, DUnsup);

  int rank, ProcSize;

  OnlineUnsupTaskDrivDicLeaJointC(XArr, trls, n, d, DUnsup, rank, ProcSize);

  //  cout<<"DUnsup = "<<subm(DUnsup,rectangle(0,0,4,4))<<endl;
  //  WriteDataToFile("DUnsup.dat", DUnsup);   



  //******
  // // DataMat xarr_rand(sum(n), 297);
  // const char* a1_filename = "DUnsup.dat";
  // // const char* a2_filename = "xarr_test.dat";
  // LoadDataFromFile(a1_filename, DUnsup);
  // // LoadDataFromFile(a2_filename, xarr_rand);
  // cout<<"DUnsup = "<<subm(DUnsup,rectangle(0,0,4,4))<<endl;
  // cout<<"xarr_rand = "<<subm(xarr_rand,rectangle(0,0,4,4))<<endl;
  //******


  /*************************
  train sample admm feature
  *************************/
  DataMat Atr;

  Atr = zeros_matrix<Dtype>(d * S, N);

    
  // cout<<"Atr"<<endl;
  // ADMM_CG_xwt(DUnsup, Xtr, n, S, lambda, rho, iterADMM, Alpha_tr, tolCG, iterCG);
    ADMM_CG_xwt(DUnsup, XArr, n, S, lambda, rho, iterADMM, Atr, tolCG, iterCG);
  //  ADMM_Dlib(DUnsup, XArr, n, lambda, rho, L, U, iterMax, Atr);//LU-no-mpi

  //  cout<<"Atr = "<<subm(Atr,rectangle(0,0,4,4))<<endl;
  //  WriteDataToFile("Atr.dat", Atr);    

  /*************************
  test sample admm feature
  *************************/
  DataMat Att;

  // if (rank == root) {
    Att = zeros_matrix<Dtype>(d * S, N_test);
  // }


 //  cout<<"Att"<<endl;
	// ADMM_CG_xwt(DUnsup, Xtt, n, S, lambda, rho, iterADMM, Alpha_tt, tolCG, iterCG);
    ADMM_CG_xwt(DUnsup, YArr, n, S, lambda, rho, iterADMM, Att, tolCG, iterCG);//no-mpi
  //  ADMM_Dlib(DUnsup, YArr, n, lambda, rho, L, U, iterMax, Att);//LU-no-mpi

 //   cout<<"Att = "<<subm(Att,rectangle(0,0,4,4))<<endl;
 //   WriteDataToFile("Att.dat", Att);    



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

    for(int s = 0; s<S; i++){
      modelQuadUnsup_W = zeros_matrix<Dtype>(d, number_classes);
      modelQuadUnsup_b = zeros_matrix<Dtype>(number_classes,1);
    }
    // DataMat modelQuadUnsup_W = zeros_matrix<Dtype>(d * S, number_classes);
    // DataMat modelQuadUnsup_b = zeros_matrix<Dtype>(number_classes, S);

  //*******
  // const char* a1_filename = "modelQuadUnsup_W.dat";
  // const char* a2_filename = "Atr.dat";
  // const char* a3_filename = "Att.dat";
  // LoadDataFromFile(a1_filename, modelQuadUnsup_W);
  // LoadDataFromFile(a2_filename, Atr);
  // LoadDataFromFile(a3_filename, Att);
  // cout<<"modelQuadUnsup_W =
  // "<<subm(modelQuadUnsup_W,rectangle(0,0,4,4))<<endl;
  // cout<<"Atr = "<<subm(Atr,rectangle(0,0,4,4))<<endl;
  // cout<<"Att = "<<subm(Att,rectangle(0,0,4,4))<<endl;
  //*******/


    
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

      modelQuadUnsup_W[s] = temp_W;
      modelQuadUnsup_b[s] = temp_b;

      // set_rowm(modelQuadUnsup_W, range(temp, temp + d - 1)) = temp_W;
      // set_colm(modelQuadUnsup_b, s) = temp_b;


      /*************************
      compute results use W*A+b
      *************************/

      DataMat modelOutTrainTemp(number_classes, N);

    // Atr

    //********
    // temp_W = rowm(modelQuadUnsup_W, range(temp, temp + d - 1));
    // temp_Atr = rowm(Atr, range(temp, temp + d - 1));
    // cout<<"temp_W = "<<subm(temp_W,rectangle(0,0,4,4))<<endl;
    // cout<<"temp_Atr = "<<subm(temp_Atr,rectangle(0,0,4,4))<<endl;
    //********

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

  // classify training samples
    IntVec predictedLableTrainUnsup(N);
    predictedLableTrainUnsup = 0;
    double sum = 0;

  //******
  // const char* a1_filename = "modelOutTrainUnsup.dat";
  // LoadDataFromFile(a1_filename, modelOutTrainUnsup);
  // cout<<"modelOutTrainUnsup="<<subm(modelOutTrainUnsup,rectangle(0,0,4,4))<<endl;
  //******

    for (int i = 0; i < N; i++) {
      matrix<double, 0, 1> colm_temp;
      colm_temp = colm(modelOutTrainUnsup, i);
      predictedLableTrainUnsup(i) = min_idx_cpp(colm_temp);
      sum += predictedLableTrainUnsup(i) == trls(i);
    }
    double CCRQuadTrainUnsup = sum / N * 100;
    cout << "train_unsup:" << CCRQuadTrainUnsup << endl;

  // classify testing samples
    IntVec predictedLableTestUnsup(N_test);
    predictedLableTestUnsup = 0;
    sum = 0;

  //******
  // const char* a1_filename = "modelOutTestUnsup.dat";
  // LoadDataFromFile(a1_filename, modelOutTestUnsup);
  // cout<<"modelOutTestUnsup="<<subm(modelOutTestUnsup,rectangle(0,0,4,4))<<endl;
  //******

    for (int i = 0; i < N_test; i++) {
      matrix<double, 0, 1> colm_temp;
      colm_temp = colm(modelOutTestUnsup, i);
      predictedLableTestUnsup(i) = min_idx_cpp(colm_temp);
      sum += predictedLableTestUnsup(i) == ttls(i);
    }
    double CCRQuadTestUnsup = sum / N_test * 100;
    cout << "test_unsup:" << CCRQuadTestUnsup << endl;


    //**************************************************
    //  begin Sup learning
    //**************************************************

    DataMat DSup(sum(n), d);

    OnlineSupTaskDrivDicLeaDecFusJointQuadC(XArr, outputVectorTrain, 
      n, d, DUnsup, DSup, W, b);

    


}
