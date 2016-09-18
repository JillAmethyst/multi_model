#include "computeWandB.hpp"

int computeWandB(const int number_classes, int S, const int N, const int N_test, const int d, const IntVec& trls, const IntVec& ttls, DataMat& Atr, DataMat& Att, std::vector<DataMat>& modelQuadUnsup_W, std::vector<DataMat>& modelQuadUnsup_b){
	#define root 0

	int rank, ProcSize;
  MPI_Comm_size(MPI_COMM_WORLD,&ProcSize);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if ( rank == root ) {
  	DataMat outputVectorTrain = zeros_matrix<Dtype>(number_classes, N);

  	for (int j = 0; j < N; ++j) {
    outputVectorTrain(trls(j), j) = 1;
  	}

    DataMat modelOutTrainUnsup = zeros_matrix<Dtype>(number_classes, N);
    DataMat modelOutTestUnsup = zeros_matrix<Dtype>(number_classes, N_test);

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
    	DataMat temp_W = zeros_matrix<Dtype>(d, number_classes);
    	DataMat temp_b = zeros_matrix<Dtype>(number_classes, 1);

    	temp_Atr = rowm(Atr, range(temp, temp + d - 1));
    	SGDMultiClassQuadC(temp_Atr, outputVectorTrain, temp_W, temp_b);

    	modelQuadUnsup_W[s] = temp_W;
    	modelQuadUnsup_b[s] = temp_b;

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

  } // if root

  return 0;
}