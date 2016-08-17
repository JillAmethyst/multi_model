#include "SGDMultiClassQuadC.hpp"

int SGDMultiClassQuadC(const DataMat& Atr, DataMat& outputVectorTrain,
  DataMat& temp_W, DataMat& temp_b) {
  // pass global parameters
  static MultimodalConfigParser<Dtype>& config =
    MultimodalConfigParser<Dtype>::Instance();
  CHECK(config.Initialized()) << "config has not been Initialzed";
  const Dtype nu = config.global_nuQuad();
  const int iter = config.global_iterQuad();
  const bool intercept = config.global_intercept();
  const int batchSize = config.global_batchSizeQuad();
  const Dtype ro = config.global_roQuad();
  const bool computeCost = config.global_computeCost();

  int N = Atr.nc();
  Dtype t0 = Dtype(N) / batchSize * iter / 10;
  int n = Atr.nr();
  int number_classes = outputVectorTrain.nr();
  DataMat W(number_classes, n);
  W = 0;
  DataMat b(number_classes, 1);
  b = 0;

  int step = 0;

  const IntVec permut = randperm_cpp(N);
  DataMat Atr_randperm = colm(Atr, permut);
  DataMat outputVectorTrain_randperm = colm(outputVectorTrain, permut);

  for (int i = 1; i <= iter; i++) {
    cout<< "SGD_iteration = "<< i<<endl;
    for (int t = 0; t < (N - N % batchSize); t += batchSize) {
      step += 1;

      DataMat temp =
        W * colm(Atr_randperm, range(t, t + batchSize - 1)) +
        repmat_cpp(b, 1, batchSize) -
        colm(outputVectorTrain_randperm, range(t, t + batchSize - 1));

      DataMat gradW = temp *
                        trans(colm(Atr_randperm, range(t, t + batchSize - 1))) /
                        batchSize +
                      nu * W;

      Dtype k1 = ro * t0 / step;
      Dtype learnRate = (ro > k1) ? k1 : ro;

      W = W - learnRate * gradW;
      LOG(INFO) << "Batch " << t;
    }
  }
  temp_W = trans(W);
  temp_b = b;

  return 0;
}
