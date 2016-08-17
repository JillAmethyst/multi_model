#include "OnlineSupTaskDrivDicLeaDecFusJointQuadC.hpp"

void OnlineSupTaskDrivDicLeaDecFusJointQuadC(const DataMat& XArr,
  const DataMat& Y, const IntVec& n, const int d, const DataMat& DUnsup,
  DataMat& DSup, std::vector<DataMat>& W, DataMat& b) {
  // pass global parameters
  static MultimodalConfigParser<Dtype>& config =
    MultimodalConfigParser<Dtype>::Instance();
  CHECK(config.Initialized()) << "config has not been Initialzed";
  const int iter = config.global_iterSupDic();
  const Dtype rho = config.global_rho();
  const bool intercept = config.global_intercept();
  const int batchSize = config.global_batchSize();
  const Dtype nuQuad = config.global_nuQuad();
  const int iterADMM = config.global_iterADMM();
  const Dtype lambda = config.global_lambda();
  const Dtype ro = config.global_ro();
  const bool ADMMwithCG = config.global_ADMMwithCG();


  const int sum_n = sum(n);
  const int S = n.size();
  const int N = XArr.nc();
  const int number_classes = Y.nr();

  DSup = DUnsup;

  // initialize W to small random numbers rather than setting all to zeros
  W.clear();
  W.resize(S);
  b = zeros_matrix<Dtype>(S, number_classes);
  for (int i = 0; i < S; ++i) {
    W[i] = 0.01 * randm(d, number_classes);
  }

  // learning rate of the stochastic gradient descent algorithm
  Dtype t0 = Dtype(N) / batchSize * iter / 10;

  const IntVec permut = randperm_cpp(N);
  DataMat XArr_randperm = colm(XArr, permut);
  DataMat Y_randperm = colm(Y, permut);

  int step = 0;
  std::vector<DataMat> BetaTemp2(S);
  for (int i = 0; i < S; ++i) {
    BetaTemp2[i] = zeros_matrix<Dtype>(d, batchSize);
  }
  DataMat Do = zeros_matrix<Dtype>(sum_n, S * d);
  // modified by Thomas Lee
  std::vector<DataMat> Atemp2(S);
  for (int i = 0; i < S; ++i) {
    Atemp2[i] = zeros_matrix<Dtype>(d, batchSize);
  }

  // gradient related
  DataMat gradD = zeros_matrix<Dtype>(sum_n, d);
  DataMat L = zeros_matrix<Dtype>(d * S, d);
  DataMat U = zeros_matrix<Dtype>(d * S, d);
  std::vector<DataMat> gradW(S);
  for (int i = 0; i < S; ++i) {
    gradW[i] = zeros_matrix<Dtype>(d, number_classes);
  }
  DataMat gradb = zeros_matrix<Dtype>(S, number_classes);

  // main loop
  for (int iteration = 0; iteration < iter; ++iteration) {
    for (int t = 0; t < (N - N % batchSize); t += batchSize) {
      step += 1;
      DataMat Atemp = zeros_matrix<Dtype>(d * S, batchSize);
      int temp = 0;
      for (int s = 0; s < S; ++s) {
        DataMat temp_L =
          factor_cpp(rowm(DSup, range(temp, temp + n(s) - 1)), rho);
        set_rowm(L, range(s * d, (s + 1) * d - 1)) = temp_L;
        set_rowm(U, range(s * d, (s + 1) * d - 1)) = trans(temp_L);
        set_subm(Do, range(temp, temp + n(s) - 1), range(s, S, S * d)) =
          rowm(DSup, range(temp, temp + n(s) - 1));
        temp += n(s);
      }
      DataMat DoTDoAll = trans(Do) * Do;

      for (int j = 0; j < batchSize; ++j) {
        DataMat XArr_sample = colm(XArr_randperm, j + t);
        DataMat YArr_sample = colm(Y_randperm, j + t);
        DataMat tmp_alpha = zeros_matrix<Dtype>(d * S, 1);
        ADMM_Dlib(DSup, XArr_sample, n, lambda, rho, L, U, iterADMM, tmp_alpha);
        set_colm(Atemp, j) = tmp_alpha;
        // GetBetaTemp2
        GetBetaTemp2(tmp_alpha, YArr_sample, W, b, DoTDoAll, d, S, j, lambda,
          Atemp2, BetaTemp2);
      }

      temp = 0;
      for (int s = 0; s < S; ++s) {
        // modified by Thomas Lee
        DataMat temp3 = BetaTemp2[s] * trans(Atemp2[s]);
        set_rowm(gradD, range(temp, temp + n(s) - 1)) =
          -1 * rowm(DSup, range(temp, temp + n(s) - 1)) * temp3 / batchSize +
          (subm(XArr_randperm, range(temp, temp + n(s) - 1),
             range(t, t + batchSize - 1)) *
              trans(BetaTemp2[s]) -
            rowm(DSup, range(temp, temp + n(s) - 1)) * trans(temp3)) /
            batchSize;
        temp += n(s);
      }

      temp = 0;
      for (int s = 0; s < S; ++s) {
        DataMat temp2 = trans(W[s]) * rowm(Atemp, range(temp, temp + d - 1)) +
                        repmat_cpp<DataMat>(trans(rowm(b, s)), 1, batchSize) -
                        colm(Y_randperm, range(t, t + batchSize - 1));
        gradW[s] =
          rowm(Atemp, range(temp, temp + d - 1)) * trans(temp2) / batchSize +
          nuQuad * W[s];
        if (intercept) {
          set_rowm(gradb, s) = trans(rowsum_cpp(temp2)) / batchSize;
        }
        temp += d;
      }

      // update W, b, D
      Dtype learnRate = std::min(ro, ro * t0 / step);
      for (int s = 0; s < S; ++s) {
        W[s] -= learnRate * gradW[s];
      }

      if (intercept) {
        b -= learnRate * gradb;
      }
      DSup -= learnRate * gradD;

      projectionDic(n, DSup);
    }
  }
}

void projectionDic(const IntVec& n, DataMat& dic) {
  const int S = n.nr();
  const int col_num = dic.nc();
  int temp = 0;
  for (int s = 0; s < S; ++s) {
    for (int j = 0; j < col_num; ++j) {
      DataVec current_atom =
        subm(dic, range(temp, temp + n(s) - 1), range(j, j));
      Dtype tmp_norm = sum(squared(current_atom));
      if (tmp_norm > Dtype(1)) {
        set_subm(dic, range(temp, temp + n(s) - 1), range(j, j)) =
          current_atom / std::sqrt(tmp_norm);
      }
    }
  }
}

void GetBetaTemp2(const DataVec& alpha_vec, const DataVec& current_Y,
  const std::vector<DataMat>& W, const DataMat& b, const DataMat& DoTDoAll,
  const int d, const int S, const int j, const Dtype lambda,
  std::vector<DataMat>& Atemp2, std::vector<DataMat>& BetaTemp2) {
  DataMat tmp_alpha = reshape(trans(alpha_vec), d, S);
  for (int s = 0; s < S; ++s) {
    set_colm(Atemp2[s], j) = colm(tmp_alpha, s);
  }
  DataVec temp_norm_squared(d);
  DataVec temp_norm(d);
  IntVec act_bin(d);
  DataMat temp_Gamma = zeros_matrix<Dtype>(d * S, d * S);
  act_bin = 0;
  int temp = 0;
  for (int row = 0; row < d; ++row) {
    DataRowVec current_row = rowm(tmp_alpha, row);
    Dtype current_row_norm = sum(squared(current_row));
    temp_norm_squared(row) = current_row_norm;
    temp_norm(row) = std::sqrt(current_row_norm);
    if (current_row_norm > 1e-8) {
      act_bin(row) = 1;
      DataMat tmp_gamma =
        (identity_matrix<Dtype>(S) -
          trans(current_row) * current_row / current_row_norm) /
        temp_norm(row);
      set_subm(temp_Gamma, range(temp, temp + S - 1),
        range(temp, temp + S - 1)) = tmp_gamma;
      temp += S;
    }
  }
  DataMat Gamma = subm(temp_Gamma, range(0, temp - 1), range(0, temp - 1));
  int num_act = sum(act_bin);
  IntVec act = binary2cardinal_cpp(act_bin);
  DataVec Beta(d * S);
  DataMat gradLs_actAlphaVec = zeros_matrix<Dtype>(num_act, S);
  DataMat alpha_act = rowm(tmp_alpha, act);
  for (int s = 0; s < S; ++s) {
    DataMat W_act = rowm(W[s], act);
    set_colm(gradLs_actAlphaVec, s) =
      W_act *
      (trans(W_act) * colm(alpha_act, s) + trans(rowm(b, s)) - current_Y);
  }
  IntMat acts_bin = repmat_cpp<IntMat>(act_bin, 1, S);
  IntVec acts = binary2cardinal_cpp(reshape_to_column_vector(acts_bin));
  DataMat DoTDo = subm(DoTDoAll, acts, acts);
  DataVec gradLs_actAt_vec = reshape_to_column_vector(gradLs_actAlphaVec);

  DataMat l_side = (DoTDo + lambda * Gamma);
  lu_decomposition<DataMat> ld(l_side);
  set_rowm(Beta, acts) = ld.solve(gradLs_actAt_vec);
  DataMat tmp_beta = reshape(Beta, d, S);
  for (int s = 0; s < S; ++s) {
    set_colm(BetaTemp2[s], j) = colm(tmp_beta, s);
  }
}