#include "OnlineUnsupTaskDrivDicLeaJointC.hpp"
#include <mpi.h>

int OnlineUnsupTaskDrivDicLeaJointC(const DataMat& XArr, const IntVec& trls,
  const IntVec& n, const int d, DataMat& D, const int N) {
  cout <<" start "<<endl;
  int rank, ProcSize;
  MPI_Comm_size(MPI_COMM_WORLD,&ProcSize);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  /**********************  
  pass global parameters
  **********************/
  static MultimodalConfigParser<Dtype>& config =
  MultimodalConfigParser<Dtype>::Instance();
  CHECK(config.Initialized()) << "config has not been Initialzed";
  const int iter = config.global_iterUnsupDic();
  const int batchSize_config = config.global_batchSize();
  const int batchSize = batchSize_config - batchSize_config%ProcSize;
  const int iterADMM = config.global_iterADMM();
  const Dtype rho = config.global_rho();
  const Dtype lambda = config.global_lambda();
  const bool computeCost = config.global_computeCost();
  const bool ADMMwithCG = config.global_ADMMwithCG();

  const Dtype tolCG = 1e-5;
  const int iterCG = 20;

  const IntVec uniqtrls = unique_cpp(trls);
  const int number_classes = uniqtrls.size();
  const int sum_n = sum(n);
  const int S = n.size();
  // const int N = XArr.nc();
  int dicIterations = 10;

  CHECK(d <= N) << "Number of dictionary columns should be smaller than or "
  "equal to the number of train samples.";

  D.set_size(sum_n, d);
  const int numAtomPerClass = d / number_classes;
  CHECK(numAtomPerClass >= 1)
  << "at least one atom per class is required, increase dictionary size d";

  /******************************************************************
    Initialize D using (randomy taken) train samples from each class
  ******************************************************************/
  int temp = 0;
  for (int i = 0; i < number_classes; ++i) {
    const IntVec tempIndex = find_cpp(trls == uniqtrls(i));
    const IntVec permut2 = randperm_cpp(tempIndex.size());
    // Thomas Lee
    CHECK(tempIndex.size() >= numAtomPerClass) <<"the " <<i+1<< "people hase Not Enough samples to init dictionary.";

    set_colm(D, range(temp, temp + numAtomPerClass - 1)) =
    colm(XArr, rowm(tempIndex, range(0, numAtomPerClass - 1)));
    temp += numAtomPerClass;
  }

  if (d > number_classes * numAtomPerClass) {
    const IntVec permut = randperm_cpp(N);
    set_colm(D, range(temp, d - 1)) =
    colm(XArr, rowm(permut, range(0, d - temp - 1)));
  }

  std::vector<DataMat> A_past;
  A_past.resize(S);
  for (int i = 0; i < S; ++i) {
    A_past[i] = zeros_matrix<Dtype>(d, d);
  }
  DataMat B_past = zeros_matrix<Dtype>(sum_n, d);
  const IntVec permut = randperm_cpp(N);
  DataMat XArr_randperm = colm(XArr, permut);

   DataMat L = zeros_matrix<Dtype>(d * S, d);
   DataMat U = zeros_matrix<Dtype>(d * S, d);


#define root 0

  for (int iteration = 0; iteration < iter; ++iteration) {
    cout<< "Unsuper Dic iteration = "<< iteration<<endl; 
    if (iteration > 1) dicIterations = 1;
    for (int t = 0; t < (N - N % batchSize); t += batchSize) {
      // fix D, optimize A
if(! ADMMwithCG) {
      temp = 0;
       for (int s = 0; s < S; s++) {
        DataMat temp_L = factor_cpp(rowm(D, range(temp, temp + n(s) - 1)), rho);
        set_rowm(L, range(s * d, (s + 1) * d - 1)) = temp_L;
        set_rowm(U, range(s * d, (s + 1) * d - 1)) = trans(temp_L);
        temp += n(s);
      }
}
     DataMat Atemp, XArr_sample; 
     if(rank== root) {
       Atemp = zeros_matrix<Dtype>(S * d, batchSize);
       XArr_sample = colm(XArr_randperm, range(t, batchSize + t - 1));
     }

     DataMat X0 = zeros_matrix<Dtype> (D.nr(),batchSize/ProcSize);
     DataMat Alpha0 = zeros_matrix<Dtype>(S * d,batchSize/ProcSize);

     MPI_Scatter(XArr_sample.begin(),XArr_sample.nr()*batchSize/ProcSize,MPI_DOUBLE,X0.begin(),X0.nr()*batchSize/ProcSize,MPI_DOUBLE,root,MPI_COMM_WORLD);

//cout <<"rank="<<rank<<endl<<subm(X0,rectangle(0,0,5,5))<<endl;
if(ADMMwithCG)
     ADMM_CG_xwt(D, X0, n, S, lambda, rho, iterADMM, Alpha0, tolCG, iterCG);
//  ADMM_CG(D, X0, S, lambda, rho, iterADMM, Alpha0, n);
else 
  ADMM_Dlib(D, X0, n, lambda, rho, L, U, iterADMM, Alpha0);

     MPI_Barrier(MPI_COMM_WORLD);

//for(int testI = 0 ; testI < Alpha0.size();testI++) Alpha0(testI) = rank*10000 + testI  ;
//cout<<subm(Alpha0,range(0,2),range(0,5))<<endl;
     MPI_Gather(Alpha0.begin(),Alpha0.nr()*batchSize/ProcSize,MPI_DOUBLE,Atemp.begin(),S*d*batchSize/ProcSize,MPI_DOUBLE,root,MPI_COMM_WORLD);

     if(rank== root){
//for(int i=0;i<ProcSize;i++)
//cout <<"Alpha0="<<endl<<subm(Atemp,range(0,3),range(i*batchSize/ProcSize, i*batchSize/ProcSize+10))<<endl;

      temp = 0;
      for (int s = 0; s < S; ++s) {
        A_past[s] += rowm(Atemp, range(s * d, (s + 1) * d - 1)) *
        trans(rowm(Atemp, range(s * d, (s + 1) * d - 1))) /
        batchSize;
        set_rowm(B_past, range(temp, temp + n(s) - 1)) =
        rowm(B_past, range(temp, temp + n(s) - 1)) +
        subm(XArr_randperm, range(temp, temp + n(s) - 1),
          range(t, t + batchSize - 1)) *
        trans(rowm(Atemp, range(s * d, (s + 1) * d - 1))) / batchSize;
        temp += n(s);
      }
cout<<"copute cost"<<endl;
      // compute cost
      if (computeCost) {
        Dtype current_cost = ComputeCost(
          XArr_randperm, D, Atemp, n, lambda, d, S, N, batchSize, t);
        LOG(INFO) << "Batch " << t << " : " << current_cost;
      }

      // fix A, optimize D
      temp = 0;
      for (int s = 0; s < S; ++s) {
        DataMat D_temp = rowm(D, range(temp, temp + n(s) - 1));
        for (int l = 0; l < dicIterations; ++l) {
          for (int j = 0; j < d; ++j) {
            if (A_past[s](j, j) > 1e-12) {
              set_colm(D_temp, j) =
              colm(D_temp, j) +
              (subm(B_past, range(temp, temp + n(s) - 1), range(j, j)) -
                D_temp * colm(A_past[s], j)) /
              A_past[s](j, j);
              Dtype tempNorm = frobenius_norm_cpp(colm(D_temp, j));
              if (tempNorm > 1) {
                set_colm(D_temp, j) = colm(D_temp, j) / tempNorm;
              }
            }
          }
        }
        set_rowm(D, range(temp, temp + n(s) - 1)) = D_temp;
        temp += n(s);
      }
    }//if rank = root 
  }
}



//cout <<"Lxd end "<<endl;  
return OnlineUnsupTaskDrivDicLeaJointC_OK;
}
