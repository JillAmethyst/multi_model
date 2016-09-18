#include "finalADMMfeature.hpp"
#include <mpi.h>

int finalADMMfeature(const DataMat& XArr, const DataMat& YArr, DataMat& DUnsup, const IntVec n, const int d, const Dtype lambda, const Dtype rho, const int iterADMM, const Dtype tolCG, const int iterCG, const bool ADMMwithCG, DataMat& Atr, DataMat& Att){
	const int S = n.size();
	const int N = XArr.nc();
	const int N_test = YArr.nc();

	#define root 0

	int rank, ProcSize;
	MPI_Comm_size(MPI_COMM_WORLD,&ProcSize);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

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
	if (rank == root) {
    Atr = zeros_matrix<Dtype>(d * S, N);
  }

  DataMat Xtr =zeros_matrix<Dtype>(DUnsup.nr(),N/ProcSize);
  DataMat Alpha_tr = zeros_matrix<Dtype>(S * d,N/ProcSize);

  MPI_Scatter(XArr.begin(),XArr.nr()*N/ProcSize,MPI_DOUBLE,Xtr.begin(),Xtr.nr()*N/ProcSize,MPI_DOUBLE,root,MPI_COMM_WORLD);
  cout<<"train_MPI_Scatter: "<<rank<<endl;

  if ( ADMMwithCG ){
    ADMM_CG_xwt(DUnsup, Xtr, n, S, lambda, rho, iterADMM, Alpha_tr, tolCG, iterCG);
    cout<<"train_ADMM_CG_xwt: "<<rank<<endl;
  }else{ 
    ADMM_Dlib(DUnsup, Xtr, n, lambda, rho, L, U, iterADMM, Alpha_tr);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"train_MPI_Barrier: "<<rank<<endl;

  MPI_Gather(Alpha_tr.begin(),N/ProcSize*Alpha_tr.nr(),MPI_DOUBLE,Atr.begin(),N/ProcSize*S*d,MPI_DOUBLE,root,MPI_COMM_WORLD);
  cout<<"train_MPI_Gather: "<<rank<<endl;

  /*************************
  test sample admm feature
  *************************/
  if (rank == root) {
    Att = zeros_matrix<Dtype>(d * S, N_test);
  }

  DataMat Xtt =zeros_matrix<Dtype>(DUnsup.nr(),N_test/ProcSize);
  DataMat Alpha_tt = zeros_matrix<Dtype>(S * d,N_test/ProcSize);

  MPI_Scatter(YArr.begin(),YArr.nr()*N_test/ProcSize,MPI_DOUBLE,Xtt.begin(),Xtt.nr()*N_test/ProcSize,MPI_DOUBLE,root,MPI_COMM_WORLD);
  cout<<"test_MPI_Scatter: "<<rank<<endl;

  if ( ADMMwithCG ){
    ADMM_CG_xwt(DUnsup, Xtt, n, S, lambda, rho, iterADMM, Alpha_tt, tolCG, iterCG);
    cout<<"test_ADMM_CG_xwt: "<<rank<<endl;
  }
  else 
    ADMM_Dlib(DUnsup, Xtt, n, lambda, rho, L, U, iterADMM, Alpha_tt);

  MPI_Barrier(MPI_COMM_WORLD);
  cout<<"test_MPI_Barrier: "<<rank<<endl;

  MPI_Gather(Alpha_tt.begin(),Alpha_tt.nr()*N_test/ProcSize,MPI_DOUBLE,Att.begin(),N_test/ProcSize*S*d,MPI_DOUBLE,root,MPI_COMM_WORLD);
  cout<<"test_MPI_Gather: "<<rank<<endl;

  return 0;
}