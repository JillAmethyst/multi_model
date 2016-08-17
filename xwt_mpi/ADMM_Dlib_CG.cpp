#include "ADMM_Dlib.hpp"
#ifdef USE_OMP
#include <omp.h>
#endif
using namespace std;
using namespace dlib;
/*********************
L and U Dlib version
*********************/

void ADMM_Dlib(const DataMat& dlibD, const DataMat& dlibX, const IntVec& n,
  const Dtype lambda, const Dtype rho, const DataMat& dlibL,
  const DataMat& dlibU, const int iterMax, DataMat& alpha) {
  //ADMM(dlibD, dlibX, n, lambda, rho, dlibL, dlibU, iterMax, n.nr(), alpha);
  DataMat Dt = trans(dlibD);
  DataMat DtD = Dt*dlibD;
  DataMat DtX = Dt*dlibX;
  ADMM(dlibD, dlibX, n, lambda, rho, dlibL, dlibU, iterMax, n.nr(), alpha); 
}

void ADMM(const DataMat& Dictionarys, const DataMat& X, const IntVec& n,
  const Dtype Lambda, const Dtype rho, const DataMat& L, const DataMat& U,
  const int iterMax, const int S, DataMat& alpha) {
  int i;
  int sum_n = Dictionarys.nr();
  int d = Dictionarys.nc();
  int SampleNum = X.nc();

  Dtype lambda = Lambda / rho;

#ifdef USE_OMP
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < SampleNum; i++) {
    Dtype scale;
    int iter, temp, s, m;

    DataMat z(d, S);
    DataMat u(d, S);
    DataMat B(d, S);
    z = 0;
    u = 0;

    DataMat DtX(d, S);
    DataMat q(d, S);
    temp = 0;
    for (s = 0; s < S; ++s) {
      set_colm(DtX, s) =
        trans(rowm(Dictionarys, range(temp, temp + n(s) - 1))) *
        subm(X, range(temp, temp + n(s) - 1), range(i, i));
      temp += n(s);
    }

    for (iter = 0; iter < iterMax; iter++) {
      q = DtX + rho * (z - u);
      for (s = 0; s < S; s++) {
        set_colm(B, s) =
          LltSolveDlib(rowm(L, range(d * s, d * s + d - 1)), colm(q, s));
      }
      u += B;
      DataVec norm_rows = rownorm_cpp(u);
      for (m = 0; m < d; ++m) {
        scale = (norm_rows(m) > lambda)
                  ? (norm_rows(m, 0) - lambda) / norm_rows(m, 0)
                  : 0;
        set_rowm(z, m) = scale * rowm(u, m);
      }
      // u update
      u -= z;
    }

    set_colm(alpha, i) = reshape_to_column_vector(trans(z));
  }
}

DataMat LltSolveDlib(const DataMat& L, const DataMat& b) {
  DataMat X(b);
  using namespace blas_bindings;
  // solve L*y=b
  triangular_solver(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, L, X);
  // solve L'*X=y
  triangular_solver(CblasLeft, CblasLower, CblasTrans, CblasNonUnit, L, X);
  return X;
}


/*********************
   CG Eigen version
*********************/

void ADMM_CG(DataMat& D, DataMat& XArr, const int S, const Dtype Lambda,
  const Dtype rho, const int iterMax, DataMat& alpha, const IntVec& n) {
  // deprecated lxd
  Dtype d = D.nc();
  int temp = 0;
  DataMat DtD = zeros_matrix<Dtype>(d * S, d);
  DataMat DtX = zeros_matrix<Dtype>(d * S, XArr.nc());
  for (int s = 0; s < S; ++s) {
   matrix<long> current_range = range(temp, temp + n(s) - 1);
   set_rowm(DtD, range(s * d, (s + 1) * d - 1)) = trans(rowm(D, current_range)) * rowm(D, current_range) + rho * identity_matrix<Dtype>(d);
   set_rowm(DtX, range(s * d, (s + 1) * d - 1)) = trans(rowm(D, current_range)) * rowm(XArr, current_range);
   temp += n(s);
  }
  Dlib2Eigen(DtD, dtd_e);
  Dlib2Eigen(DtX, dtx_e);
  Dlib2Eigen(alpha, alpha_e);
  ADMM_ConjugateGradient(dtd_e, dtx_e, S, Lambda, rho, iterMax, alpha_e);
}

void ADMM_ConjugateGradient(const EigenMap& dtd_e, const EigenMap& dtx_e,
  const int S, const Dtype Lambda, const Dtype rho, const int iterMax,
  EigenMap& alpha_e) {
  int i;
  const int SampleNum = dtx_e.cols();
  const int d = dtd_e.cols();

  // cout << "dtd:\n" << dtd_e << endl;

  Dtype lambda = Lambda / rho;
  std::vector<ConjugateGradient<EigenMat, Lower | Upper>*> cg_pointer_vec(S);
  for (int s = 0; s < S; ++s) {
    cg_pointer_vec[s] = new ConjugateGradient<EigenMat, Lower | Upper>;
    cg_pointer_vec[s]->compute(dtd_e.middleRows(s * d, d));
  }

#ifdef USE_OMP
#pragma omp parallel for private(i)
#endif
  for (i = 0; i < SampleNum; ++i) {
    Dtype scale;

    EigenMap z(alpha_e.data() + i * d * S, d, S);
    EigenMat u = EigenMat::Zero(d, S);
    EigenMat B = EigenMat::Zero(d, S);
    EigenMat q = EigenMat::Zero(d, S);
    EigenVec tmp_dtx = dtx_e.col(i);
    EigenMap dtx(tmp_dtx.data(), d, S);

    for (int iter = 0; iter < iterMax; iter++) {
      q = dtx + rho * (z - u);
      for (int s = 0; s < S; s++) {
        B.col(s) = cg_pointer_vec[s]->solve(q.col(s));
      }
      // cout << "B:\n" << B << endl;
      u += B;
      EigenVec norm_rows = u.rowwise().norm();
      for (int m = 0; m < d; ++m) {
        scale =
          (norm_rows(m) > lambda) ? (norm_rows(m) - lambda) / norm_rows(m) : 0;
        z.row(m) = scale * u.row(m);
      }
      // u update
      u -= z;
    }
  }

  // cleanup cg_pointer_vec
  for (int s = 0; s < S; ++s) {
    delete cg_pointer_vec[s];
  }
}


/**************************
   CG Eigen manual version
**************************/
//#include <vector>
void ADMM_CG_xwt(DataMat& D, DataMat& XArr, const IntVec& n, const int S,
  const Dtype Lambda, const Dtype rho, const int iterMax, DataMat& alpha, const Dtype& tolCG, const int iterCG) {
  // xwt
  Dtype d = D.nc();
  int temp = 0;

  std::vector<DataMat> DtVector(S),DVector(S);
  for (int s = 0; s < S; ++s) {
    matrix<long> current_range = range(temp, temp + n(s) - 1);
    DtVector[s] = trans(rowm(D, current_range)) ;
    DVector[s] = rowm(D, current_range);
    temp += n(s);
  }

  Dtype lambda = Lambda / rho;
  int i;
  const int SampleNum = XArr.nc();

    DataMat u(d, S);
    DataMat B (d, S);
    DataMat q (d, S);
    DataMat dtx( d, S);
    DataMat z(d, S); 

#ifdef USE_OMP
#pragma omp parallel for private(i,temp,z,u,B,q,dtx)
#endif
  for (i = 0; i < SampleNum; ++i) {
    Dtype scale;

	u=0; B=0; z=0;
       dtx = DtVector[s] * reshape(colm(XArr,i),d,S);
    for (int iter = 0; iter < iterMax; iter++) {
      q = dtx + rho * (z - u);
      temp = 0;
      for (int s = 0; s < S; s++) {
        CG_cpp(DVector[s], DtVector[s], colm(q,s), iterCG, rho, colm(B,s), tolCG);
        B.col(s) = temp_b;
        temp += n(s);
      }
      // cout << "B:\n" << B << endl;
      u += B;
      EigenVec norm_rows = u.rowwise().norm();
      for (int m = 0; m < d; ++m) {
        scale =
          (norm_rows(m) > lambda) ? (norm_rows(m) - lambda) / norm_rows(m) : 0;
        z.row(m) = scale * u.row(m);
      }
      // u update
      u -= z;
    }
  }
}

void CG_cpp(const EigenMat& D, const EigenMat& DT, const EigenVec& b, const int iter,
  const Dtype rho, EigenVec& x, const Dtype& tolCG) {
  //EigenMat DT = D.transpose();
  EigenVec r = b - DT * (D * x) - rho * x;
  //EigenVec r = b - DTD * x - rho * x;
  EigenVec p = r;
  double error0 = r.dot(r), error1=error0;
  double nipsilon = tolCG * tolCG;
  for (int i = 0; i < iter && error0 > nipsilon; i++,error0=error1)
 {
    EigenVec Ap = DT * (D * p) + rho * p;
    //EigenVec Ap = DTD * p + rho * p;
    Dtype alpha = error0 / (p.dot(Ap));
    x += alpha * p;
    r -= alpha * Ap;
    error1 = r.dot(r);   
    p = error1 / error0 * p + r;
    }
  return;
}



// void CG_cpp(const EigenMat& D, EigenMat& DT, const EigenVec& b, const int iter,
//   const Dtype rho, EigenVec& x0, const Dtype& tolCG) {
//   cout<<"step in"<<endl;

//   EigenMat Dt = D.transpose();

//   EigenVec r0 = b - DT * (D * x0) - rho * x0;

//   EigenVec p0 = r0;

//   EigenVec Ap = DT * (D * p0) + rho * p0;

//   Dtype tmp = r0.dot(r0);
//   Dtype alpha = tmp / (p0.dot(Ap));

//   EigenVec x1 = x0 + alpha * p0;

//   EigenVec r1 = r0 - alpha * Ap;

//   Dtype beta = (r1.dot(r1)) / tmp;

//   EigenVec p1 = r1 + beta * p0;

//   for (int i = 0; i < iter; i++) {
//     x0 = x1;
//     p0 = p1;
//     r0 = r1;
//     Ap = DT * (D * p0) + rho * p0;
//     tmp = r0.dot(r0);
//     alpha = tmp / (p0.dot(Ap));
//     x1 = x0 + alpha * p0;
//     r1 = r0 - alpha * Ap;
//     beta = (r1.dot(r1)) / tmp;
//     p1 = r1 + beta * p0;
//     if (r1.norm() < tolCG) {
//       return;
//     }
//   }
// }






