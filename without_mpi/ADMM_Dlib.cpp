#include "ADMM_Dlib.hpp"
#ifdef USE_OMP
#include <omp.h>
#endif

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

// void ADMM_CG_xwt(DataMat& D, DataMat& XArr, const IntVec& n, const int S,
//   const Dtype Lambda, const Dtype rho, const int iterMax, DataMat& alpha, const Dtype& tolCG, const int iterCG) {
//   // xwt
//   Dtype d = D.nc();
//   int temp = 0;
//   DataMat DtD = zeros_matrix<Dtype>(d * S, d);
//   DataMat DtX = zeros_matrix<Dtype>(d * S, XArr.nc());
//   for (int s = 0; s < S; ++s) {
//     matrix<long> current_range = range(temp, temp + n(s) - 1);
//     // set_rowm(DtD, range(s * d, (s + 1) * d - 1)) = trans(rowm(D, current_range)) * rowm(D, current_range) + rho * identity_matrix<Dtype>(d);
//     set_rowm(DtX, range(s * d, (s + 1) * d - 1)) = trans(rowm(D, current_range)) * rowm(XArr, current_range);
//     temp += n(s);
//   }
//   Dlib2Eigen(D, d_e);
//   Dlib2Eigen(DtD, dtd_e);
//   Dlib2Eigen(DtX, dtx_e);
//   Dlib2Eigen(alpha, alpha_e);
//   EigenMat n_e(S, 1);
//   for (int i = 0; i < S; ++i) {
//     n_e(i) = n(i);
//   }
//   ADMM_ConjugateGradient_xwt(
//     d_e, dtd_e, dtx_e, n_e, n.nr(), Lambda, rho, iterMax, alpha_e, tolCG, iterCG);
// }

// void ADMM_ConjugateGradient_xwt(const EigenMap& D,  const EigenMap& dtd_e, const EigenMap& dtx_e,
//   const EigenMat& n, const int S, const Dtype Lambda, const Dtype rho,
//   const int iterMax, EigenMap& alpha_e, const Dtype& tolCG, const int iterCG) {
//   // return;
//   Dtype lambda = Lambda / rho;
//   const int SampleNum = dtx_e.cols();
//   const int d = D.cols();

//   // std::vector<int> iterloop(S), tolloop(S);
//   // for (int s = 0; s<S; s++) {
//   //   iterloop[s] = iterCG;
//   //   tolloop[s] = tolCG;
//   // }

// #ifdef USE_OMP
// #pragma omp parallel for //firstprivate(iterloop, tolloop)
// #endif
//   for (int i = 0; i < SampleNum; ++i) {
//     //cout<<"SampleNum = "<<i<<endl;
//     Dtype scale;

//     EigenMap z(alpha_e.data() + i * d * S, d, S);
//     EigenMat u = EigenMat::Zero(d, S);
//     EigenMat B = EigenMat::Zero(d, S);
//     EigenMat q = EigenMat::Zero(d, S);
//     EigenVec tmp_dtx = dtx_e.col(i);
//     EigenMap dtx(tmp_dtx.data(), d, S);

//     for (int iter = 0; iter < iterMax; iter++) {
//       q = dtx + rho * (z - u);
//       int temp = 0;

//       for (int s = 0; s < S; s++) {
//         // if (iterloop[s] < 3) {
//         //   tolloop[s] = ( (tolloop[s]*1e-1 - 1e-5) > 0 )? (tolloop[s]*1e-1) : 1e-5;
//         // }

// 	      EigenMat temp_D = D.middleRows(temp, n(s));
//         EigenMat temp_Dt = temp_D.transpose();
//         EigenVec temp_q = q.col(s);
//         EigenVec temp_b = B.col(s);
//         //iterloop[s] = CG_cpp(temp_D, temp_Dt, temp_q, iterCG, rho, temp_b, tolloop);
//         CG_cpp(temp_D, temp_Dt, temp_q, iterCG, rho, temp_b, tolCG);
//         B.col(s) = temp_b;
//         temp += n(s);
//       }
//       // cout << "B:\n" << B << endl;
//       u += B;
//       EigenVec norm_rows = u.rowwise().norm();
//       for (int m = 0; m < d; ++m) {
//         scale =
//           (norm_rows(m) > lambda) ? (norm_rows(m) - lambda) / norm_rows(m) : 0;
//         z.row(m) = scale * u.row(m);
//       }
//       // u update
//       u -= z;
//     }
//     EigenMap alpha_i(z.data(), d * S, 1);
//     alpha_e.col(i) = alpha_i;
//   }
// }

// inline void CG_cpp(const EigenMat& D, const EigenMat& DT, const EigenVec& b, const int iter,
//   const Dtype rho, EigenVec& x, const Dtype& tolCG) {
//   //int iterloop = 0;
//   EigenVec r = b - DT * (D * x) - rho * x;
//   //EigenVec r = b - DTD * x - rho * x;
//   EigenVec p = r;
//   double error0 = r.dot(r), error1=error0;
//   double nipsilon = tolCG * tolCG;
//   for (int i = 0; i < iter && error0 > nipsilon; i++,error0=error1)
//  {
//     //iterloop += 1;
//     EigenVec Ap = DT * (D * p) + rho * p;
//     //EigenVec Ap = DTD * p + rho * p;
//     Dtype alpha = error0 / (p.dot(Ap));
//     x += alpha * p;
//     r -= alpha * Ap;
//     error1 = r.dot(r);   
//     p = error1 / error0 * p + r;
//   }
//   //return iterloop;
// }



/**************************
   CG dlib manual version
**************************/
void ADMM_CG_xwt(DataMat& D, const DataMat& XArr, const IntVec& n, const int S,
  const Dtype Lambda, const Dtype rho, const int iterMax, DataMat& alpha, const Dtype& tolCG, const int iterCG) {
  // xwt
  Dtype d = D.nc();
  int temp = 0;

  std::vector<DataMat> DtVector(S),DVector(S),DtXVector(S);
  for (int s = 0; s < S; ++s) {
    matrix<long> current_range = range(temp, temp + n(s) - 1);

    DVector[s] = rowm(D, current_range);
    DtVector[s] = trans(DVector[s]);
    DtXVector[s] = DtVector[s] * rowm(XArr, current_range);
    
    temp += n(s);
  }

  Dtype lambda = Lambda / rho;
  const int SampleNum = XArr.nc();


  // DataMat dtx( d, S);

  // std::vector<int> iterloop(S);
  // std::vector<Dtype> tolloop(S);
  // for (int s = 0; s<S; s++) {
  //   iterloop[s] = iterCG;
  //   tolloop[s] = tolCG; 
  // }

#ifdef USE_OMP
#pragma omp parallel for private(temp) //firstprivate(iterloop, tolloop) 
#endif
  for (int i = 0; i < SampleNum; ++i) {
    cout<<"SampleNum = "<<i<<endl;
    Dtype scale;
    DataMat u(d, S);
    DataMat B(d, S);
    DataMat z(d, S);
    u=0; B=0; z=0;
    //dtx = DtXVector[s];

    for (int iter = 0; iter < iterMax; iter++) {
      //q = dtx + rho * (z - u);
      temp = 0;      

      for (int s = 0; s < S; s++) {
        // if (iterloop[s] < 3) {
        //   tolloop[s] = ( (tolloop[s]*1e-1 - 1e-5) > 0 )? (tolloop[s]*1e-1) : 1e-5;
        // }

        DataVec q = colm(DtXVector[s], i) + rho * (colm(z, s) - colm(u,s));

        DataVec temp_b = colm(B, s);
        //DataVec temp_q = colm(q, s);
        // iterloop[s] = CG_cpp(DVector[s], DtVector[s], q, iterCG, rho, temp_b, tolloop[s]);
        CG_cpp(DVector[s], DtVector[s], q, iterCG, rho, temp_b, tolCG);
        //B.col(s) = temp_b;
        set_colm(B, s) = temp_b;
        temp += n(s);
      }
      // cout << "B:\n" << B << endl;
      u += B;
      DataVec norm_rows = rownorm_cpp(u);
      for (int m = 0; m < d; ++m) {
        scale = (norm_rows(m) > lambda)
                  ? ((norm_rows(m, 0) - lambda) / norm_rows(m, 0) )
                  : 0;
        set_rowm(z, m) = scale * rowm(u, m);
      }
      // u update
      u -= z;
    }

    set_colm(alpha, i) = reshape_to_column_vector(trans(z));
  }
}

inline void CG_cpp(const DataMat& D, const DataMat& DT, const DataVec& b, const int iter,
  const Dtype rho, DataVec& x, const Dtype& tolCG) {
  //int iterloop = 0;

  DataVec r = b - DT * (D * x) - rho * x;
  DataVec p = r;
  Dtype error0 = trans(r)*r, error1 = error0;
  double nipsilon = tolCG * tolCG;
  
  for (int i = 0; i < iter && error0 > nipsilon; i++,error0=error1)
 {
    //iterloop += 1;
    DataVec Ap = DT * (D * p) + rho * p;
    Dtype alpha = error0 / (trans(p)*Ap);
    p = alpha * p;
    x = x + p;
    r = r - alpha * Ap;
    // x -= alpha *  p;
    // r -= alpha * Ap;
    error1 = trans(r)*r;
    p = error1 / error0 * p + r;
  }
  //return iterloop;
}






