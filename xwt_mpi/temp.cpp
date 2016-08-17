#include <mpi.h>
// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt

/* 
	compile:  g++ matrix_ex.cpp ../dlib-18.16/dlib/all/source.cpp  -I../dlib-18.16 -I../eigen-3.2.8 -lpthread  -lX11
    


	This is an example illustrating the use of the matrix object 
    from the dlib C++ Library.
*/

#include "../eigen-3.2.8/Eigen/Dense"
//#define DLIB_USE_BLAS
//#define DLIB_USE_LAPACK

#include <iostream>
#include <math.h>
#include "../dlib_18.16/dlib/matrix.h"

using namespace std;
using namespace Eigen;
using namespace dlib; 
typedef Map<MatrixXd> MexMat;


int ADMM(MexMat &Dictionarys, MexMat X,MexMat &U,MexMat &L,double *alpha, double lambda,double rho, int iterMax,size_t S,size_t sum_n,size_t d,size_t SampleNum,double *n )
{


cout <<Dictionarys<<"this is second admm"<<endl;
cout << X<<endl;

   lambda /= rho;
    int i;
  #pragma omp parallel for private(i) 
  for (i = 0; i < SampleNum; i++)
  {
  double scale;
  int iter, temp, s, m;
    
  MexMat z ( alpha + d*S *i, d, S);
  MatrixXd u(d, S);
  MatrixXd B(d, S);
  // initialize Z and u to zero
  z.setZero();
  u.setZero(); 
  // precompute X'Y
  MatrixXd DtX(d, S);
  MatrixXd q(d, S);
  MatrixXd norm_rows(d,1);
  temp = 0;
  //cout << n << endl;
  for (s=0; s < S; s++){
    DtX.col(s).noalias() = Dictionarys.block(temp, 0, (int)n[s], d).transpose() * X.block(temp, i, (int)n[s], 1);
    temp += (int)n[s];
  }
  //cout << xy.col(0) << endl;
  //cout << U.block(d, 0, d, d) << endl;
  for (iter = 0; iter < iterMax; iter++){
    q = DtX + rho * (z-u);
    for (s=0; s < S; s++){    
      B.col(s) = L.block(d*s, 0, d, d).triangularView<Eigen::Lower>().solve(q.col(s)); //taking advantage of the fact that L is lower-triangular:"
      U.block(d*s, 0, d, d).triangularView<Eigen::Upper>().solveInPlace(B.col(s)); //taking advantage of the fact that U is Upper-triangular and doing in place
    }
    // z update, solve associated proximal problem
    u += B;
    norm_rows = u.rowwise().norm();
    //cout << norm_rows;
    for (m=0; m < d; ++m) {
      if (norm_rows.coeff(m,0) > lambda) {
        scale = (norm_rows.coeff(m,0) - lambda)/norm_rows.coeff(m,0);
        z.row(m) = scale*u.row(m);
      }
    } 
    // u update
    u -= z;
  }
cout<<"this is alpha"<<endl<<z<<endl;
  }

return 0;
}

#define Dlib2Eigen(x,z)  MexMat (z)(x.begin(), x.nr(),x.nc())



int ADMM(matrix<double>  dlibD, matrix<double>  dlibX,matrix<double>  dlibU,matrix<double>  dlibL,double *alpha, double lambda,double rho, int iterMax,size_t S,size_t sum_n,size_t d,size_t SampleNum,double *n )
{

//	MexMat z ( alpha + d*S *i, d, S);
	Dlib2Eigen(dlibD,Dictionarys) ;
	Dlib2Eigen(dlibX,X) ;
	Dlib2Eigen(dlibU,U) ;
	Dlib2Eigen(dlibL,L) ;

cout <<Dictionarys<<"this is first admm"<<endl;
cout << X<<endl;

    ADMM( Dictionarys, X,U,L,alpha, lambda, rho, iterMax, S,sum_n,d,SampleNum,n );
return 0;

}
// ----------------------------------------------------------------------------------------

int main()
{
   
    matrix<double> DD = randm(4,4);

	matrix<double> D = trans(DD)*DD;
    // MATLAB: A = chol(E,'lower') 
    matrix<double> L = chol(D);
    // MATLAB: var = min(min(A))
    matrix<double> U = trans(L);
    matrix<double,4,1> X,a;

	X = colm(D,2);
	double * alpha = a.begin();

double n=4;
    ADMM( D, X,U,L,alpha, 0.01, .1, 100, 1,4,4,1,  &n );

cout << D <<endl;
cout<< X<<endl;
cout<<a<<endl;

}

// ----------------------------------------------------------------------------------------


