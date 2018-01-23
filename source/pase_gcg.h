#ifndef __PASE_GCG_H__
#define __PASE_GCG_H__

#include "lapacke.h"
#include "lapacke_utils.h"
#include "pase_mg_solver.h"

void GCG_Eigen(PASE_AUX_MATRIX A, PASE_AUX_MATRIX B, PASE_INT Product_type, PASE_REAL *eval, PASE_AUX_VECTOR *evec, PASE_INT nev, PASE_REAL abs_tol, PASE_REAL cg_tol, PASE_INT max_iter, PASE_INT nsmooth, PASE_INT start, PASE_REAL *time_inner, PASE_REAL *time_lapack, PASE_REAL *time_other, PASE_REAL *time_orth);

//用Petsc的矩阵和向量操作构造的函数
void AllocateVecs(PASE_AUX_MATRIX A, PASE_AUX_VECTOR *evec, PASE_AUX_VECTOR **V_1, PASE_AUX_VECTOR **V_2, PASE_AUX_VECTOR **V_3, PASE_INT nev, PASE_INT a, PASE_INT b, PASE_INT c);
void VecsMatrixVecsForRayleighRitz(PASE_AUX_MATRIX A, PASE_AUX_VECTOR *V, PASE_REAL *AA, PASE_INT dim_xp, PASE_INT dim_xpw, PASE_AUX_VECTOR tmp, PASE_REAL *time_inner);
void RayleighRitz(PASE_AUX_MATRIX A, PASE_AUX_MATRIX B, PASE_INT Product_type, PASE_AUX_VECTOR *V, PASE_REAL *AA, PASE_REAL *approx_eval, PASE_REAL *AA_sub, PASE_REAL *AA_copy, PASE_INT start, PASE_INT last_dim, PASE_INT dim, PASE_AUX_VECTOR tmp, PASE_REAL *small_tmp, PASE_REAL *time_inner, PASE_REAL *time_lapack, PASE_REAL *time_other);
void GetRitzVectors(PASE_REAL *SmallEvec, PASE_AUX_VECTOR *V, PASE_AUX_VECTOR *RitzVec, PASE_INT dim, PASE_INT nev);
void ChangeVecPointer(PASE_AUX_VECTOR *V_1, PASE_AUX_VECTOR *V_2, PASE_AUX_VECTOR *tmp, PASE_INT size);
void SumSeveralVecs(PASE_AUX_VECTOR *V, PASE_REAL *x, PASE_AUX_VECTOR U, PASE_INT n_vec);
void GCG_Orthogonal(PASE_AUX_VECTOR *V, PASE_AUX_MATRIX A, PASE_AUX_MATRIX M, PASE_INT Product_type, PASE_INT start, PASE_INT *end, PASE_AUX_VECTOR *V_tmp, PASE_AUX_VECTOR *Nonzero_Vec, PASE_INT *Ind, PASE_REAL *time_inner);
PASE_REAL VecMatrixVec(PASE_AUX_VECTOR a, PASE_AUX_MATRIX Matrix, PASE_AUX_VECTOR b, PASE_AUX_VECTOR temp);

//小规模的向量或稠密矩阵操作，这些应该是串行的，所以没有改动
void OrthogonalSmall(PASE_REAL *V, PASE_REAL **B, PASE_INT dim_xpw, PASE_INT dim_x, PASE_INT *dim_xp, PASE_INT *Ind);
void DenseMatVec(PASE_REAL *DenseMat, PASE_REAL *x, PASE_REAL *b, PASE_INT dim);
void DenseVecsMatrixVecs(PASE_REAL *LVecs, PASE_REAL *DenseMat, PASE_REAL *RVecs, PASE_REAL *ProductMat, PASE_INT nl, PASE_INT nr, PASE_INT dim, PASE_REAL *tmp);
void ScalVecSmall(PASE_REAL alpha, PASE_REAL *a, PASE_INT n);
PASE_REAL NormVecSmall(PASE_REAL *a, PASE_INT n);
PASE_REAL VecDotVecSmall(PASE_REAL *a, PASE_REAL *b, PASE_INT n);
void SmallAXPBY(PASE_REAL alpha, PASE_REAL *a, PASE_REAL beta, PASE_REAL *b, PASE_INT n);


//Petsc CG算法

void GetLAPACKMatrix(PASE_AUX_MATRIX A, PASE_AUX_VECTOR *V, PASE_REAL *AA, PASE_REAL *AA_sub, PASE_INT start, PASE_INT last_dim, PASE_INT dim, PASE_REAL *AA_copy, PASE_AUX_VECTOR tmp, PASE_REAL *small_tmp, PASE_REAL *time_inner, PASE_REAL *time_other);
void GetWinV(PASE_INT nev, PASE_INT nunlock, PASE_INT *unlock, PASE_AUX_VECTOR *V, PASE_REAL *approx_eval, PASE_AUX_MATRIX A, PASE_AUX_MATRIX B, PASE_REAL cg_tol, PASE_INT nsmooth, PASE_AUX_VECTOR rhs);
void CheckConvergence(PASE_AUX_MATRIX A, PASE_AUX_MATRIX B, PASE_INT *unlock, PASE_INT *nunlock, PASE_INT start, PASE_INT nev, PASE_AUX_VECTOR *X_tmp, PASE_REAL *approx_eval, PASE_REAL abs_tol, PASE_AUX_VECTOR *V_tmp, PASE_INT iter, PASE_REAL *RRes, PASE_REAL *time_inner);
void GetPinV(PASE_REAL *AA, PASE_AUX_VECTOR *V, PASE_INT dim_x, PASE_INT last_dim_x, PASE_INT *dim_xp, PASE_INT dim_xpw, PASE_INT nunlock, PASE_INT *unlock, PASE_AUX_VECTOR *V_tmp, PASE_AUX_VECTOR *Orth_tmp, PASE_INT *Ind);
void GetXinV(PASE_AUX_VECTOR *V, PASE_AUX_VECTOR *X_tmp, PASE_AUX_VECTOR *tmp, PASE_INT dim_x);
void Updatedim_x(PASE_INT start, PASE_INT end, PASE_INT *dim_x, PASE_REAL *approx_eval);
void PrintSmallEigen(PASE_INT iter, PASE_INT nev, PASE_REAL *approx_eval, PASE_REAL *AA, PASE_INT dim, PASE_REAL *RRes);

extern lapack_int LAPACKE_dsyev( int matrix_order, char jobz, char uplo, lapack_int n, double* a, lapack_int lda, double* w );
void SortEigen(PASE_REAL *evec, PASE_REAL *eval, PASE_INT dim, PASE_INT dim_x);

#endif
