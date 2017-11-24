/*************************************************************************
	> File Name: ../include/PASEonoldi_C.h
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: Fri Nov 17 22:10:13 2017
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pase_param.h"
#include "pase_aux_matrix.h"
#include "pase_aux_vector.h" 

#define REORTH_TOL 0.75//是否进行重正交化的参数
//#define H_TOL 1e-25//Arnoldi迭代判断H(j-1,j)是否足够小的参数
#define MUL_TOL 1e-5//判断是否是模相等的特征值 
#define PASE_LinearSolver PASE_Linear_solve_by_cg_aux
#define PASE_VecCreate PASE_Aux_vector_create_by_aux_vector 
#define PASE_FreeVec PASE_Aux_vector_destroy 

#define PASE_EVAL PASE_SCALAR
#define PASE_VEC PASE_AUX_VECTOR 
#define PASE_MAT PASE_AUX_MATRIX//用户需要定义的矩阵类型
#define PASE_SumVecSelf PASE_Aux_vector_add//用户定义的实向量相加
#define PASE_ScalVec PASE_Aux_vector_scale//用户定义的实向量乘以系数
#define PASE_NormVec PASE_Aux_vector_norm//用户定义的实向量乘以
#define PASE_VecDotVec PASE_Aux_vector_inner_product
#define PASE_MatrixDotVec PASE_Aux_matrix_multiply_aux_vector

#define PASE_VECComb  PASE_Multi_aux_vector_combination
#define PASE_GetEVECs PASE_Multi_aux_vector_by_matrix 

PASE_INT Mat_nrows;//需要在main函数文件中声明extern PASE_INT Mat_nrows;

//用户要调用的函数是PASE_RestartArnoldi
void IRA_RestartArnoldi(PASE_EVAL *EVAL, PASE_VEC *EVEC, PASE_MAT A, PASE_MAT M, PASE_VEC *V, PASE_INT ncv, PASE_INT nev, PASE_INT max_loop, PASE_REAL tol);

void Arnoldi_Iteration(PASE_MAT A, PASE_MAT M, PASE_SCALAR **H, PASE_INT start, PASE_INT *ncv, PASE_INT nev, PASE_VEC *EVEC, PASE_VEC *V, PASE_REAL tol);
void Reorth(PASE_VEC *V, PASE_SCALAR **H, PASE_INT j, PASE_INT n);
void IRA_FindEigen(PASE_MAT A, PASE_SCALAR **H, PASE_VEC *V, PASE_INT *p, PASE_INT nev, PASE_INT *nncv, 
		PASE_EVAL *Eval, PASE_VEC *Evec, PASE_INT loop, PASE_INT max_loop);
void SortEigen(PASE_SCALAR *ev_modulus, PASE_SCALAR *wr, PASE_SCALAR *wi, PASE_SCALAR *vr, 
		PASE_INT flag, PASE_INT ncv, PASE_INT left, PASE_INT right);
PASE_SCALAR **IRShiftQR_C(PASE_SCALAR **H, PASE_INT ncv, PASE_INT p,
		PASE_INT num_real, PASE_SCALAR *Shifts_real,
		PASE_INT num_imag, PASE_SCALAR *Shifts_imag);
void GetNewV( PASE_VEC *V, PASE_SCALAR **Q, PASE_INT p, PASE_INT ncv, PASE_SCALAR **H );

void ShiftQR_C(PASE_SCALAR **Q, PASE_SCALAR **R, PASE_INT nrows, PASE_INT num_real, PASE_SCALAR *shifts_real);
void QR_Givens_C( PASE_SCALAR **R, PASE_INT nrows, PASE_SCALAR *cs );
void GetGivens( PASE_SCALAR *x, PASE_SCALAR *cs );
void LeftGivens( PASE_SCALAR **R, PASE_INT i, PASE_INT n, PASE_SCALAR c, PASE_SCALAR s );
void RightGivens( PASE_SCALAR **R, PASE_INT i, PASE_INT n, PASE_SCALAR c, PASE_SCALAR s );
