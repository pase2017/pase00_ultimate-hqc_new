/*************************************************************************
	> File Name: Arnoldi_Iteration.c
	> Author: nzhang
	> Mail: zhangning114@lsec.cc.ac.cn
	> Created Time: 2017年09月04日 星期一 19时10分47秒
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "pase_arnoldi.h"
#include "lapacke.h"
#include "lapacke_utils.h"

//TODO: cg 函数暂时放在pase_mg_solver.h 里, 之后会移动，此处需改动
#include "pase_mg_solver.h"
extern lapack_int LAPACKE_dgeev( int matrix_order, char jobvl, char jobvr,
                          lapack_int n, PASE_SCALAR* a, lapack_int lda, PASE_SCALAR* wr,
                          PASE_SCALAR* wi, PASE_SCALAR* vl, lapack_int ldvl, PASE_SCALAR* vr,
                          lapack_int ldvr );

//在此函数外面给V分配空间，分配ncv+1个Vec,以及释放V
void IRA_RestartArnoldi(PASE_EVAL *EVAL, PASE_VEC *EVEC, PASE_MAT A, PASE_MAT M, PASE_VEC *V, int ncv, int nev, int max_loop, PASE_REAL tol)
{
	//printf("nev: %d, ncv: %d, max_loop: %d\n", nev, ncv, max_loop);
	int i;
	int p = nev, loop = 0;
	PASE_REAL norm;
	PASE_SCALAR **H = (PASE_SCALAR**)malloc(ncv*sizeof(PASE_SCALAR*));
	for( i=0; i<ncv-1; i++)
		H[i]  = (PASE_SCALAR*)calloc(ncv,   sizeof(PASE_SCALAR));
	H[ncv-1] = (PASE_SCALAR*)calloc(ncv+1, sizeof(PASE_SCALAR));
	//把EVEC各列加起来作为初值V[0]
	PASE_SCALAR *this_ev = (PASE_SCALAR*)calloc(nev, sizeof(PASE_SCALAR));
	for( i=0; i<nev; i++)
		this_ev[i] = 1.0;
	PASE_VECComb(EVEC, nev, this_ev, V[0]);
	//PASE_GetEVECs(EVEC, nev, &this_ev, 1, &(V[0]));
	free(this_ev);  this_ev = NULL;

	PASE_NormVec(V[0], &norm);
	PASE_ScalVec(1.0/norm, V[0]);
	//p表示要lock住的特征对的个数

	Arnoldi_Iteration(A, M, H, 0, &ncv, nev, EVEC, V, tol);

	//printf("loop = %d, max_loop = %d, ncv = %d, p =%d\n", loop, max_loop, ncv, p);
	while((loop < max_loop)&&(ncv>p))
	{
		if(loop > 0)
			Arnoldi_Iteration(A, M, H, p, &ncv, nev, EVEC, V, tol);
		IRA_FindEigen(A, H, V, &p, nev, &ncv, EVAL, EVEC, loop, max_loop);

		if(H[p-1][p] < tol)
		{
			//需要释放H的一些内容
			for( i=p; i<ncv; i++ )
			{
				free(H[i]);  H[i] = NULL;
			}
			//不变子空间的维数
			ncv = p;
		} else {
			PASE_ScalVec(1.0/H[p-1][p], V[p]);
		}
		loop += 1;
	}
	//printf("the number of iterations: %d\n", loop);

	for( i=0; i<ncv; i++)
	{
		free(H[i]);  H[i] = NULL;
	}
	free(H);  H = NULL;
}

//H是上Hessenberg矩阵，用稠密矩阵PASE_SCALAR**
//用这个函数之前，要把H先做转置,H一开始就分配好(2+m+1)*m-1的空间
//H:ncv*ncv，最后一个分向量长ncv+1
//V:n*ncv+1
void Arnoldi_Iteration(PASE_MAT A, PASE_MAT M, PASE_SCALAR **H, int start, int *ncv, int nev, PASE_VEC *EVEC, PASE_VEC *V, PASE_REAL tol)
{
	int i, j;
	//CG 参数
	PASE_SCALAR cg_tol = 1e-10;
	int nsmooth = 1000;
	int size = Mat_nrows;
	PASE_SCALAR *this_ev;
	PASE_VEC V_temp;
	PASE_REAL inner, norm;

	V_temp = PASE_VecCreate(V[0]);
	for(j = start; j < *ncv; j++)
	{
		PASE_MatrixDotVec(M, V[j], V_temp);
		//只有近似特征值，但是nev之后的近似特征值近似效果很差
		PASE_LinearSolver(A, V_temp, V[j+1], cg_tol, nsmooth);
		for(i=0; i<*ncv; i++ )
			H[j][i] = 0.0;
		Reorth(V, H, j, size);
		if(H[j][j+1] < tol)
		{
			printf("in Arnoldi_Iteration, get a smaller subspace! j: %d\n", j);
			if(j<nev-1)//j可以是nev-1,也是足够的子空间
			{
				printf("in Arnoldi_Iteration, get a new vector!!\n");

				this_ev = (PASE_SCALAR*)calloc(nev-j-1, sizeof(PASE_SCALAR));
				for( i=0; i<nev-j-1; i++)
					this_ev[i] = 1.0;
				PASE_VECComb(EVEC+j+1, nev-j-1, this_ev, V[j+1]);//从j+1加到nev-1
				//PASE_GetEVECs(EVEC+j+1, nev-j-1, &this_ev, 1, &(V[j+1]));
				free(this_ev);  this_ev = NULL;

				PASE_NormVec(V[j+1], &norm);
				PASE_ScalVec(1.0/norm, V[j+1]);
				for( i=0; i<j; i++ ) {
				    PASE_VecDotVec(V[i], V[j+1], &inner);
				    PASE_SumVecSelf(inner, V[i], V[j+1]);
				}
				PASE_NormVec(V[j+1], &norm);
				PASE_ScalVec(1.0/norm, V[j+1]);
				H[j][j+1] = 0.0;
			}
			else
			{
				for( i=j+1; i<*ncv; i++ )
				{
					free(H[i]);  H[i] = NULL;
				}

				*ncv = j+1;
				H[j][j+1] = 0.0;
				break;
			}
		}
		else
			PASE_ScalVec(1.0/H[j][j+1], V[j+1]);
	}
	PASE_FreeVec(V_temp);
}

//num_oldvec和idx_col不能合并，idx_col==-1意味着只重正交化而不更新H
void Reorth(PASE_VEC *V, PASE_SCALAR **H, int j, int n)
{
	PASE_SCALAR vin = 1.0;
	PASE_SCALAR vout = 0.0;
	PASE_SCALAR tmp;
	int i;
	int reorth_time=0;
	do{
		PASE_NormVec(V[j+1], &vin);
		for(i = 0; i < j+1; i++)
		{
			PASE_VecDotVec(V[i], V[j+1], &tmp);
			H[j][i] += tmp;
			PASE_SumVecSelf(-tmp, V[i], V[j+1]);
		}
		PASE_NormVec(V[j+1], &vout);
		reorth_time += 1;
	}while((vout/vin < REORTH_TOL)&&(reorth_time<3));

	H[j][j+1] = vout;
}

//ncv应该是H->N_Rows,应该可以去掉
//不用后nev个，随机再取初值，与前面进行正交化，应该也可以（由PASE收敛速度）
void IRA_FindEigen(PASE_MAT A, PASE_SCALAR **H, PASE_VEC *V, int *p, int nev, int *nncv, 
		PASE_EVAL *Eval, PASE_VEC *Evec, int loop, int max_loop)
{
	int i, j;
	//int info;
	int ncv; 
	ncv = *nncv;

	PASE_SCALAR *wr = (PASE_SCALAR*)calloc(ncv, sizeof(PASE_SCALAR));
	PASE_SCALAR *wi = (PASE_SCALAR*)calloc(ncv, sizeof(PASE_SCALAR));
	//PASE_SCALAR *vl;
	PASE_SCALAR *vr = (PASE_SCALAR*)calloc(ncv*ncv, sizeof(PASE_SCALAR));
	PASE_SCALAR *a  = (PASE_SCALAR*)calloc(ncv*ncv, sizeof(PASE_SCALAR));
	for( i=0; i<ncv; i++ )
		for( j=0; j<ncv; j++ )
			a[i*ncv+j] = H[i][j];
	//102:LAPACK_COL_MAJOR
	//101:LAPACK_ROW_MAJOR
	LAPACKE_dgeev( 102, 'N', 'V', ncv, a, ncv, wr, wi, NULL, 1, vr, ncv );
	free(a);  a = NULL;

	PASE_SCALAR *ev_modulus = (PASE_SCALAR*)calloc(ncv, sizeof(PASE_SCALAR));
	for( i=0; i<ncv; i++ )
		ev_modulus[i] = fabs(wr[i]);

	SortEigen( ev_modulus, wr, wi, vr, 1, ncv, 0, ncv-1 );

	//double *addre;
	for( i=0; i<nev; i++ ) {
	    //addre = vr+i*ncv; 
	    PASE_VECComb(V, ncv, vr+i*ncv, Evec[i]);
	    //PASE_GetEVECs(V, ncv, &addre, 1, &(Evec[i]));
	}
	for( i=0; i<nev; i++ )
		Eval[i] = 1.0/wr[i];

	if( loop < max_loop-1 )
	{
		*p = nev;
		for( i=nev; i<ncv; i++ )
		{
			if( fabs((ev_modulus[i]-ev_modulus[nev-1])/ev_modulus[nev-1]) < MUL_TOL )
				*p = i+1;
			else
				break;
		}

		int num_real=ncv-(*p);
		PASE_SCALAR *Shifts_real = (PASE_SCALAR*)calloc(num_real, sizeof(PASE_SCALAR));
		for( i=0; i<num_real; i++ )
			Shifts_real[i] = wr[i+(*p)];

		PASE_SCALAR **Q = (PASE_SCALAR **)malloc(ncv*sizeof(PASE_SCALAR*));
		for( i=0; i<ncv; i++ )
			Q[i] = (PASE_SCALAR *)calloc(ncv, sizeof(PASE_SCALAR));
		ShiftQR_C(Q, H, ncv, num_real, Shifts_real);
		GetNewV( V, Q, *p, ncv, H);
		PASE_NormVec(V[*p], &(H[*p-1][*p]));

		free(Shifts_real);  Shifts_real = NULL;
	}
	free(ev_modulus);  ev_modulus = NULL;
	free(wr);  wr = NULL;
	free(wi);  wi = NULL;
	free(vr);  vr = NULL;
	*nncv = ncv;

}


void SortEigen(PASE_SCALAR *ev_modulus, PASE_SCALAR *wr, PASE_SCALAR *wi, PASE_SCALAR *vr, 
		int flag, int ncv, int left, int right)
{
    if ( left < right )
    {
	int i   = left;
	int j   = right;
	PASE_SCALAR key = *(ev_modulus+left);
	PASE_SCALAR tmp1, tmp2, tmp3;
	PASE_SCALAR *tmp_ev = (PASE_SCALAR*)calloc(ncv, sizeof(PASE_SCALAR));
	while ( i<j )
	{
	    while(i<j && *(ev_modulus+j)<key) j--;
	    if(i<j)//swap_indexvec(a+i++, a+j);
	    {
	        tmp1   = *(ev_modulus+i);
	        *(ev_modulus+i) = *(ev_modulus+j);
	        *(ev_modulus+j) = tmp1;
	        
	        tmp2   = *(wr+i);
	        *(wr+i) = *(wr+j);
	        *(wr+j) = tmp2;
	        
	        tmp3   = *(wi+i);
	        *(wi+i) = *(wi+j);
	        *(wi+j) = tmp3;
	        
			memcpy(tmp_ev, vr+i*ncv, ncv*sizeof(PASE_SCALAR));
			memcpy(vr+i*ncv, vr+j*ncv, ncv*sizeof(PASE_SCALAR));
			memcpy(vr+j*ncv, tmp_ev, ncv*sizeof(PASE_SCALAR));

	        i++;
	    }
	    while(i<j && *(ev_modulus+i)>key) i++;
	    if(i<j)//swap_indexvec(a+j--, a+i);
	    {
	        tmp1   = *(ev_modulus+i);
	        *(ev_modulus+i) = *(ev_modulus+j);
	        *(ev_modulus+j) = tmp1;
	        
	        tmp2   = *(wr+i);
	        *(wr+i) = *(wr+j);
	        *(wr+j) = tmp2;
	        
	        tmp3   = *(wi+i);
	        *(wi+i) = *(wi+j);
	        *(wi+j) = tmp3;
	        
			memcpy(tmp_ev, vr+i*ncv, ncv*sizeof(PASE_SCALAR));
			memcpy(vr+i*ncv, vr+j*ncv, ncv*sizeof(PASE_SCALAR));
			memcpy(vr+j*ncv, tmp_ev, ncv*sizeof(PASE_SCALAR));

	        j--;
	    }
	}
	free(tmp_ev);  tmp_ev = NULL;
	*(ev_modulus+i) = key;//swap_indexvec(a+i, &key);
	SortEigen( ev_modulus, wr, wi, vr, flag, ncv, left, i-1 );
	SortEigen( ev_modulus, wr, wi, vr, flag, ncv, i+1, right );
    }
}


void GetNewV( PASE_VEC *V, PASE_SCALAR **Q, int p, int ncv, PASE_SCALAR **H )
{
	int i;
	PASE_VEC *V_new = (PASE_VEC*)malloc(ncv*sizeof(PASE_VEC));
	for( i=0; i<ncv; i++ )
		V_new[i] = PASE_VecCreate(V[0]);
	/*V_new = V*Q */
	PASE_GetEVECs(V, ncv, Q, p+1, V_new);

	PASE_ScalVec(H[p-1][p], V_new[p]);
	PASE_SumVecSelf(H[ncv-1][ncv]*Q[p-1][ncv-1], V[ncv], V_new[p]); 

	for( i=0; i<ncv; i++ )
	{
		free(Q[i]);  Q[i] = NULL;
	}
	free(Q);  Q = NULL;
	for( i=0; i<ncv; i++ )
		PASE_FreeVec(V[i]);
	for( i=0; i<ncv; i++ )
		V[i] = V_new[i];
	free(V_new);  V_new = NULL;

}




//对矩阵R做nshift次带位移的QR分解，位移为shifts
//运行这个函数之前，给矩阵Q分配空间
void ShiftQR_C(PASE_SCALAR **Q, PASE_SCALAR **R, int nrows, int num_real, PASE_SCALAR *shifts_real)
{
	int is, i;
	PASE_SCALAR *this_cs, shift_real;
	//矩阵Q也是按列存储,首先令Q=I
	for( i=0; i<nrows; i++ )
		memset( Q[i], 0.0, nrows*sizeof(PASE_SCALAR) );
	for( i=0; i<nrows; i++ )
		Q[i][i] = 1.0;

	PASE_SCALAR **cs = malloc( num_real*sizeof(PASE_SCALAR*) );
	for( is=0; is<num_real; is++ )
		cs[is] = calloc( 2*nrows-2, sizeof(PASE_SCALAR) );
	for( is=0; is<num_real; is++ )
	{
		shift_real = shifts_real[is];
		//H=H-\muI
		for( i=0; i<nrows; i++ )
			R[i][i] = R[i][i]-shift_real;
    
		//printf("\n-shift: %lf, H:\n", shift_real);
		//Print_R( R, nrows, nrows);
		//H进行QR分解，然后计算H=RQ
		QR_Givens_C( R, nrows, cs[is] );
		//H=H+\muI
		for( i=0; i<nrows; i++ )
			R[i][i] = R[i][i]+shift_real;
	}
	for( is=0; is<num_real; is++ )
	{
		this_cs = cs[is];
		for( i=0; i<nrows-1; i++ )
			RightGivens( Q, i, nrows, this_cs[2*i], this_cs[2*i+1] );
	}
	for( is=0; is<num_real; is++ )
	{
		free(cs[is]);  cs[is] = NULL;
	}
	free(cs);  cs = NULL;
	//printf("\nQ:\n");
	//Print_R( Q, nrows, nrows);

}

//
//用Givens变换实现H=Q*R分解，再计算H=R*Q
void QR_Givens_C( PASE_SCALAR **R, int nrows, PASE_SCALAR *cs )
{
	int i;
	//获取R,Q
	for( i=0; i<nrows-1; i++ )
	{
		GetGivens( R[i]+i, cs+2*i );
		LeftGivens( R, i, nrows, cs[2*i], cs[2*i+1] );
	}
	//printf("R:\n");
	//Print_R( R, nrows, nrows);
	//计算H=RQ
	for( i=0; i<nrows-1; i++ )
		RightGivens( R, i, nrows, cs[2*i], cs[2*i+1] );
	//printf("H:\n");
	//Print_R( R, nrows, nrows);
	
}

//这里只针对QR分解的Givens变换，即k=i+1
void GetGivens( PASE_SCALAR *x, PASE_SCALAR *cs )
{
	PASE_SCALAR tol = 1e-10;
	PASE_SCALAR t   = 0.0;
	//printf("x[1] = %lf, abs_fem(x[1]) = %lf\n", x[1], abs_fem(x[1]));
	if(fabs(x[1]) < tol)
	{
		cs[0] = 1.0;
		cs[1] = 0.0;
	}
	else
	{
		if(fabs(x[0]) < fabs(x[1]))
		{
			t     = x[0]/x[1];
			cs[1] = 1.0/sqrt(1+t*t);
			cs[0] = cs[1]*t;
		}
		else
		{
			t     = x[1]/x[0];
			cs[0] = 1.0/sqrt(1+t*t);
			cs[1] = cs[0]*t;
		}
	}
}

//左乘Givens矩阵，只针对QR分解
void LeftGivens( PASE_SCALAR **R, int i, int n, PASE_SCALAR c, PASE_SCALAR s )
{
	PASE_SCALAR temp;
	int j;
	//前i-1列第i,i+1分量都为0
	R[i][i]   = c*R[i][i] + s*R[i][i+1];
	R[i][i+1] = 0.0;
	//更新第i,i+1行，按列存储，所以更新第i+1到n-1列的R[][i],R[][i+1]
	for( j=i+1; j<n; j++ )
	{
		temp      =    R[j][i];
		R[j][i]   =  c*temp + s*R[j][i+1];
		R[j][i+1] = -s*temp + c*R[j][i+1];
	}
}

void RightGivens( PASE_SCALAR **R, int i, int n, PASE_SCALAR c, PASE_SCALAR s )
{
	PASE_SCALAR temp;
	int j;
	//更新第i,i+1列
	for( j=0; j<n; j++ )
	{
		temp      = R[i][j];
		R[i][j]   =  c*temp + s*R[i+1][j];
		R[i+1][j] = -s*temp + c*R[i+1][j];
	}
}
