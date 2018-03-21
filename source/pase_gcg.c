#include "pase_gcg.h"
#include "pase_mg_solver.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"

#if PASE_USE_HYPRE
#include "_hypre_parcsr_mv.h"
#endif


#define EPS 2.220446e-16
#define REORTH_TOL 0.75 

//Product_type=1表示A内积，其它表示B内积
void 
GCG_Eigen(PASE_AUX_MATRIX A, PASE_AUX_MATRIX B, PASE_INT Product_type, PASE_REAL *eval, PASE_AUX_VECTOR *evec, PASE_INT nev, PASE_REAL abs_tol, PASE_REAL cg_tol, PASE_INT max_iter, PASE_INT nsmooth, PASE_INT start, PASE_REAL *time_inner, PASE_REAL *time_lapack, PASE_REAL *time_other, PASE_REAL *time_orth)
{
  //--------------------定义变量--------------------------------------------
  PASE_INT        i, max_dim_x = nev*5/4;//最大是1.25×nev
  //unlock用来记录没有收敛的特征值和特征向量在V中的编号,nunlock为未收敛的特征对个数
  //dim_xpw表示V的长度,dim_xp表示[X,P]的向量个数,dim_x表示X中的向量个数
  PASE_INT        *unlock, nunlock, dim_xpw, last_dim_xpw, dim_xp, dim_x = nev, last_dim_x = nev, iter = 0, *Ind;
  //AA_copy用来存储矩阵AA的备份，为了下次计算做准备
  //AA_sub用来存储小规模矩阵AA中与[X,P]对应的对角部分
  PASE_REAL       *AA, *approx_eval, *AA_copy, *AA_sub, *small_tmp, *RRes;
  PASE_AUX_VECTOR *V, *V_tmp, *X_tmp, *Orth_tmp;
  clock_t start_t, end_t;

  //--------------------分配空间--------------------------------------------
  //给V，Vtmp,x2分配空间,V用于存储[X,d,W],Vtmp是临时存储空间,x2用于存储近似Ritz向量
  AllocateVecs(A, evec, &V, &V_tmp, &X_tmp, nev, 3*max_dim_x, 3*max_dim_x, max_dim_x);
  //给小规模特征值计算的变量分配空间,small_tmp是临时存储空间,计算AA_sub时用
  approx_eval = (PASE_REAL*)calloc(3*max_dim_x, sizeof(PASE_REAL));
  AA          = (PASE_REAL*)calloc(9*max_dim_x*max_dim_x, sizeof(PASE_REAL));
  AA_copy     = (PASE_REAL*)calloc(9*max_dim_x*max_dim_x, sizeof(PASE_REAL));
  AA_sub      = (PASE_REAL*)calloc(4*max_dim_x*max_dim_x, sizeof(PASE_REAL));
  small_tmp   = (PASE_REAL*)calloc(3*max_dim_x, sizeof(PASE_REAL));
  unlock      = calloc(nev, sizeof(PASE_INT));
  Ind         = calloc(3*max_dim_x, sizeof(PASE_INT));
  RRes        = (PASE_REAL*)calloc(nev, sizeof(PASE_REAL));
  Orth_tmp    = (PASE_AUX_VECTOR*)PASE_Malloc(max_dim_x*sizeof(PASE_AUX_VECTOR));
  for(i = 0 ; i < nev; ++i) {
    approx_eval[i] = eval[i];
  }

  //------------------开始CGC计算特征值--------------------------------
  //GetRandomInitValue(V, dim_x);//krylovschur怎么取的随机初值
  //ReadVec("randx_5.txt", V, nev, 1089);
  //对初值做一次B正交化,暂时默认初值不会有线性相关的向量，即nev不会被修改
  //计算得到小规模特征值计算的矩阵AA,BB并保存备份AAt
  //RayleighRitz(A, V, AA, approx_eval, NULL, AA_copy, 0, 0, dim_x, V_tmp[0], NULL);
  //计算初始近似特征向量并正交化
  //GetRitzVectors(AA, V, X_tmp, dim_x, dim_x);
  //ChangeVecPointer(V, X_tmp, Orth_tmp, dim_x);
  GCG_Orthogonal(V, A, B, Product_type, 0, &dim_x, V_tmp, Orth_tmp, Ind, time_orth);
  CheckConvergence(A, B, unlock, &nunlock, start, nev, V, approx_eval, abs_tol, V_tmp, -2, RRes, time_inner);
  //用CG迭代获取W向量
  GetWinV(dim_x, nunlock, unlock, V, approx_eval, A, B, cg_tol, nsmooth, V_tmp[0]);
  //对V进行正交化,并计算evec=V^T*A*V,B1=V^T*B*V
  dim_xpw = 2*dim_x;
  GCG_Orthogonal(V, A, B, Product_type, dim_x, &dim_xpw, V_tmp, Orth_tmp, Ind, time_orth);
  RayleighRitz(A, B, Product_type, V, AA, approx_eval, NULL, AA_copy, 0, 0, dim_xpw, V_tmp[0], NULL, time_inner, time_lapack, time_other);
  //计算Ritz向量,由于得到的特征向量是关于B正交的，不需要对x2进行正交化

  start_t = clock();
  GetRitzVectors(AA, V, X_tmp, dim_xpw, dim_x);
  end_t   = clock();
  *time_other += ((double)(end_t-start_t))/1000000;
  CheckConvergence(A, B, unlock, &nunlock, start, nev, X_tmp, approx_eval, abs_tol, V_tmp, -1, RRes, time_inner);

  //--------------------开始循环--------------------------------------------
  while((nunlock > 0)&&(iter < max_iter)) {
    //更新dim_xp,dim_xp表示[x2,d]的向量个数
    dim_xp = dim_x+nunlock;
    //计算P
    start_t = clock();
    GetPinV(AA, V, dim_x, last_dim_x, &dim_xp, dim_xpw, nunlock, unlock, V_tmp, Orth_tmp, Ind);
    GetXinV(V, X_tmp, Orth_tmp, dim_x);
    end_t   = clock();
    *time_other += ((double)(end_t-start_t))/1000000;
    //更新dim_xpw为V=[x2,d,w]的向量个数
    last_dim_xpw = dim_xpw;
    dim_xpw = dim_xp+nunlock;
    //对unlock的x2进行CG迭代得到w,V的前nev列为x2,dim_xp列之后是w
    GetWinV(dim_xp, nunlock, unlock, V, approx_eval, A, B, cg_tol, nsmooth, V_tmp[0]);
    //对W与前dim_xp个向量进行正交化,Ind记录W中的非零向量的列号
    GCG_Orthogonal(V, A, B, Product_type, dim_xp, &dim_xpw, V_tmp, Orth_tmp, Ind, time_orth);
    //PrintVec(V+dim_xp, dim_xpw-dim_xp);
    //计算小规模矩阵特征值
    RayleighRitz(A, B, Product_type, V, AA, approx_eval, AA_sub, AA_copy, dim_xp, last_dim_xpw, dim_xpw, V_tmp[0], small_tmp, time_inner, time_lapack, time_other);
    //检查特征值重数
    start_t = clock();
    last_dim_x = dim_x;
    Updatedim_x(nev, max_dim_x, &dim_x, approx_eval);
    //计算Ritz向量
    GetRitzVectors(AA, V, X_tmp, dim_xpw, dim_x);
    end_t = clock();
    *time_other += ((double)(end_t-start_t))/1000000;

    CheckConvergence(A, B, unlock, &nunlock, start, nev, X_tmp, approx_eval, abs_tol, V_tmp, iter, RRes, time_inner);

    //PrintSmallEigen(iter, nev, approx_eval, NULL, 0, RRes);
    iter += 1;
  }
  //PASE_Printf(MPI_COMM_WORLD, "iter = %d\n", iter);
  //PrintSmallEigen(iter, nev, approx_eval, AA, 0, RRes);
  //eval,evec是大规模矩阵的近似特征对
  //memcpy(eval, approx_eval, nev*sizeof(PASE_REAL));
  for(i = start; i < nev; ++i) {
    PASE_Aux_vector_copy(X_tmp[i], evec[i]);
    eval[i] = approx_eval[i];
  }
  //------------GCG迭代结束------------------------------------

  //释放空间
  free(approx_eval);  approx_eval = NULL;
  free(unlock);       unlock      = NULL;
  free(AA_sub);       AA_sub      = NULL;
  free(AA_copy);      AA_copy     = NULL;
  free(AA);           AA          = NULL;
  free(small_tmp);    small_tmp   = NULL;
  free(RRes);         RRes        = NULL;
  free(Ind);          Ind         = NULL;
  for(i = 0; i < 3*max_dim_x; ++i) {
    PASE_Aux_vector_destroy(V[i]);
    PASE_Aux_vector_destroy(V_tmp[i]);
  }
  PASE_Free(V);
  PASE_Free(V_tmp);
  for(i = 0; i < max_dim_x; ++i) {
    PASE_Aux_vector_destroy(X_tmp[i]);
  }
  PASE_Free(X_tmp);
  PASE_Free(Orth_tmp);
}


void 
AllocateVecs(PASE_AUX_MATRIX A, PASE_AUX_VECTOR *evec, PASE_AUX_VECTOR **V_1, PASE_AUX_VECTOR **V_2, PASE_AUX_VECTOR **V_3, PASE_INT nev, PASE_INT a, PASE_INT b, PASE_INT c)
{
  (*V_1) = (PASE_AUX_VECTOR*)PASE_Malloc(a*sizeof(PASE_AUX_VECTOR));
  (*V_2) = (PASE_AUX_VECTOR*)PASE_Malloc(b*sizeof(PASE_AUX_VECTOR));
  (*V_3) = (PASE_AUX_VECTOR*)PASE_Malloc(c*sizeof(PASE_AUX_VECTOR));
  PASE_INT i = 0;
  (*V_1)[0] = PASE_Aux_vector_create_by_aux_matrix(A);
  for(i = 1; i < a; ++i) {
    (*V_1)[i] = PASE_Aux_vector_create_by_aux_vector((*V_1)[0]);
  }
  for(i = 0; i < b; ++i) {
    (*V_2)[i] = PASE_Aux_vector_create_by_aux_vector((*V_1)[0]);
  }
  for(i = 0; i < c; ++i) {
    (*V_3)[i] = PASE_Aux_vector_create_by_aux_vector((*V_1)[0]);
  }
  for(i = 0; i < nev; ++i) {
    PASE_Aux_vector_copy(evec[i], (*V_1)[i]);
  }
}

void 
PrintSmallEigen(PASE_INT iter, PASE_INT nev, PASE_REAL *approx_eval, PASE_REAL *AA, PASE_INT dim, PASE_REAL *RRes)
{
  PASE_INT i, j;
  PASE_Printf(MPI_COMM_WORLD, "'in while, the iter: %d LAPACKsyev:'\n", iter);
  for(i = 0; i < nev; ++i) {
    PASE_Printf(MPI_COMM_WORLD, "'approx_eval[%d] = %18.15lf, abosolute residual: %e'\n", i, approx_eval[i], RRes[i]);
  }
  if(AA != NULL) {
    for(i = 0; i < nev; ++i) {
      for(j = 0; j < dim; ++j) {
	PASE_Printf(MPI_COMM_WORLD, "small evec[%d][%d] = %18.15lf\n", i, j, AA[i*dim+j]);
      }
    }
  }
}


//计算lapack计算特征值时的小规模矩阵AA,start表示需要计算的起始列号,dim表示小规模矩阵的维数
void 
GetLAPACKMatrix(PASE_AUX_MATRIX A, PASE_AUX_VECTOR *V, PASE_REAL *AA, PASE_REAL *AA_sub, PASE_INT start, PASE_INT last_dim, PASE_INT dim, PASE_REAL *AA_copy, PASE_AUX_VECTOR tmp, PASE_REAL *small_tmp, PASE_REAL *time_inner, PASE_REAL *time_other)
{
  clock_t start_t, end_t;
  PASE_INT       i;
  //计算AA_sub
  start_t = clock();
  if(start != 0) {
    DenseVecsMatrixVecs(NULL, AA_copy, AA, AA_sub, 0, start, last_dim, small_tmp);
  }
  memset(AA, 0.0, dim*dim*sizeof(PASE_REAL));
  for(i = 0; i<start; ++i) {
    memcpy(AA+i*dim, AA_sub+i*start, start*sizeof(PASE_REAL));
  }
  end_t = clock();
  *time_other += ((double)(end_t-start_t))/1000000;
  VecsMatrixVecsForRayleighRitz(A, V, AA, start, dim, tmp, time_inner);
  start_t = clock();
  memcpy(AA_copy, AA, dim*dim*sizeof(PASE_REAL));
  end_t = clock();
  *time_other += ((double)(end_t-start_t))/1000000;
}

//用CG迭代得到向量W
void 
GetWinV(PASE_INT start, PASE_INT nunlock, PASE_INT *unlock, PASE_AUX_VECTOR *V, PASE_REAL *approx_eval, PASE_AUX_MATRIX A, PASE_AUX_MATRIX B, PASE_REAL cg_tol, PASE_INT nsmooth, PASE_AUX_VECTOR rhs)
{
  PASE_INT       i, j;
  for(i = 0; i < nunlock; ++i) {
    j = unlock[i];
    //初值V[start+i]=V[i]/approx_eval[i]
    //Vtmp[0]=B*V[i]作为右端项
    //计算V[start+i]=A^(-1)BV[i]
    //调用CG迭代来计算线性方程组
    PASE_Aux_matrix_multiply_aux_vector(B, V[j], rhs); 
    //PASE_Aux_vector_copy(rhs, V[start+i]);
    PASE_Aux_vector_copy(V[j], V[start+i]); 
    PASE_Aux_vector_scale(1.0/approx_eval[j], V[start+i]); 
    PASE_Linear_solve_by_cg_aux(A, rhs, V[start+i], cg_tol, nsmooth);
  }
}

//计算残差，并获取未收敛的特征对编号及个数
void 
CheckConvergence(PASE_AUX_MATRIX A, PASE_AUX_MATRIX B, PASE_INT *unlock, PASE_INT *nunlock, PASE_INT start, PASE_INT nev, PASE_AUX_VECTOR *X_tmp, PASE_REAL *approx_eval, PASE_REAL abs_tol, PASE_AUX_VECTOR *V_tmp, PASE_INT iter, PASE_REAL *RRes, PASE_REAL *time_inner)
{
  PASE_INT       i, nunlocktmp = 0;
  PASE_REAL      res_norm, evec_norm, res, max_res = 0.0, min_res = 0.0;
  clock_t start_t, end_t;
  for(i = start; i < nev; ++i) {
    PASE_Aux_matrix_multiply_aux_vector(A, X_tmp[i], V_tmp[0]); 
    PASE_Aux_matrix_multiply_aux_vector(B, X_tmp[i], V_tmp[1]); 
    PASE_Aux_vector_axpy(-approx_eval[i], V_tmp[1], V_tmp[0]); 
    start_t = clock();
    PASE_Aux_vector_norm(V_tmp[0], &res_norm); 
    PASE_Aux_vector_norm(X_tmp[i], &evec_norm); 
    end_t   = clock();
    *time_inner += ((double)(end_t-start_t))/1000000;
    res  = res_norm/evec_norm;
    RRes[i] = res;
    if(i == start) {
      max_res = res;
      min_res = res;
    } else {
      if(res > max_res) {
	max_res = res;
      }
      if(res < min_res) {
	min_res = res;
      }
    }
    if(res > abs_tol) {
      unlock[nunlocktmp]  = i;
      nunlocktmp         += 1;
    }
  }
  *nunlock = nunlocktmp;
  //PASE_Printf(MPI_COMM_WORLD,"nunlock(%d)= %d; max_res(%d)= %e; min_res(%d)= %e\n", iter+1, nunlocktmp, iter+1, max_res, iter+1, min_res);
}

//获取d
void 
GetPinV(PASE_REAL *AA, PASE_AUX_VECTOR *V, PASE_INT dim_x, PASE_INT last_dim_x, PASE_INT *dim_xp, PASE_INT dim_xpw, PASE_INT nunlock, PASE_INT *unlock, PASE_AUX_VECTOR *V_tmp, PASE_AUX_VECTOR *Orth_tmp, PASE_INT *Ind)
{
  PASE_INT i = 0;
  //小规模正交化，构造d,构造AA_sub用于下次计算
  for(i = 0; i < nunlock; ++i) {
    memset(AA+(dim_x+i)*dim_xpw, 0, dim_xpw*sizeof(PASE_REAL));
    memcpy(AA+(dim_x+i)*dim_xpw+last_dim_x, AA+unlock[i]*dim_xpw+last_dim_x, (dim_xpw-last_dim_x)*sizeof(PASE_REAL));
  }
  //小规模evec中，X部分是已经正交的（BB正交），所以对P部分正交化
  OrthogonalSmall(AA, NULL, dim_xpw, dim_x, dim_xp, Ind);
  //计算P所对应的长向量，存在V中
  GetRitzVectors(AA+dim_x*dim_xpw, V, V_tmp, dim_xpw, (*dim_xp)-dim_x);
  ChangeVecPointer(V+dim_x, V_tmp, Orth_tmp, (*dim_xp)-dim_x);
}

void 
GetXinV(PASE_AUX_VECTOR *V, PASE_AUX_VECTOR *X_tmp, PASE_AUX_VECTOR *tmp, PASE_INT dim_x)
{
  ChangeVecPointer(V, X_tmp, tmp, dim_x);
}

//如果对称，那么nl=0,LVecs=NULL;
void 
DenseVecsMatrixVecs(PASE_REAL *LVecs, PASE_REAL *DenseMat, PASE_REAL *RVecs, PASE_REAL *ProductMat, PASE_INT nl, PASE_INT nr, PASE_INT dim, PASE_REAL *tmp)
{
  PASE_INT  i, j;
  for(i = 0; i < nr; ++i) {
    //t=A*u[i]
    DenseMatVec(DenseMat, RVecs+i*dim, tmp, dim);
    if(nl == 0) {
      for(j = 0; j < i+1; ++j) {
	ProductMat[i*nr+j] = VecDotVecSmall(RVecs+j*dim, tmp, dim);
	ProductMat[j*nr+i] = ProductMat[i*nr+j];
      }
    } else {
      for(j = 0; j < nl; ++j) {
	ProductMat[i*nl+j] = VecDotVecSmall(LVecs+j*dim, tmp, dim);
      }
    }
  }
}

void 
RayleighRitz(PASE_AUX_MATRIX A, PASE_AUX_MATRIX B, PASE_INT Product_type, PASE_AUX_VECTOR *V, PASE_REAL *AA, PASE_REAL *approx_eval, PASE_REAL *AA_sub, PASE_REAL *AA_copy, PASE_INT start, PASE_INT last_dim, PASE_INT dim, PASE_AUX_VECTOR tmp, PASE_REAL *small_tmp, PASE_REAL *time_inner, PASE_REAL *time_lapack, PASE_REAL *time_other)
{
  clock_t start_t, end_t;
  PASE_AUX_MATRIX RitzMat = A;
  if(Product_type == 1){
    RitzMat = B;
  }
  GetLAPACKMatrix(RitzMat, V, AA, AA_sub, start, last_dim, dim, AA_copy, tmp, small_tmp, time_inner, time_other);
  start_t = clock();
  LAPACKE_dsyev( 102, 'V', 'U', dim, AA, dim, approx_eval );
  if(Product_type == 1)
  {
    SortEigen(AA, approx_eval, dim, dim);
  }	
  end_t   = clock();
  *time_lapack += ((double)(end_t-start_t))/1000000;
}

void 
SortEigen(PASE_REAL *evec, PASE_REAL *eval, PASE_INT dim, PASE_INT dim_x)
{
    PASE_INT head = 0, tail = dim-1;
    PASE_REAL *work = (PASE_REAL*)calloc(dim, sizeof(PASE_REAL));
	for( head=0; head<dim_x; head++ )
	{
	    tail = dim-1-head;
	    if(head < tail)
	    {
			memcpy(work, evec+head*dim, dim*sizeof(PASE_REAL));
			memcpy(evec+head*dim, evec+tail*dim, dim*sizeof(PASE_REAL));
			memcpy(evec+tail*dim, work, dim*sizeof(PASE_REAL));
			work[0] = eval[head];
			eval[head] = 1.0/eval[tail];
			eval[tail] = 1.0/work[0];
	    }
	    else
	    {
		    break;
	    }
	}
    free(work); work = NULL;
}

void 
Updatedim_x(PASE_INT start, PASE_INT end, PASE_INT *dim_x, PASE_REAL *approx_eval)
{
  PASE_INT tmp, i;
  tmp = start;
  //dsygv求出的特征值已经排序是ascending,从小到大
  //检查特征值的数值确定下次要进行计算的特征值个数
  for(i=start; i<end; ++i) {
    if((fabs(fabs(approx_eval[tmp]/approx_eval[tmp-1])-1))<0.2) {
      tmp += 1;
    } else {
      break;
    }
  }
  *dim_x = tmp;
}

void 
VecsMatrixVecsForRayleighRitz(PASE_AUX_MATRIX A, PASE_AUX_VECTOR *V, PASE_REAL *AA, PASE_INT start, PASE_INT dim, PASE_AUX_VECTOR tmp, PASE_REAL *time_inner)
{
#if 0
  clock_t start_t, end_t;
  PASE_INT i = 0;
  PASE_INT j = 0;
  for(i=start; i<dim; ++i) {
    PASE_Aux_matrix_multiply_aux_vector(A, V[i], tmp); 
    for(j = 0; j < i+1; ++j) {
      start_t = clock();
      PASE_Aux_vector_inner_product(V[j], tmp, AA+i*dim+j); 
      end_t = clock();
      *time_inner += ((double)(end_t-start_t))/1000000;
      AA[j*dim+i] = AA[i*dim+j];
    }
  }
#else
  PASE_INT i = 0;
  PASE_INT j = 0;
  PASE_INT k = 0;
  PASE_INT num_half_inner = ((start+dim+1) * (dim-start)) / 2;
  PASE_INT start_inner = 0;
  MPI_Status status;
  MPI_Request request; 
  //MPI_Request *requests = (MPI_Request*)PASE_Malloc((dim-start)*sizeof(MPI_Request)); 
  PASE_SCALAR *block_tmp1 = (PASE_SCALAR*)PASE_Malloc(V[0]->block_size*sizeof(PASE_SCALAR));
  PASE_SCALAR *block_tmp2 = (PASE_SCALAR*)PASE_Malloc(V[0]->block_size*sizeof(PASE_SCALAR));
  PASE_SCALAR *inner_product_tmp = (PASE_SCALAR*)calloc(num_half_inner, sizeof(PASE_SCALAR));
  for(i = start; i < dim; ++i) {
    start_inner = ((start+i+1) * (i-start)) / 2;
    PASE_Matrix_multiply_vector(A->mat, V[i]->vec, tmp->vec);
    for(j = 0; j < V[i]->block_size; ++j) {
      PASE_Vector_axpy(V[i]->block[j], A->vec[j], tmp->vec);
      block_tmp1[j] = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(A->vec[j]->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(V[i]->vec->vector_data)));
    }
    for(j = 0; j <= i; ++j) {
      inner_product_tmp[start_inner+j]  = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(V[j]->vec->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(tmp->vec->vector_data)));
      for(k = 0; k < V[i]->block_size; ++k) {
	inner_product_tmp[start_inner+j] += V[j]->block[k] * block_tmp1[k];
      }
    }
    //MPI_Iallreduce(MPI_IN_PLACE, &(inner_product_tmp[(i-start)*dim]), i+1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &(requests[i-start]));
  }
  MPI_Iallreduce(MPI_IN_PLACE, inner_product_tmp, num_half_inner, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);

  for(i = start; i < dim; ++i) {
    for(j = 0; j < V[i]->block_size; ++j) {
      block_tmp2[j] = 0.0;
      for(k = 0; k < V[i]->block_size; ++k) {
	block_tmp2[j] += A->block[j][k] * V[i]->block[k];
      }
    }
    for(j = 0; j <= i; ++j) {
      AA[i*dim+j] = 0.0;
      for(k = 0; k < V[i]->block_size; ++k) {
	AA[i*dim+j] += V[j]->block[k] * block_tmp2[k];
      }
    }
  }

  MPI_Wait(&request, &status);
  for(i = start; i < dim; ++i) {
    start_inner = (start+i+1) * (i-start) / 2;
    //MPI_Wait(&(requests[i-start]), &status);
    for(j = 0; j <= i; ++j) {
      AA[i*dim+j] += inner_product_tmp[start_inner+j];
      AA[j*dim+i] = AA[i*dim+j];
    }
  }
  PASE_Free(block_tmp1);
  PASE_Free(block_tmp2);
  PASE_Free(inner_product_tmp);
  //PASE_Free(requests);

#endif
}

//U = V*x
void 
SumSeveralVecs(PASE_AUX_VECTOR *V, PASE_REAL *x, PASE_AUX_VECTOR U, PASE_INT n_vec)
{
  PASE_INT i = 0;
  PASE_Aux_vector_set_constant_value(U, 0.0);
  for(i = 0; i < n_vec; ++i) {
    PASE_Aux_vector_axpy(x[i], V[i], U);
  }
}

//EVEC = V*H_ev
void 
GetRitzVectors(PASE_REAL *SmallEvec, PASE_AUX_VECTOR *V, PASE_AUX_VECTOR *RitzVec, PASE_INT dim, PASE_INT n_vec) 
{
  PASE_INT i = 0;
  for(i = 0; i < n_vec; ++i) {
    SumSeveralVecs(V, SmallEvec+i*dim, RitzVec[i], dim);
  }
}

void 
ChangeVecPointer(PASE_AUX_VECTOR *V_1, PASE_AUX_VECTOR *V_2, PASE_AUX_VECTOR *tmp, PASE_INT size)
{
  memcpy(tmp, V_1, size*sizeof(PASE_AUX_VECTOR));
  memcpy(V_1, V_2, size*sizeof(PASE_AUX_VECTOR));
  memcpy(V_2, tmp, size*sizeof(PASE_AUX_VECTOR));
}

//对V的所有列向量做关于矩阵A的正交化，如果A=NULL，那么实际上做的是L2正交化
//V1:是一个临时的存储空间，是表示零向量的列的指针
//全部正交化，则start=0
void 
GCG_Orthogonal(PASE_AUX_VECTOR *V, PASE_AUX_MATRIX A, PASE_AUX_MATRIX M, PASE_INT Product_type, PASE_INT start, PASE_INT *end, PASE_AUX_VECTOR *V_tmp, PASE_AUX_VECTOR *Nonzero_Vec, PASE_INT *Ind, PASE_REAL *time_orth)
{
  PASE_INT       i, j, n_nonzero = 0, n_zero = 0;
  PASE_REAL      vin, vout, tmp, dd;
  PASE_AUX_MATRIX B = M;
  if(Product_type == 1){
    B = A; 
  }
  clock_t start_t, end_t;
  start_t = clock(); 
  //地址是int型，所以这里只要分配int空间就可以，不需要PASE_REAL**
  if(B == NULL) {
    for(i = start; i < (*end); ++i) {
      if(i == 0) {
        PASE_Aux_vector_norm(V[0], &dd); 
	if(dd > 10*EPS) {
	  PASE_Aux_vector_scale(1.0/dd, V[0]); 
	  Ind[0] = 0;
	  n_nonzero = 1;
	}
      } else {
        PASE_Aux_vector_norm(V[i], &vout); 
	do {
	  vin = vout;
	  for(j = 0; j < start; ++j) {
	    //计算 V[i]= V[i]-(V[i]^T*V[j])*V[j]
	    PASE_Aux_vector_inner_product(V[i], V[j], &tmp); 
	    PASE_Aux_vector_axpy(-tmp, V[j], V[i]); 
	  }
	  for(j = 0; j < n_nonzero; ++j) {
	    //计算 V[i]= V[i]-(V[i]^T*V[Ind[j]])*V[Ind[j]]
	    PASE_Aux_vector_inner_product(V[i], V[Ind[j]], &tmp); 
	    PASE_Aux_vector_axpy(-tmp, V[Ind[j]], V[i]); 
	  }
          PASE_Aux_vector_norm(V[i], &vout); 
	} while(vout/vin < REORTH_TOL);
	if(vout > 10*EPS) {
	  PASE_Aux_vector_scale(1.0/vout, V[i]); 
	  Ind[n_nonzero++] = i;
	} else {
	  //PASE_Printf(MPI_COMM_WORLD, "In GCG_Orthogonal, there is a zero vector! i = %d, start = %d, end: %d\n", i, start, *end);
	  Nonzero_Vec[n_zero++] = V[i];
	}
      }
    }
  } else {

#if 0
    for(i = 0; i < start; ++i) {
      PASE_Aux_matrix_multiply_aux_vector(B, V[i], V_tmp[i]); 
    }
    for(i = start; i < (*end); ++i) {
      if(i == 0) {
	//计算 V[0]^T*A*V[0]
	PASE_Aux_vector_inner_product_general(V[0], V[0], B, &dd);
	dd = sqrt(dd);
	if(dd > 10*EPS) {
	  PASE_Aux_vector_scale(1.0/dd, V[0]); 
	  Ind[n_nonzero++] = 0;
	  PASE_Aux_matrix_multiply_aux_vector(B, V[0], V_tmp[0]); 
	}
      } else {
	PASE_Aux_vector_inner_product_general(V[i], V[i], B, &vout);
	vout = sqrt(vout);
	do {
	  vin = vout;
	  for(j = 0; j < start; ++j) {
	    //计算 V[i]= V[i]-(V[i]^T*V[j])_B*V[j]
	    PASE_Aux_vector_inner_product(V[i], V_tmp[j], &tmp); 
	    PASE_Aux_vector_axpy(-tmp, V[j], V[i]); 
	  }
	  for(j = 0; j < n_nonzero; ++j) {
	    //计算 V[i]= V[i]-(V[i]^T*V[Ind[j]])_B*V[Ind[j]]
	    PASE_Aux_vector_inner_product(V[i], V_tmp[start+j], &tmp); 
	    PASE_Aux_vector_axpy(-tmp, V[Ind[j]], V[i]); 
	  }
	  PASE_Aux_vector_inner_product_general(V[i], V[i], B, &vout);
	  vout = sqrt(vout);
	  //PASE_Printf(MPI_COMM_WORLD, "i = %d, vin = %e, vout = %e\n", i, vin, vout);
	} while(vout/vin < REORTH_TOL);
	if(vout > 10*EPS) {
	  PASE_Aux_vector_scale(1.0/vout, V[i]); 
	  PASE_Aux_matrix_multiply_aux_vector(B, V[i], V_tmp[start+n_nonzero]); 
	  Ind[n_nonzero++] = i;
	} else {
	  PASE_Printf(MPI_COMM_WORLD, "In GCG_Orthogonal, there is a zero vector! i = %d, start = %d, end: %d\n", i, start, *end);
	  Nonzero_Vec[n_zero++] = V[i];
	}
      }
    }
#else
    MPI_Status status;
    MPI_Request request; 
    //MPI_Request *requests = (MPI_Request*)PASE_Malloc((*end)*sizeof(MPI_Request));
    PASE_SCALAR *block_tmp1 = (PASE_SCALAR*)PASE_Malloc(V[0]->block_size*sizeof(PASE_SCALAR));
    PASE_SCALAR *block_tmp2 = (PASE_SCALAR*)PASE_Malloc(V[0]->block_size*sizeof(PASE_SCALAR));
    PASE_SCALAR *inner_product = (PASE_SCALAR*)calloc(*end, sizeof(PASE_SCALAR));
    PASE_SCALAR *inner_product_tmp = (PASE_SCALAR*)calloc(*end, sizeof(PASE_SCALAR));
    PASE_INT k = 0;
    //PASE_SCALAR norm = 0.0;
    PASE_INT iter = 0;

    for(i = start; i < (*end); ++i) {
      if(i == 0) {
	//计算 V[0]^T*A*V[0]
	PASE_Aux_vector_inner_product_general(V[0], V[0], B, &dd);
	dd = sqrt(dd);
	if(dd > 10*EPS) {
	  PASE_Aux_vector_scale(1.0/dd, V[0]); 
	  Ind[n_nonzero++] = 0;
	}
      } else {
	iter = 0;
	do {
	  iter ++;
	  PASE_Matrix_multiply_vector(B->mat, V[i]->vec, V_tmp[0]->vec);
	  for(j = 0; j < V[i]->block_size; ++j) {
	    PASE_Vector_axpy(V[i]->block[j], B->vec[j], V_tmp[0]->vec);
	    block_tmp1[j] = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(B->vec[j]->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(V[i]->vec->vector_data)));
	  }

	  for(j = 0; j < start; ++j) {
	    inner_product[j]  = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(V[j]->vec->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(V_tmp[0]->vec->vector_data)));
	    for(k = 0; k < V[i]->block_size; ++k) {
	      inner_product[j] += V[j]->block[k] * block_tmp1[k];
	    }
            //MPI_Iallreduce(MPI_IN_PLACE, &(inner_product[j]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &(requests[j]));
	  }
	  for(j = 0; j < n_nonzero; ++j) {
	    inner_product[Ind[j]]  = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(V[Ind[j]]->vec->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(V_tmp[0]->vec->vector_data)));
	    for(k = 0; k < V[i]->block_size; ++k) {
	      inner_product[Ind[j]] += V[Ind[j]]->block[k] * block_tmp1[k];
	    }
            //MPI_Iallreduce(MPI_IN_PLACE, &(inner_product[Ind[j]]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &(requests[Ind[j]]));
	  }
	  inner_product[i] = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(V[i]->vec->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(V_tmp[0]->vec->vector_data)));
	  for(k = 0; k < V[i]->block_size; ++k) {
	    inner_product[i] += V[i]->block[k] * block_tmp1[k];
	  }
          //MPI_Iallreduce(MPI_IN_PLACE, &(inner_product[i]), 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &(requests[i]));
          MPI_Iallreduce(MPI_IN_PLACE, inner_product, i+1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);

	  for(j = 0; j < V[i]->block_size; ++j) {
	    block_tmp2[j] = 0.0;
	    for(k = 0; k < V[i]->block_size; ++k) {
	      block_tmp2[j] += B->block[j][k] * V[i]->block[k];
	    }
	  }
	  for(j = 0; j < start; ++j) {
	    inner_product_tmp[j] = 0.0;
	    for(k = 0; k < V[i]->block_size; ++k) {
	      inner_product_tmp[j] += V[j]->block[k] * block_tmp2[k];
	    }
	  }
	  for(j = 0; j < n_nonzero; ++j) {
	    inner_product_tmp[Ind[j]] = 0.0;
	    for(k = 0; k < V[i]->block_size; ++k) {
	      inner_product_tmp[Ind[j]] += V[Ind[j]]->block[k] * block_tmp2[k];
	    }
	  }
	  inner_product_tmp[i] = 0.0;
	  for(k = 0; k < V[i]->block_size; ++k) {
	    inner_product_tmp[i] += V[i]->block[k] * block_tmp2[k];
	  }

          MPI_Wait(&request, &status);
	  for(j = 0; j < start; ++j) {
            //MPI_Wait(&(requests[j]), &status);
	    inner_product[j] += inner_product_tmp[j];
	  }
	  for(j = 0; j < n_nonzero; ++j) {
            //MPI_Wait(&(requests[Ind[j]]), &status);
	    inner_product[Ind[j]] += inner_product_tmp[Ind[j]];
	  }
          //MPI_Wait(&(requests[i]), &status);
	  inner_product[i] += inner_product_tmp[i];

	  vout = inner_product[i];
	  for(j = 0; j < start; ++j) {
	    vout -= inner_product[j]*inner_product[j];
	    //PASE_Printf(MPI_COMM_WORLD, "inner_product[%d] = %e, ", j, inner_product[j]);
	  }
	  for(j = 0; j < n_nonzero; ++j) {
            vout -= inner_product[Ind[j]]*inner_product[Ind[j]];
	    //PASE_Printf(MPI_COMM_WORLD, "inner_product[%d] = %e, ", Ind[j], inner_product[Ind[j]]);
	  }
	  //PASE_Printf(MPI_COMM_WORLD, "\n", Ind[j], inner_product[Ind[j]]);
	  vin = sqrt(inner_product[i]);
	  if(vout < EPS*EPS) {
	    Nonzero_Vec[n_zero++] = V[i];
	    //PASE_Printf(MPI_COMM_WORLD, "Here is a zero vector!!!! i = %d, vin = %e, vout = %e\n", i, vin, vout);
	    break;
	  } else {
	    for(j = 0; j < start; ++j) {
	      PASE_Aux_vector_axpy(-inner_product[j], V[j], V[i]);
	    }
	    for(j = 0; j < n_nonzero; ++j) {
	      PASE_Aux_vector_axpy(-inner_product[Ind[j]], V[Ind[j]], V[i]);
	    }
	    vout = sqrt(vout);
	    PASE_Aux_vector_scale(1.0/vout, V[i]); 
	    //PASE_Printf(MPI_COMM_WORLD, "i = %d, vin = %e, vout = %e, inner_product[0] = %e\n", i, vin, vout, inner_product[0]);
	    //for(j = 0; j <= i; ++j) {
	    //  PASE_Aux_vector_inner_product_general(V[i], V[j], B, &norm);
	    //  PASE_Printf(MPI_COMM_WORLD, "A[%d,%d]=%e, ", i, j, norm);
	    //}
	    //PASE_Printf(MPI_COMM_WORLD, "\n");
	  }
	} while(vout/vin < REORTH_TOL && iter < 50);

	if(vout > EPS*EPS) { //说明并非因为 vout < EPS*EPS 跳出
	  if(vout/vin < REORTH_TOL) { //vout/vin < REORTH_TOL 说明达到最大迭代步数仍旧未满足要求, 为程序稳定性, 将其判定了零向量去除
	    Nonzero_Vec[n_zero++] = V[i];
	    PASE_Printf(MPI_COMM_WORLD, "Here is a zero vector!!!! i = %d, vin = %e, vout = %e\n", i, vin, vout);
	  } else { //当前向量不是零向量, 正交化成功
	    Ind[n_nonzero++] = i;
	    //PASE_Printf(MPI_COMM_WORLD, "Ind[%d] = %d\n", n_nonzero-1, Ind[n_nonzero-1]);
	  }
	}
      }
    }
    //PASE_Free(requests);
    PASE_Free(block_tmp1);
    PASE_Free(block_tmp2);
    PASE_Free(inner_product);
    PASE_Free(inner_product_tmp);
#endif
  }
  //接下来要把V的所有非零列向量存储在地址表格中靠前位置
  *end = start + n_nonzero;
  if(n_zero > 0) {
    for(i = 0; i < n_nonzero; ++i) {
      V[start+i] = V[Ind[i]];
    }
    memcpy(V+(*end), Nonzero_Vec, n_zero*sizeof(PASE_AUX_VECTOR));
  }
  end_t = clock(); 
  *time_orth += ((double)(end_t-start_t))/1000000;
}


//进行部分的正交化, 对V中start位置中后的向量与前start的向量做正交化，同时V的start之后的向量自己也做正交化
//dim_xpw表示V中总的向量个数
//V1:用来存零向量的地址指针
//对小规模的向量组V做正交化, B:表示度量矩阵,dim_xpw表示向量长度，dim_xp:向量个数，V_1：存储零向量的位置
//Vtmp:dim_xp*dim_xpw
void 
OrthogonalSmall(PASE_REAL *V, PASE_REAL **B, PASE_INT dim_xpw, PASE_INT dim_x, PASE_INT *dim_xp, PASE_INT *Ind)
{
  PASE_INT i, j, n_nonzero = 0;
  PASE_REAL vin, vout, tmp;

  if(B == NULL) {
    for(i = dim_x; i < (*dim_xp); ++i) {
      vout = NormVecSmall(V+i*dim_xpw, dim_xpw);
      do {
	vin = vout;
	for(j = 0; j < dim_x; ++j) {
	  tmp = VecDotVecSmall(V+j*dim_xpw, V+i*dim_xpw, dim_xpw);
	  SmallAXPBY(-tmp, V+j*dim_xpw, 1.0, V+i*dim_xpw, dim_xpw);
	}
	for(j = 0; j < n_nonzero; ++j) {
	  tmp = VecDotVecSmall(V+Ind[j]*dim_xpw, V+i*dim_xpw, dim_xpw);
	  SmallAXPBY(-tmp, V+Ind[j]*dim_xpw, 1.0, V+i*dim_xpw, dim_xpw);
	}
	vout = NormVecSmall(V+i*dim_xpw, dim_xpw);
      } while(vout/vin < REORTH_TOL);

      if(vout > 10*EPS) {
	ScalVecSmall(1.0/vout, V+i*dim_xpw, dim_xpw);
	Ind[n_nonzero++] = i;
      } else {
	//PASE_Printf(MPI_COMM_WORLD, "in OrthogonalSmall, there appears a zero vector! i: %d\n", i);
      }
    }
  }

  if(n_nonzero < (*dim_xp-dim_x)) {
    *dim_xp = dim_x + n_nonzero;
    for(i = 0; i < n_nonzero; ++i) {
      memcpy(V+(dim_x+i)*dim_xpw, V+Ind[i]*dim_xpw, dim_xpw*sizeof(PASE_REAL));
    }
  }
}

//右乘:b=Ax,A是方阵，按列优先存储
void 
DenseMatVec(PASE_REAL *DenseMat, PASE_REAL *x, PASE_REAL *b, PASE_INT dim)
{
  PASE_INT i = 0;
  memset(b, 0.0, dim*sizeof(PASE_REAL));
  for(i = 0; i < dim; ++i) {
    SmallAXPBY(x[i], DenseMat+i*dim, 1.0, b, dim);
  }
}

//a=alpha*a, n:表示向量a的长度
void 
ScalVecSmall(PASE_REAL alpha, PASE_REAL *a, PASE_INT n)
{
  PASE_INT i = 0;
  for(i = 0; i < n; ++i) {
    a[i] *= alpha;
  }
}

//对计算向量a的范数，n：表示向量a的长度
PASE_REAL 
NormVecSmall(PASE_REAL *a, PASE_INT n)
{
  PASE_INT  i     = 0;
  PASE_REAL value = 0.0;
  for(i = 0; i < n; ++i) {
    value += a[i] * a[i];
  }
  return sqrt(value);
}

//计算向量a和b的内积，n：表示向量的长度
PASE_REAL 
VecDotVecSmall(PASE_REAL *a, PASE_REAL *b, PASE_INT n)
{
  PASE_INT  i     = 0;
  PASE_REAL value = 0.0;
  for(i = 0; i < n; ++i) {
    value += a[i] * b[i];
  }
  return value;
}

//b = alpha*a+beta*b,n表示向量的长度
void 
SmallAXPBY(PASE_REAL alpha, PASE_REAL *a, PASE_REAL beta, PASE_REAL *b, PASE_INT n)
{
  PASE_INT i = 0;
  for(i = 0; i < n; ++i) {
    b[i] = alpha*a[i] + beta*b[i];
  }
}
