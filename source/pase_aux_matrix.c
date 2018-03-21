#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "time.h"
#include "pase_matrix.h"
#include "pase_vector.h"
#include "pase_aux_matrix.h"

#if PASE_USE_HYPRE
#include "_hypre_parcsr_mv.h"
#endif

#define DEBUG_PASE_AUX_MATRIX 1

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_create"
/**
 * @brief 创建辅助空间 V_H + span{ u_h[0], ..., u_h[block_size-1]} 
 *        对应的矩阵 [mat   vec  ],
 *                   [vec^T block]
 *
 *        其中, mat   = A_H, 
 *              vec   = R_hH * A_h * u_h, 
 *              block = u_h^T * A_h * u_h.
 *
 *
 * @param A_H         输入参数, 粗空间 V_H 对应的矩阵
 * @param R_hH        输入参数, 从细空间 V_h 到粗空间 V_H 的限制矩阵
 * @param A_h         输入参数, 细空间 V_h 对应的矩阵
 * @param u_h         输入参数, 细空间的向量组 
 * @param block_size  输入参数, 向量组 u_h 的维数
 *
 * @return PASE_AUX_MATRIX
 */
PASE_AUX_MATRIX 
PASE_Aux_matrix_create(PASE_MATRIX A_H, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h, PASE_INT block_size)
{
#if DEBUG_PASE_AUX_MATRIX
  if(NULL == A_H) {
    PASE_Error(__FUNCT__": Cannot create PASE AUX MATRIX without coarse grid matrix A_H.\n");
  }
  if(NULL == R_hH) {
    PASE_Error(__FUNCT__": Cannot create PASE AUX MATRIX without restrict matrix R_hH.\n");
  }
  if(NULL == A_h) {
    PASE_Error(__FUNCT__": Cannot create PASE AUX MATRIX without fine grid matrix A_h.\n");
  }
  if(NULL == u_h) {
    PASE_Error(__FUNCT__": Cannot create PASE AUX MATRIX without fine grid vectors u_h.\n");
  }
  if(0 >= block_size) {
    PASE_Error(__FUNCT__": Cannot create PASE AUX MATRIX with a nonpostive block size %d.\n", block_size);
  }
#endif

  PASE_AUX_MATRIX aux_A = (PASE_AUX_MATRIX)PASE_Malloc(sizeof(PASE_AUX_MATRIX_PRIVATE));
  aux_A->mat            = A_H;
  aux_A->is_mat_owner   = PASE_NO;
  aux_A->block_size     = block_size;
  aux_A->vec            = NULL;
  aux_A->block          = NULL;
  PASE_Aux_matrix_set_aux_space(aux_A, R_hH, A_h, u_h);
  return aux_A;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_set_aux_space_some"
/**
 * @brief 设置 aux_A = [mat   vec  ] 从第 i 个到第 j 个的辅助空间, 
 *                     [vec^T block]
 *        即 vec[i],...,vec[j], 以及 block[i][:],...,block[j][:],
 *        使其成为 V_H + span{ u_h[0], ..., u_h[aux_A->block_size-1]} 对应的矩阵.
 *
 *        其中, vec[k]      = R_hH * A_h * u_h[k],     i <= k <= j,
 *              block[k][l] = u_h[k]^T * A_h * u_h[l], i <= k <= j, 0 <= l <= (aux_A->block_size-1).
 *
 *        通常用于 1. 初次创建 aux_A (i=0, j=aux_A->block_size-1), 
 *                 2. 更新 aux_A 的部分辅助空间 (比如, i>0).
 *
 *
 * @param aux_A  输入/输出参数
 * @param i      输入参数, 起始位置, i >= 0
 * @param j      输入参数, 终止位置, j <= (aux_A->block_size-1)
 * @param R_hH   输入参数, 细空间 V_h 到粗空间 V_H 的限制矩阵
 * @param A_h    输入参数, 细空间 V_h 对应的矩阵
 * @param u_h    输入参数, 细空间的向量组, 维数需为 aux_A->block_size 
 */
void 
PASE_Aux_matrix_set_aux_space_some(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h)
{
#if DEBUG_PASE_AUX_MATRIX
  if(NULL == aux_A) {
    PASE_Error(__FUNCT__": Cannot set some aux space without PASE AUX MATRIX.\n");
  }
  if(0 > i || j >= aux_A->block_size || 0 >= aux_A->block_size) {
    PASE_Error(__FUNCT__": Cannot set aux space from %dth to %dth of PASE AUX MATRIX with block size being %d.\n", i, j, aux_A->block_size);
  }
  if(NULL == R_hH) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without restrict matrix R_hH.\n");
  }
  if(NULL == A_h) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without fine grid matrix A_h.\n");
  }
  if(NULL == u_h) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without fine grid vectors u_h.\n");
  }
#endif 
  PASE_Aux_matrix_set_block_some(aux_A, i, j, A_h, u_h);
  PASE_Aux_matrix_set_vec_some(aux_A, i, j, R_hH, A_h, u_h);
}

PASE_INT
PASE_Aux_matrix_set_vec_some(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h)
{
  PASE_INT k;
  if(NULL == aux_A->vec) {
    aux_A->vec = (PASE_VECTOR*)PASE_Malloc(aux_A->block_size*sizeof(PASE_VECTOR));
    for(k = 0; k < aux_A->block_size; ++k) {
      aux_A->vec[k] = PASE_Vector_create_by_matrix_and_vector_data_operator(aux_A->mat, u_h[0]->ops);
    }
  }

  PASE_VECTOR workspace_h = PASE_Vector_create_by_vector(u_h[0]);
  for(k = i; k <= j; ++k) {
    PASE_Matrix_multiply_vector(A_h, u_h[k], workspace_h);
    PASE_Matrix_multiply_vector(R_hH, workspace_h, aux_A->vec[k]);
  }
  PASE_Vector_destroy(workspace_h);
  return 0;
}

PASE_INT
PASE_Aux_matrix_set_vec(PASE_AUX_MATRIX aux_A, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h)
{
  PASE_Aux_matrix_set_vec_some(aux_A, 0, aux_A->block_size-1, R_hH, A_h, u_h);
  return 0;
}

PASE_INT
PASE_Aux_matrix_set_block_some(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX A_h, PASE_VECTOR *u_h)
{
  PASE_INT k, l;
  PASE_INT block_size = aux_A->block_size;
  if(NULL == aux_A->block) {
    aux_A->block = (PASE_SCALAR**)PASE_Malloc(block_size*sizeof(PASE_SCALAR*));
    for(k = 0; k < block_size; ++k) {
      aux_A->block[k] = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
    }
  }
  PASE_VECTOR workspace_h = PASE_Vector_create_by_vector(u_h[0]);

#if PASE_USE_HYPRE
  MPI_Status status;
  MPI_Request *requests = (MPI_Request*)PASE_Malloc((j-i+1)*sizeof(MPI_Request));
  //MPI_Request requests;
  PASE_SCALAR *data = (PASE_SCALAR*)PASE_Malloc(block_size*(j-i+1)*sizeof(PASE_SCALAR));
  for(k = i; k <= j; ++k) {
    PASE_Matrix_multiply_vector(A_h, u_h[k], workspace_h);
    for(l = 0; l < block_size; ++l) {
      if(l < i) {
        data[(k-i)*block_size+l] = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(workspace_h->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(u_h[l]->vector_data)));
      } else if(l >= k) {
        data[(k-i)*block_size+l-k+i] = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(workspace_h->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(u_h[l]->vector_data)));
      }
    }
    MPI_Iallreduce(MPI_IN_PLACE, &(data[(k-i)*block_size]), block_size-k+i, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &(requests[k-i]));
  }
  for(k = i; k <= j; ++k) {
    MPI_Wait(&(requests[k-i]), &status);
    for(l = 0; l < block_size; ++l) {
      if(l < i) {
        aux_A->block[k][l] = data[(k-i)*block_size+l];
        aux_A->block[l][k] = data[(k-i)*block_size+l];
      } if(i <= l && l < k) {
        MPI_Wait(&(requests[l-i]), &status);
        aux_A->block[k][l] = data[(l-i)*block_size+k-l+i];
      } else if(k <= l && l <= j) {
        aux_A->block[k][l] = data[(k-i)*block_size+l-k+i];
      } else if(l > j) {
        aux_A->block[k][l] = data[(k-i)*block_size+l-k+i];
        aux_A->block[l][k] = data[(k-i)*block_size+l-k+i];
      }
    }
  }
  PASE_Free(requests);
  PASE_Free(data);
#else
  for(k = i; k <= j; ++k) {
    PASE_Matrix_multiply_vector(A_h, u_h[k], workspace_h);
    for(l = 0; l < aux_A->block_size; ++l) {
      if(l >= i && l <= j) {
        PASE_Vector_inner_product(workspace_h, u_h[l], &(aux_A->block[k][l]));
      } else {
        PASE_Vector_inner_product(workspace_h, u_h[l], &(aux_A->block[k][l]));
	aux_A->block[l][k] = aux_A->block[k][l];
      }
    }
  }
#endif
  PASE_Vector_destroy(workspace_h);
  return 0;
}

PASE_INT
PASE_Aux_matrix_set_block(PASE_AUX_MATRIX aux_A, PASE_MATRIX A_h, PASE_VECTOR *u_h)
{
  PASE_Aux_matrix_set_block_some(aux_A, 0, aux_A->block_size-1, A_h, u_h);
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_set_aux_space"
/**
 * @brief 设置 aux_A = [mat   vec  ] 的全部辅助空间, 即 vec 和 block,
 *                     [vec^T block]
 *        使其成为 V_H + span{ u_h[0], ..., u_h[aux_A->block_size-1]} 对应的矩阵.
 *
 *        其中, vec   = R_hH * A_h * u_h, 
 *              block = u_h^T * A_h * u_h.
 *
 *
 * @param aux_A  输入/输出参数
 * @param R_hH   输入参数, 细空间 V_h 到粗空间 V_H 的限制矩阵
 * @param A_h    输入参数, 细空间 V_h 对应的矩阵
 * @param u_h    输入参数, 细空间的向量组, 其维数为 aux_A->block_size
 */
void 
PASE_Aux_matrix_set_aux_space(PASE_AUX_MATRIX aux_A, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h)
{
#if DEBUG_PASE_AUX_MATRIX
  if(NULL == aux_A) {
    PASE_Error(__FUNCT__": Cannot set some aux space without PASE AUX MATRIX.\n");
  }
  if(0 >= aux_A->block_size) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX with block size being %d.\n", aux_A->block_size);
  }
  if(NULL == R_hH) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without restrict matrix R_hH.\n");
  }
  if(NULL == A_h) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without fine grid matrix A_h.\n");
  }
  if(NULL == u_h) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without fine grid vectors u_h.\n");
  }
#endif 

  PASE_Aux_matrix_set_aux_space_some(aux_A, 0, aux_A->block_size-1, R_hH, A_h, u_h);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_create_by_aux_matrix"
/**
 * @brief 创建辅助空间 V_H + span{ aux_u_h[0], ..., aux_u_h[block_size-1]}
 *        对应的矩阵 [mat   vec  ], 
 *                   [vec^T block]
 *        其中, mat   = A_H,
 *              vec   = R_hH * (aux_A_h*aux_u_h)->vec,
 *              block = aux_u_h^T * aux_A_h * aux_u_h.
 *
 * @param A_H         输入参数, 粗空间 V_H 对应的矩阵
 * @param R_hH        输入参数, 细空间 V_h 到粗空间 V_H 的限制矩阵
 * @param aux_A_h     输入参数, 细辅助空间对应的矩阵, 其中 aux_A_h->mat 为细空间 V_h 对应的矩阵
 * @param aux_u_h     输入参数, 细辅助空间的向量组
 * @param block_size  输入参数, aux_u_h 的维数
 *
 * @return PASE_AUX_MATRIX
 */
PASE_AUX_MATRIX 
PASE_Aux_matrix_create_by_aux_matrix(PASE_MATRIX A_H, PASE_MATRIX R_hH, PASE_AUX_MATRIX aux_A_h, PASE_AUX_VECTOR *aux_u_h, PASE_INT block_size)
{
#if DEBUG_PASE_AUX_MATRIX
  if(NULL == A_H) {
    PASE_Error(__FUNCT__": Cannot create PASE AUX MATRIX without coarse grid matrix A_H.\n");
  }
  if(NULL == R_hH) {
    PASE_Error(__FUNCT__": Cannot create PASE AUX MATRIX without restrict matrix R_hH.\n");
  }
  if(NULL == aux_A_h) {
    PASE_Error(__FUNCT__": Cannot create PASE AUX MATRIX without fine grid aux matrix aux_A_h.\n");
  }
  if(NULL == aux_u_h) {
    PASE_Error(__FUNCT__": Cannot create PASE AUX MATRIX without fine grid aux vectors aux_u_h.\n");
  }
  if(0 >= block_size) {
    PASE_Error(__FUNCT__": Cannot create PASE AUX MATRIX with a nonpostive block size %d.\n", block_size);
  }
#endif

  PASE_AUX_MATRIX aux_A = (PASE_AUX_MATRIX)PASE_Malloc(sizeof(PASE_AUX_MATRIX_PRIVATE));
  aux_A->mat            = A_H;
  aux_A->is_mat_owner   = PASE_NO;
  aux_A->block_size     = block_size;
  aux_A->vec            = NULL;
  aux_A->block          = NULL;
  PASE_Aux_matrix_set_aux_space_by_aux_matrix(aux_A, R_hH, aux_A_h, aux_u_h);
  return aux_A;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_set_aux_space_some_by_aux_matrix"
/**
 * @brief 设置 aux_A = [mat   vec  ] 从第 i 个到第 j 个的辅助空间,
 *                     [vec^T block]
 *        即 vec[i],...,vec[j], 以及 block[i][:],...,block[j][:],
 *        使其成为 V_H + span{ aux_u_h[0], ..., aux_u_h[aux_A->block_size-1]} 对应的矩阵.
 *
 *        其中, vec[k]      = R_hH * (aux_A_h*aux_u_h[k])->vec,     i <= k <= j,
 *              block[k][l] = aux_u_h[k]^T * aux_A_h * aux_u_h[l], i <= k <= j, 0 <= l <= (aux_A->block_size-1).
 *
 *        通常用于 1. 初次创建 aux_A (i=0, j=aux_A->block_size-1), 
 *                 2. 更新 aux_A 的部分辅助空间 (比如, i>0).
 *        
 *
 * @param aux_A    输入/输出参数
 * @param i        输入参数, 起始位置, i >= 0
 * @param j        输入参数, 终止位置, j <= (aux_A->block_size-1)
 * @param R_hH     输入参数, 细空间 V_h 到粗空间 V_H 的限制矩阵
 * @param aux_A_h  输入参数, 细辅助空间对应的矩阵
 * @param aux_u_h  输入参数, 细辅助空间的向量组 
 */
void 
PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX R_hH, PASE_AUX_MATRIX aux_A_h, PASE_AUX_VECTOR *aux_u_h)
{
#if DEBUG_PASE_AUX_MATRIX
  if(NULL == aux_A) {
    PASE_Error(__FUNCT__": Cannot set some aux space without PASE AUX MATRIX.\n");
  }
  if(0 > i || j >= aux_A->block_size || 0 >= aux_A->block_size) {
    PASE_Error(__FUNCT__": Cannot set aux space from %dth to %dth of PASE AUX MATRIX with block size being %d.\n", i, j, aux_A->block_size);
  }
  if(NULL == R_hH) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without restrict matrix R_hH.\n");
  }
  if(NULL == aux_A_h) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without fine grid aux matrix aux_A_h.\n");
  }
  if(NULL == aux_u_h) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without fine grid aux vectors aux_u_h.\n");
  }
#endif 

  PASE_Aux_matrix_set_block_some_by_aux_matrix(aux_A, i, j, aux_A_h, aux_u_h);
  PASE_INT k;
  if(NULL == aux_A->vec) {
    aux_A->vec = (PASE_VECTOR*)PASE_Malloc(aux_A->block_size*sizeof(PASE_VECTOR));
    for(k = 0; k < aux_A->block_size; ++k) {
      aux_A->vec[k] = PASE_Vector_create_by_matrix_and_vector_data_operator(aux_A->mat, aux_A_h->vec[0]->ops);
    }
  }
  PASE_AUX_VECTOR aux_workspace_h = PASE_Aux_vector_create_by_aux_vector(aux_u_h[0]);
  for(k = i; k <= j; ++k) {
    PASE_Aux_matrix_multiply_aux_vector(aux_A_h, aux_u_h[k], aux_workspace_h);
    PASE_Matrix_multiply_vector(R_hH, aux_workspace_h->vec, aux_A->vec[k]);
  }
  PASE_Aux_vector_destroy(aux_workspace_h);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_set_aux_space_by_aux_matrix"
/**
 * @brief 设置 aux_A = [mat   vec  ] 的全部辅助空间, 即 vec 和 block,
 *                     [vec^T block]
 *        使其成为 V_H + span{ aux_u_h[0], ..., aux_u_h[aux_A->block_size-1]} 对应的矩阵.
 *
 *        其中, vec[k]      = R_hH * A_h * u_h[k],     i <= k <= j,
 *              block[k][l] = u_h[k]^T * A_h * u_h[l], i <= k <= j, 0 <= l <= (aux_A->block_size-1).
 *
 *
 * @param aux_A    输入/输出参数
 * @param R_hH     输入参数, 细空间 V_h 到粗空间 V_H 的限制矩阵
 * @param aux_A_h  输入参数, 细辅助空间对应的矩阵
 * @param aux_u_h  输入参数, 细辅助空间的向量组 
 */
void 
PASE_Aux_matrix_set_aux_space_by_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_MATRIX R_hH, PASE_AUX_MATRIX aux_A_h, PASE_AUX_VECTOR *aux_u_h)
{
#if DEBUG_PASE_AUX_MATRIX
  if(NULL == aux_A) {
    PASE_Error(__FUNCT__": Cannot set some aux space without PASE AUX MATRIX.\n");
  }
  if(0 >= aux_A->block_size) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX with block size being %d.\n", aux_A->block_size);
  }
  if(NULL == R_hH) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without restrict matrix R_hH.\n");
  }
  if(NULL == aux_A_h) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without fine grid aux matrix aux_A_h.\n");
  }
  if(NULL == aux_u_h) {
    PASE_Error(__FUNCT__": Cannot set aux space of PASE AUX MATRIX without fine grid aux vectors aux_u_h.\n");
  }
#endif 

  PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(aux_A, 0, aux_A->block_size-1, R_hH, aux_A_h, aux_u_h);
}

void
PASE_Aux_matrix_set_block_some_by_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_AUX_MATRIX aux_A_h, PASE_AUX_VECTOR *aux_u_h)
{
  PASE_INT k = 0;
  PASE_INT l = 0;
  if(NULL == aux_A->block) {
    aux_A->block = (PASE_SCALAR**)PASE_Malloc(aux_A->block_size*sizeof(PASE_SCALAR*));
    for(k = 0; k < aux_A->block_size; ++k) {
      aux_A->block[k] = (PASE_SCALAR*)PASE_Malloc(aux_A->block_size*sizeof(PASE_SCALAR));
    }
  }
  PASE_AUX_VECTOR aux_workspace_h = PASE_Aux_vector_create_by_aux_vector(aux_u_h[0]);
  for(k = i; k <= j; ++k) {
    PASE_Aux_matrix_multiply_aux_vector(aux_A_h, aux_u_h[k], aux_workspace_h);
    for(l = 0; l < aux_A->block_size; ++l) {
      if(l >= i && l <= j) {
        PASE_Aux_vector_inner_product(aux_workspace_h, aux_u_h[l], &(aux_A->block[k][l]));
      } else {
        PASE_Aux_vector_inner_product(aux_workspace_h, aux_u_h[l], &(aux_A->block[l][k]));
        PASE_Aux_vector_inner_product(aux_workspace_h, aux_u_h[l], &(aux_A->block[k][l]));
      }
    }
  }
  PASE_Aux_vector_destroy(aux_workspace_h);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_destroy"
/**
 * @brief 销毁辅助矩阵
 */
void 
PASE_Aux_matrix_destroy(PASE_AUX_MATRIX aux_A)
{
  if(NULL == aux_A) return;

#if DEBUG_PASE_AUX_MATRIX
  if((PASE_YES != aux_A->is_mat_owner) &&
     (PASE_NO  != aux_A->is_mat_owner)) {
    PASE_Error(__FUNCT__": Cannot decide whether the owner of matrix is.");
  }
#endif

  PASE_INT i;
  if(NULL != aux_A->mat && PASE_YES == aux_A->is_mat_owner) {
    PASE_Matrix_destroy(aux_A->mat);
  }
  if(NULL != aux_A->vec) {
    for(i = 0; i < aux_A->block_size; ++i) {
      PASE_Vector_destroy(aux_A->vec[i]);
    }
    PASE_Free(aux_A->vec);
  }
  if(NULL != aux_A->block) {
    for(i = 0; i < aux_A->block_size; ++i) {
      PASE_Free(aux_A->block[i]);
    }
    PASE_Free(aux_A->block);
  }
  PASE_Free(aux_A);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_copy"
/**
 * @brief 矩阵复制 (TODO 此函数有些问题, 未做测试, 暂时也不需要被调用)
 *
 * @param aux_A  输入参数, 被复制的矩阵
 * @param aux_B  输入参数, 复制得到的矩阵
 */
void 
PASE_Aux_matrix_copy(PASE_AUX_MATRIX aux_A, PASE_AUX_MATRIX aux_B)
{
#if DEBUG_PASE_AUX_MATRIX
  if(NULL == aux_A || NULL == aux_B) {
    PASE_Error(__FUNCT__": Neither the two matrices can be empty.\n");
  }
#endif 

  PASE_Matrix_copy(aux_A->mat, aux_B->mat);
  memcpy(aux_B->block, aux_A->block, aux_A->block_size*aux_A->block_size*sizeof(PASE_SCALAR));
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_multiply_aux_matrix"
/**
 * @brief 矩阵与矩阵相乘 aux_C = aux_A * aux_B (TODO 尚未实现, 暂时不需要用到这个函数)
 *
 * @param aux_A  输入参数
 * @param aux_B  输入参数
 * @param aux_C  输出参数
 */
void 
PASE_Aux_matrix_multiply_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_AUX_MATRIX aux_B, PASE_AUX_MATRIX aux_C)
{

}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_multiply_aux_vector"
/**
 * @brief 矩阵与向量相乘 aux_y = aux_A * aux_x (其中 y 不能与 x 相同)
 *
 * @param aux_A 输入参数
 * @param aux_x 输入参数
 * @param aux_y 输出参数
 */
void 
PASE_Aux_matrix_multiply_aux_vector(PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y)
{
#if DEBUG_PASE_AUX_MATRIX
  if((NULL == aux_A) || (NULL == aux_x) || (NULL == aux_y)) {
    PASE_Error(__FUNCT__": Matrix and vectors cannot be empty.\n");
  }
  if(aux_y == aux_x) {
    PASE_Error(__FUNCT__": Vectors aux_x and aux_y cannot be the same.\n");
  }
#endif

  PASE_INT i, j;

#if PASE_USE_HYPRE 
  clock_t start, end, start_total, end_total;
  MPI_Request request;
  MPI_Status  status;
  start_total = clock();
  if(PASE_NO == aux_A->is_diag) {
    start = clock();
    for(i = 0; i < aux_y->block_size; ++i) {
      aux_y->block[i] = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_A->vec[i]->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_x->vec->vector_data)));
    }
    MPI_Iallreduce(MPI_IN_PLACE, aux_y->block, aux_A->block_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);
    end = clock();
    aux_A->Tvecvec += ((double)(end-start))/1000000;
  } else {
    memset(aux_y->block, 0, aux_y->block_size*sizeof(PASE_SCALAR));
  }

  start = clock();
  PASE_Matrix_multiply_vector(aux_A->mat, aux_x->vec, aux_y->vec);
  end = clock();
  aux_A->Tmatvec += ((double)(end-start))/1000000;

  if(PASE_NO == aux_A->is_diag) {
    start = clock();
    for(i = 0; i < aux_A->block_size; ++i) {
      PASE_Vector_axpy(aux_x->block[i], aux_A->vec[i], aux_y->vec);
    }
    end = clock();
    aux_A->Tveccom += ((double)(end-start))/1000000;

    start = clock();
    MPI_Wait(&request, &status);
    end = clock();
    aux_A->Tvecvec += ((double)(end-start))/1000000;
  }

  start = clock();
  for(i = 0; i < aux_A->block_size; ++i) {
    for(j = 0; j<aux_A->block_size; ++j) {
      aux_y->block[i] += aux_A->block[i][j] * aux_x->block[j];
    }
  }
  end = clock();
  end_total = clock();
  aux_A->Tblockb += ((double)(end-start))/1000000;
  aux_A->Ttotal += ((double)(end_total-start_total))/1000000;

#else
  PASE_Matrix_multiply_vector(aux_A->mat, aux_x->vec, aux_y->vec);
  for(i = 0; i < aux_A->block_size; ++i) {
    PASE_Vector_axpy(aux_x->block[i], aux_A->vec[i], aux_y->vec);
    PASE_Vector_inner_product(aux_A->vec[i], aux_x->vec, &(aux_y->block[i]));
    for(j = 0; j<aux_A->block_size; ++j) {
      aux_y->block[i] += aux_A->block[i][j] * aux_x->block[j];
    }
  }
#endif
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_multiply_aux_vector_general"
/**
 * @brief 矩阵与向量相乘 aux_y = a * aux_A * aux_x  + b * aux_y (其中 aux_y 不能与 aux_x 相同)
 *
 * @param a      输入参数
 * @param aux_A  输入参数
 * @param aux_x  输入参数
 * @param b      输入参数
 * @param aux_y  输入/输出参数
 */
void
PASE_Aux_matrix_multiply_aux_vector_general(PASE_SCALAR a, PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_x, PASE_SCALAR b, PASE_AUX_VECTOR aux_y)
{
#if DEBUG_PASE_AUX_MATRIX
  if((NULL == aux_A) || (NULL == aux_x) || (NULL == aux_y)) {
    PASE_Error(__FUNCT__": Matrix and vectors cannot be empty.\n");
  }
  if(aux_y == aux_x) {
    PASE_Error(__FUNCT__": Vectors aux_x and aux_y cannot be the same.\n");
  }
#endif

  PASE_AUX_VECTOR aux_z = PASE_Aux_vector_create_by_aux_vector(aux_y);
  PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_x, aux_z);
  PASE_Aux_vector_scale(b, aux_y);
  PASE_Aux_vector_axpy(a, aux_z, aux_y);

  PASE_Aux_vector_destroy(aux_z);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_matrix_get_comm_info"
/**
 * @brief 获得矩阵通信器
 *
 * @param aux_A  输入参数
 * @param comm   输出参数
 */
void
PASE_Aux_matrix_get_mpi_comm(PASE_AUX_MATRIX aux_A, MPI_Comm *comm)
{
#if DEBUG_PASE_AUX_MATRIX
  if(NULL == aux_A) {
    PASE_Error(__FUNCT__": Matrix cannot be empty.\n");
  }
#endif

  *comm = PASE_Matrix_get_mpi_comm(aux_A->mat);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_create_by_aux_matrix"
/**
 * @brief 依据给定的辅助矩阵 aux_A, 创建一个新的辅助向量 
 *
 * @param aux_A 输入参数
 *
 * @return PASE_AUX_VECTOR
 */
PASE_AUX_VECTOR 
PASE_Aux_vector_create_by_aux_matrix(PASE_AUX_MATRIX aux_A)
{
#if DEBUG_PASE_AUX_MATRIX
  if(NULL == aux_A) {
    PASE_Error(__FUNCT__": Cannot create a new PASE AUX VECTOR without PASE AUX MATRIX.\n");
  }
#endif

  PASE_VECTOR  vec        = PASE_Vector_create_by_matrix_and_vector_data_operator(aux_A->mat, aux_A->vec[0]->ops);
  PASE_INT     block_size = aux_A->block_size;
  PASE_SCALAR *block      = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
  memset(block, 0, block_size*sizeof(PASE_SCALAR));

  PASE_AUX_VECTOR aux_y   = PASE_Aux_vector_create(vec, block, block_size);
  aux_y->is_vec_owner     = PASE_YES;
  return aux_y;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_inner_product_general"
/**
 * @brief 广义向量内积 *prod = aux_x^T *aux_A * aux_y
 *
 * @param aux_x  输入参数
 * @param aux_y  输入参数
 * @param aux_A  输入参数
 * @param prod   输出参数
 */
void
PASE_Aux_vector_inner_product_general(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y, PASE_AUX_MATRIX aux_A, PASE_REAL *prod)
{
#if PASE_USE_HYPRE 
  PASE_INT i = 0;
  PASE_INT j = 0;
  clock_t start_t, end_t;
  start_t = clock();

  if(PASE_NO == aux_A->is_diag) {
    MPI_Status status;
    MPI_Request request;
    PASE_VECTOR workspace = PASE_Vector_create_by_vector(aux_x->vec);
    PASE_Vector_set_constant_value(workspace, 0.0);
    PASE_SCALAR tmp = 0; 

    for(i = 0; i < aux_A->block_size; ++i) {
      PASE_Vector_axpy(aux_x->block[i], aux_A->vec[i], workspace);   
    }
    tmp += hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(workspace->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_y->vec->vector_data)));
    PASE_Matrix_multiply_vector(aux_A->mat, aux_y->vec, workspace);
    for(i = 0; i < aux_A->block_size; ++i) {
      PASE_Vector_axpy(aux_y->block[i], aux_A->vec[i], workspace);   
    }
    tmp += hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(workspace->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_x->vec->vector_data)));
    MPI_Iallreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);

    *prod = 0.0;
    for(i = 0; i < aux_A->block_size; ++i) {
      for(j = 0; j < aux_A->block_size; ++j) {
        *prod += aux_x->block[i] * aux_A->block[i][j] * aux_y->block[j];
      }
    }
    MPI_Wait(&request, &status);
    *prod += tmp;
    PASE_Vector_destroy(workspace);
  } else {
    PASE_Vector_inner_product_general(aux_x->vec, aux_y->vec, aux_A->mat, prod);
    for(i = 0; i < aux_A->block_size; ++i) {
      for(j = 0; j < aux_A->block_size; ++j) {
        *prod += aux_x->block[i] * aux_A->block[i][j] * aux_y->block[j];
      }
    }
  }

  end_t = clock();
  aux_A->Tinnergeneral += ((double)(end_t-start_t))/1000000;
#else
  PASE_AUX_VECTOR aux_workspace = PASE_Aux_vector_create_by_aux_vector(aux_x);
  PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_y, aux_workspace);
  clock_t start_t, end_t;
  start_t = clock();
  PASE_Aux_vector_inner_product(aux_x, aux_workspace, prod);
  end_t   = clock();
  aux_A->Tinnergeneral += ((double)(end_t-start_t))/1000000;
  PASE_Aux_vector_destroy(aux_workspace);
#endif

}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_inner_product_general_some"
/**
 * @brief 向量 aux_x[start], ..., aux_x[end] 全体计算 aux_A 内积
 *
 * @param aux_x  输入参数
 * @param start  输入参数
 * @param end    输入参数
 * @param aux_A  输入参数
 * @param prod   输出参数
 */
void
PASE_Aux_vector_inner_product_general_some(PASE_AUX_VECTOR *aux_x, PASE_INT start, PASE_INT end, PASE_AUX_MATRIX aux_A, PASE_REAL **prod)
{
#if DEBUG_PASE_AUX_MATRIX
  if((NULL == aux_A) || (NULL == aux_x) || (NULL == prod)) {
    PASE_Error(__FUNCT__": Matrix and vectors and products cannot be empty.\n");
  }
#endif

  PASE_AUX_VECTOR tmp_aux_Axi = PASE_Aux_vector_create_by_aux_vector(aux_x[start]);

  PASE_INT i = 0;
  PASE_INT j = 0;
  for(i = start; i <= end; ++i) {
    PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_x[i], tmp_aux_Axi);
    for(j = start; j <= i; ++j) {
      PASE_Aux_vector_inner_product(aux_x[j], tmp_aux_Axi, &(prod[i-start][j-start]));
      prod[j-start][i-start] = prod[i-start][j-start];
    }
  }

  PASE_Aux_vector_destroy(tmp_aux_Axi);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_orthogonalize_general"
/**
 * @brief 向量 aux_x[i] 与 aux_x[start], ..., aux_x[end] 在 aux_A 内积下正交化
 *
 * @param aux_x  输入/输出参数
 * @param i      输入参数
 * @param start  输入参数
 * @param end    输入参数
 * @parma aux_A  输入参数
 */
void
PASE_Aux_vector_orthogonalize_general(PASE_AUX_VECTOR *aux_x, PASE_INT i, PASE_INT start, PASE_INT end, PASE_AUX_MATRIX aux_A)
{
#if DEBUG_PASE_AUX_MATRIX
  if((NULL == aux_A) || (NULL == aux_x)) {
    PASE_Error(__FUNCT__": Matrix and vectors cannot be empty.\n");
  }
  if((i >= start) && (i <= end)) {
    PASE_Error(__FUNCT__": index %d cannot locate in [%d, %d].\n", i, start, end);
  }
#endif

  PASE_AUX_VECTOR tmp_aux_Axi = PASE_Aux_vector_create_by_aux_vector(aux_x[i]);
  PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_x[i], tmp_aux_Axi);
  
  PASE_INT  j    = 0;
  PASE_REAL prod = 0.0;
  for(j = start; j <= end; ++j) {
    PASE_Aux_vector_inner_product(aux_x[j], tmp_aux_Axi, &prod);
    PASE_Aux_vector_axpy(-prod, aux_x[j], aux_x[i]);
  }
  PASE_Aux_vector_inner_product(aux_x[i], tmp_aux_Axi, &prod);
  PASE_Aux_vector_scale(1.0/sqrt(prod), aux_x[i]);

  PASE_Aux_vector_destroy(tmp_aux_Axi);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_orthogonalize_general_all"
/**
 * @brief 向量 aux_x[0], ..., aux_x[num-1] 全体在 aux_A 内积下正交化
 * 
 * @param aux_x  输入/输出参数
 * @param num    输入参数
 * @param aux_A  输入参数
 */
void
PASE_Aux_vector_orthogonalize_general_all(PASE_AUX_VECTOR *aux_x, PASE_INT num, PASE_AUX_MATRIX aux_A)
{
#if DEBUG_PASE_AUX_MATRIX
  if((NULL == aux_A) || (NULL == aux_x)) {
    PASE_Error(__FUNCT__": Matrix and vectors cannot be empty.\n");
  }
#endif

  PASE_INT j = 0;
  for(j = 0; j < num; ++j) {
    PASE_Aux_vector_orthogonalize_general(aux_x, j, 0, j-1, aux_A);
  }
}

