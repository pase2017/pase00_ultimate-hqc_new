#ifndef __PASE_AUX_MATRIX_H__
#define __PASE_AUX_MATRIX_H__

#include "pase_matrix.h"
#include "pase_vector.h"
#include "pase_aux_vector.h"

//typedef struct PASE_AUX_MATRIX_OPERATOR_PRIVATE_ {
//
//} PASE_AUX_MATRIX_OPERATOR_PRIVATE; 
//typedef PASE_AUX_MATRIX_OPERATOR_PRIVATE * PASE_AUX_MATRIX_OPERATOR;

/*
 * aux_matrix = [mat    vec  ]
 *              [vec^T  block]
 */
typedef struct PASE_AUX_MATRIX_PRIVATE_ {
  PASE_MATRIX   mat; 
  PASE_VECTOR  *vec;
  PASE_SCALAR **block;
  PASE_INT      block_size;   //辅助空间的维数
  PASE_INT      is_mat_owner; //是否为 mat 属主

  PASE_INT  is_diag;
#if 1
  PASE_REAL Tmatvec;
  PASE_REAL Tvecvec;
  PASE_REAL Tveccom;
  PASE_REAL Tblockb; 
  PASE_REAL Ttotal;
  PASE_REAL Tinnergeneral;
#endif
} PASE_AUX_MATRIX_PRIVATE;
typedef PASE_AUX_MATRIX_PRIVATE * PASE_AUX_MATRIX;

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
PASE_AUX_MATRIX PASE_Aux_matrix_create(PASE_MATRIX A_H, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h, PASE_INT block_size);

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
void PASE_Aux_matrix_set_aux_space_some(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h); 

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
void PASE_Aux_matrix_set_aux_space(PASE_AUX_MATRIX aux_A, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h); 

PASE_INT PASE_Aux_matrix_set_vec_some(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h);

PASE_INT PASE_Aux_matrix_set_vec(PASE_AUX_MATRIX aux_A, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h);

PASE_INT PASE_Aux_matrix_set_block_some(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX A_h, PASE_VECTOR *u_h);

PASE_INT PASE_Aux_matrix_set_block(PASE_AUX_MATRIX aux_A, PASE_MATRIX A_h, PASE_VECTOR *u_h);

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
PASE_AUX_MATRIX PASE_Aux_matrix_create_by_aux_matrix(PASE_MATRIX A_H, PASE_MATRIX R_hH, PASE_AUX_MATRIX aux_A_h, PASE_AUX_VECTOR *aux_u_h, PASE_INT block_size);

/**
 * @brief 设置 aux_A = [mat   vec  ] 从第 i 个到第 j 个的辅助空间,
 *                     [vec^T block]
 *        即 vec[i],...,vec[j], 以及 block[i][:],...,block[j][:],
 *        使其成为 V_H + span{ aux_u_h[0], ..., aux_u_h[aux_A->block_size-1]} 对应的矩阵.
 *
 *        其中, vec[k]      = R_hH * A_h * u_h[k],     i <= k <= j,
 *              block[k][l] = u_h[k]^T * A_h * u_h[l], i <= k <= j, 0 <= l <= (aux_A->block_size-1).
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
void PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX R_hH, PASE_AUX_MATRIX aux_A_h, PASE_AUX_VECTOR *aux_u_h);

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
void PASE_Aux_matrix_set_aux_space_by_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_MATRIX R_hH, PASE_AUX_MATRIX aux_A_h, PASE_AUX_VECTOR *aux_u_h);

void PASE_Aux_matrix_set_block_some_by_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_AUX_MATRIX aux_A_h, PASE_AUX_VECTOR *aux_u_h);

/**
 * @brief 销毁辅助矩阵
 */
void PASE_Aux_matrix_destroy(PASE_AUX_MATRIX aux_A);
 
/**
 * @brief 矩阵复制
 *
 * @param aux_A  输入参数, 被复制的矩阵
 * @param aux_B  输入参数, 复制得到的矩阵
 */
void PASE_Aux_matrix_copy(PASE_AUX_MATRIX aux_A, PASE_AUX_MATRIX aux_B);

/**
 * @brief 矩阵与矩阵相乘 aux_C = aux_A * aux_B
 *
 * @param aux_A  输入参数
 * @param aux_B  输入参数
 * @param aux_C  输出参数
 */
void PASE_Aux_matrix_multiply_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_AUX_MATRIX aux_B, PASE_AUX_MATRIX aux_C);

/**
 * @brief 矩阵与向量相乘 aux_y = aux_A * aux_x (其中 y 不能与 x 相同)
 *
 * @param aux_A 输入参数
 * @param aux_x 输入参数
 * @param aux_y 输出参数
 */
void PASE_Aux_matrix_multiply_aux_vector(PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y);

/**
 * @brief 矩阵与向量相乘 aux_y = a * aux_A * aux_x  + b * aux_y (其中 aux_y 不能与 aux_x 相同)
 *
 * @param a      输入参数
 * @param aux_A  输入参数
 * @param aux_x  输入参数
 * @param b      输入参数
 * @param aux_y  输入/输出参数
 */
void PASE_Aux_matrix_multiply_aux_vector_general(PASE_SCALAR a, PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_x, PASE_SCALAR b, PASE_AUX_VECTOR aux_y);

/**
 * @brief 获得矩阵通信器
 *
 * @param aux_A  输入参数
 * @param comm   输出参数
 */
void PASE_Aux_matrix_get_mpi_comm(PASE_AUX_MATRIX aux_A, MPI_Comm *comm);

/**
 * @brief 依据给定的辅助矩阵 aux_A, 创建一个新的辅助向量 
 *
 * @param aux_A 输入参数
 *
 * @return PASE_AUX_VECTOR
 */
PASE_AUX_VECTOR PASE_Aux_vector_create_by_aux_matrix(PASE_AUX_MATRIX aux_A);

/**
 * @brief 广义向量内积 *prod = aux_x^T *aux_A * aux_y
 *
 * @param aux_x  输入参数
 * @param aux_y  输入参数
 * @param aux_A  输入参数
 * @param prod   输出参数
 */
void PASE_Aux_vector_inner_product_general(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y, PASE_AUX_MATRIX aux_A, PASE_REAL *prod);

/**
 * @brief 向量 aux_x[start], ..., aux_x[end] 全体计算 aux_A 内积
 *
 * @param aux_x  输入参数
 * @param start  输入参数
 * @param end    输入参数
 * @param aux_A  输入参数
 * @param prod   输出参数
 */
void PASE_Aux_vector_inner_product_general_some(PASE_AUX_VECTOR *aux_x, PASE_INT start, PASE_INT end, PASE_AUX_MATRIX aux_A, PASE_REAL **prod);

/**
 * @brief 向量 aux_x[i] 与 aux_x[start], ..., aux_x[end] 在 aux_A 内积下正交化
 *
 * @param aux_x  输入/输出参数
 * @param i      输入参数
 * @param start  输入参数
 * @param end    输入参数
 * @parma aux_A  输入参数
 */
void PASE_Aux_vector_orthogonalize_general(PASE_AUX_VECTOR *aux_x, PASE_INT i, PASE_INT start, PASE_INT end, PASE_AUX_MATRIX aux_A);

/**
 * @brief 向量 aux_x[0], ..., aux_x[num-1] 全体在 aux_A 内积下正交化
 * 
 * @param aux_x  输入/输出参数
 * @param num    输入参数
 * @param aux_A  输入参数
 */
void PASE_Aux_vector_orthogonalize_general_all(PASE_AUX_VECTOR *aux_x, PASE_INT num, PASE_AUX_MATRIX aux_A);

#endif
