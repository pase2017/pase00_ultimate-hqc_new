#ifndef __PASE_MATRIX_H__
#define __PASE_MATRIX_H__

#define PASE_Malloc malloc
#define PASE_Free free

#include "pase_vector.h"


/* 矩阵相关操作函数集合 */
typedef struct PASE_MATRIX_OPERATOR_PRIVATE_ {
    void *   (*create_matrix_by_matrix) (void *A);
    void     (*copy_matrix)             (void *A, void *B);
    void     (*destroy_matrix)          (void *A);
    /**
     * @brief 矩阵矩阵相乘 C = A * B
     */
    void *   (*multiply_matrix_matrix)  (void *A, void *B);
    /**
     * @brief 矩阵向量相乘 y = a * A * x + b * y
     */
    void     (*multiply_matrix_vector)  (PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y);
    PASE_INT (*get_global_nrow)         (void *A);
    PASE_INT (*get_global_ncol)         (void *A);
} PASE_MATRIX_OPERATOR_PRIVATE;
typedef PASE_MATRIX_OPERATOR_PRIVATE * PASE_MATRIX_OPERATOR;

typedef struct PASE_MATRIX_PRIVATE_ {
  void                 *matrix_data;
  PASE_INT              global_nrow; // 行数
  PASE_INT              global_ncol; // 列数
  PASE_MATRIX_OPERATOR  ops;
  PASE_INT              is_matrix_data_owner;
  PASE_INT              is_ops_owner;
} PASE_MATRIX_PRIVATE;
typedef PASE_MATRIX_PRIVATE * PASE_MATRIX;

/**
 * @brief 通过此函数进行外部矩阵类型到 PASE_MATRIX 的转换.
 *        例如对于 HYPRE 矩阵, 可设置 external_package 为 HYPRE.
 */
PASE_MATRIX PASE_Create_matrix(void *matrix_data, PASE_PARAMETER param, PASE_MATRIX_OPERATOR ops);
PASE_MATRIX_OPERATOR PASE_Create_matrix_operator(
    void *   (*create_matrix_by_matrix) (void *A),
    void     (*copy_matrix)             (void *A, void *B),
    void     (*destroy_matrix)          (void *A),
    void *   (*multiply_matrix_matrix)  (void *A, void *B),
    void     (*multiply_matrix_vector)  (PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y),
    PASE_INT (*get_global_nrow)         (void *A),
    PASE_INT (*get_global_ncol)         (void *A)
	);
PASE_MATRIX_OPERATOR PASE_Create_matrix_operator_default( PASE_PARAMETER param);
void PASE_Destroy_matrix_operator(PASE_MATRIX_OPERATOR ops);

void* PASE_Hypre_create_matrix_by_matrix(void *A);
void PASE_Hypre_copy_matrix(void *A, void *B);
PASE_Int PASE_Hypre_get_matrix_global_nrow(void *A);
PASE_Int PASE_Hypre_get_matrix_global_ncol(void *A);

void PASE_Destroy_matrix(PASE_MATRIX A);
PASE_MATRIX PASE_Matrix_multiply_matrix(PASE_MATRIX A, PASE_MATRIX B);

void PASE_Copy_matrix(PASE_MATRIX A, PASE_MATRIX B);


#endif
