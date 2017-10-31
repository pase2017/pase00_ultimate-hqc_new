#ifndef __PASE_MATRIX_H__
#define __PASE_MATRIX_H__
 
#include <mpi.h>
#include "stdio.h"
#include "pase_param.h"
#include "pase_config.h"


/* 矩阵相关操作函数集合 */
typedef struct PASE_MATRIX_OPERATOR_PRIVATE_ {
    void *   (*create_matrix_by_matrix) (void *A);
    void     (*copy_matrix)             (void *A, void *B);
    void     (*destroy_matrix)          (void *A);
    /**
     * @brief 矩阵矩阵相乘 C = A * B
     */
    void *   (*multiply_matrix_matrix)  (void *A, void *B);
    void *   (*multiply_matrixT_matrix) (void *A, void *B);
    /**
     * @brief 矩阵向量相乘 y = A * x 
     */
    void     (*multiply_matrix_vector)  (void *A, void *x, void *y);
    void     (*multiply_matrixT_vector) (void *A, void *x, void *y);
    PASE_INT (*get_global_nrow)         (void *A);
    PASE_INT (*get_global_ncol)         (void *A);
    MPI_Comm (*get_comm_info)           (void *A);
} PASE_MATRIX_OPERATOR_PRIVATE;
typedef PASE_MATRIX_OPERATOR_PRIVATE * PASE_MATRIX_OPERATOR;

typedef struct PASE_MATRIX_PRIVATE_ {
  void                 *matrix_data;
  PASE_INT              global_nrow; // 行数
  PASE_INT              global_ncol; // 列数
  PASE_MATRIX_OPERATOR  ops;
  PASE_INT              is_matrix_data_owner;
  PASE_INT              data_struct;
  //PASE_INT              is_ops_owner;
} PASE_MATRIX_PRIVATE;
typedef PASE_MATRIX_PRIVATE * PASE_MATRIX;

#include "pase_vector.h"
/**
 * @brief 通过此函数进行外部矩阵类型到 PASE_MATRIX 的转换.
 *        例如对于 HYPRE 矩阵, 可设置 external_package 为 HYPRE.
 */
PASE_MATRIX PASE_Matrix_create_by_operator(void *matrix_data, PASE_MATRIX_OPERATOR ops);
PASE_MATRIX PASE_Matrix_create_default(void *matrix_data, PASE_INT data_struct);

PASE_MATRIX_OPERATOR PASE_Matrix_operator_create
    (void *   (*create_matrix_by_matrix) (void *A),
     void     (*copy_matrix)             (void *A, void *B),
     void     (*destroy_matrix)          (void *A),
     void *   (*multiply_matrix_matrix)  (void *A, void *B),
     void     (*multiply_matrix_vector)  (void *A, void *x, void *y),
     PASE_INT (*get_global_nrow)         (void *A),
     PASE_INT (*get_global_ncol)         (void *A),
     MPI_Comm (*get_comm_info)           (void *A));
PASE_MATRIX_OPERATOR PASE_Matrix_operator_create_default(PASE_INT data_struct);
void PASE_Matrix_operator_destroy(PASE_MATRIX_OPERATOR ops);

void     PASE_Matrix_destroy(PASE_MATRIX A);
void     PASE_Matrix_copy(PASE_MATRIX A, PASE_MATRIX B);
PASE_MATRIX PASE_Matrix_multiply_matrix(PASE_MATRIX A, PASE_MATRIX B);
void     PASE_Matrix_multiply_vector(PASE_MATRIX A, PASE_VECTOR x, PASE_VECTOR y);
void     PASE_Matrix_multiply_vector_general(PASE_SCALAR a, PASE_MATRIX A, PASE_VECTOR x, PASE_SCALAR b, PASE_VECTOR y);
MPI_Comm PASE_Matrix_get_comm_info(PASE_MATRIX A);

void*    PASE_Matrix_create_by_matrix_hypre(void *A);
void     PASE_Matrix_copy_hypre(void *A, void *B);
void     PASE_Matrix_destroy_hypre(void *A);
void*    PASE_Matrix_multiply_matrix_hypre(void *A, void *B);
void     PASE_Matrix_multiply_vector_hypre(void *A, void *x, void *y);
PASE_INT PASE_Matrix_get_global_nrow_hypre(void *A);
PASE_INT PASE_Matrix_get_global_ncol_hypre(void *A);
MPI_Comm PASE_Matrix_get_comm_info_hypre(void *A);

#endif
