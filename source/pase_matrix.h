#ifndef __PASE_MATRIX_H__
#define __PASE_MATRIX_H__
 
#include <stdio.h>
#include "pase_config.h"
#include "pase_vector.h"

/* 矩阵运算集合 */
typedef struct PASE_MATRIX_DATA_OPERATOR_PRIVATE_ {
  /**
   * @brief 由矩阵创建矩阵
   */
  void * (*create_by_matrix)(void *A);

  /**
   * @brief 矩阵复制
   * @param src  输入参数, 被复制的矩阵
   * @param dst  输出参数, 复制得到的矩阵
   */
  void (*copy)(void *src, void *dst);

  /**
   * @brief 矩阵销毁
   */
  void (*destroy)(void *A);

  /**
   * @brief 矩阵转置
   * @return 转置后的矩阵
   */
  void * (*transpose)(void *A);

  /**
   * @brief 矩阵矩阵相乘 A * B
   * @return 矩阵乘积
   */
  void * (*multiply_matrix_matrix)  (void *A, void *B);

  /**
   * @brief 矩阵矩阵相乘 A^T * B
   * @return 矩阵乘积
   */
  void * (*multiply_matrixT_matrix) (void *A, void *B);

  /**
   * @brief 矩阵向量相乘 y = A * x 
   */
  void (*multiply_matrix_vector)(void *A, void *x, void *y);

  /**
   * @brief 矩阵向量相乘 y = a * A * x  + b * y
   */
  void (*multiply_matrix_vector_general)(PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y);

  /**
   * @brief 矩阵向量相乘 y = A^T * x 
   */
  void (*multiply_matrixT_vector) (void *A, void *x, void *y);

  /**
   * @brief 矩阵向量相乘 y = a * A^T * x  + b * y
   */
  void (*multiply_matrixT_vector_general)(PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y);

  /**
   * @brief 获得矩阵全局行数
   */
  void (*get_global_nrow)(void *A, PASE_INT *nrow);

  /**
   * @brief 获得矩阵全局列数
   */
  void (*get_global_ncol)(void *A, PASE_INT *ncol);

  /**
   * @brief 获得矩阵通信器
   */
  MPI_Comm (*get_mpi_comm)(void *A);
} PASE_MATRIX_DATA_OPERATOR_PRIVATE;

typedef PASE_MATRIX_DATA_OPERATOR_PRIVATE * PASE_MATRIX_DATA_OPERATOR;

typedef struct PASE_MATRIX_PRIVATE_ {
  void                      *matrix_data; // 矩阵数据
  PASE_INT                   global_nrow; // 全局行数
  PASE_INT                   global_ncol; // 全局列数
  PASE_MATRIX_DATA_OPERATOR  ops;         // 矩阵运算集合
  PASE_INT                   is_matrix_data_owner; // 是否为矩阵数据属主
  PASE_INT                   data_form;   // 矩阵数据格式
} PASE_MATRIX_PRIVATE;

typedef PASE_MATRIX_PRIVATE * PASE_MATRIX;

/**
 * @brief 指定矩阵运算集合
 *      
 * @param create_by_matrix                 输入参数, 函数指针, 由矩阵生成矩阵
 * @param destroy                          输入参数, 函数指针, 销毁矩阵数据
 * @param transpose                        输入参数, 函数指针, 矩阵数据转置
 * @param copy                             输入参数, 函数指针, 矩阵数据复制
 * @param multiply_matrix_matrix           输入参数, 函数指针, 矩阵与矩阵相乘
 * @param multiply_matrixT_matrix          输入参数, 函数指针, 矩阵转置与矩阵相乘
 * @param multiply_matrix_vector           输入参数, 函数指针, 矩阵与向量相乘
 * @param multiply_matrixT_vector          输入参数, 函数指针, 矩阵转置与向量相乘
 * @param multiply_matrix_vector_general   输入参数, 函数指针, 矩阵与向量相乘并与另一向量做线性组合
 * @param multiply_matrixT_vector_general  输入参数, 函数指针, 矩阵转置与向量相乘并与另一向量做线性组合
 * @param get_global_nrow                  输入参数, 函数指针, 获取矩阵全局行数
 * @param get_global_ncol                  输入参数, 函数指针, 获取矩阵全局列数
 * @param get_mpi_comm_info                输入参数, 函数指针, 获取矩阵通信器
 *
 * @note  multiply_matrix_vector 与 multiply_matrix_vector_general 不能同时为 NULL
 * @note  multiply_matrixT_vector 与 multiply_matrixT_vector_general 不能同时为 NULL
 *
 * @return PASE_VECTOR_DATA_OPERATOR
 */
PASE_MATRIX_DATA_OPERATOR PASE_Matrix_data_operator_assign
    (void * (*create_by_matrix)               (void *A),
     void   (*destroy)                        (void *A),
     void * (*transpose)                      (void *A),
     void   (*copy)                           (void *A, void *B),
     void * (*multiply_matrix_matrix)         (void *A, void *B),
     void * (*multiply_matrixT_matrix)        (void *A, void *B),
     void   (*multiply_matrix_vector)         (void *A, void *x, void *y), // 可以是 NULL
     void   (*multiply_matrixT_vector)        (void *A, void *x, void *y), // 可以是 NULL
     void   (*multiply_matrix_vector_general) (PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y),
     void   (*multiply_matrixT_vector_general)(PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y),
     void   (*get_global_nrow)                (void *A, PASE_INT *nrow),
     void   (*get_global_ncol)                (void *A, PASE_INT *ncol),
     MPI_Comm (*get_mpi_comm_info)                (void *A));

/**
 * @brief 依据指定格式生成矩阵运算集合, 需要在编译时指定可用的格式
 *
 * @param data_form  输入参数, 数据格式类型
 *                   1: PACKAGE_HYPRE
 *                   2: PACKAGE_JXPAMG
 *
 * @return PASE_MATRIX_DATA_OPERATOR
 */
PASE_MATRIX_DATA_OPERATOR PASE_Matrix_data_operator_create(PASE_INT data_form);

/**
 * @brief 销毁矩阵运算集合
 */
void PASE_Matrix_data_operator_destroy(PASE_MATRIX_DATA_OPERATOR ops);

/**
 * @brief 将矩阵数据与运算集合关联至矩阵
 *
 * @param matrix_data  输入参数, void * 指针, 指向矩阵数据
 * @param ops          输入参数, 矩阵运算集合
 *
 * @return PASE_MATRIX
 */
PASE_MATRIX PASE_Matrix_assign(void *matrix_data, PASE_MATRIX_DATA_OPERATOR ops);

/**
 * @brief 依据指定格式生成矩阵
 *
 * @param matrix_data  输入参数, void * 指针, 指向矩阵数据
 * @param data_form    输入参数, 数据格式类型
 *
 * @return PASE_MATRIX
 */
PASE_MATRIX PASE_Matrix_create(void *matrix_data, PASE_INT data_form);

/**
 * @brief 销毁矩阵
 */
void PASE_Matrix_destroy(PASE_MATRIX A);


/**
 * @brief 矩阵转置
 * @return 转置后的矩阵
 */
PASE_MATRIX PASE_Matrix_transpose(PASE_MATRIX A);

/**
 * @brief 矩阵复制
 * @param A  输入参数, 被复制的矩阵
 * @param B  输出参数, 复制得到的矩阵
 */
void PASE_Matrix_copy(PASE_MATRIX A, PASE_MATRIX B);

/**
 * @brief 矩阵与矩阵相乘 A * B
 * @return 矩阵乘积
 */
PASE_MATRIX PASE_Matrix_multiply_matrix(PASE_MATRIX A, PASE_MATRIX B);

/**
 * @brief 矩阵转置与矩阵相乘 A^T * B
 * @return 矩阵乘积
 */
PASE_MATRIX PASE_MatrixT_multiply_matrix(PASE_MATRIX A, PASE_MATRIX B);

/**
 * @brief 矩阵与向量相乘 y = A * x (其中 y 不能与 x 相同)
 */
void PASE_Matrix_multiply_vector(PASE_MATRIX A, PASE_VECTOR x, PASE_VECTOR y);

/**
 * @brief 矩阵与向量相乘 y = a * A * x  + b * y (其中 y 不能与 x 相同)
 */
void PASE_Matrix_multiply_vector_general(PASE_SCALAR a, PASE_MATRIX A, PASE_VECTOR x, 
                                         PASE_SCALAR b, PASE_VECTOR y);

/**
 * @brief 矩阵转置与向量相乘 y = AT * x (其中 y 不能与 x 相同)
 */
void PASE_MatrixT_multiply_vector(PASE_MATRIX A, PASE_VECTOR x, PASE_VECTOR y);

/**
 * @brief 矩阵转置与向量相乘 y = a * AT * x  + b * y (其中 y 不能与 x 相同)
 */
void PASE_MatrixT_multiply_vector_general(PASE_SCALAR a, PASE_MATRIX A, PASE_VECTOR x, 
                                          PASE_SCALAR b, PASE_VECTOR y);

/**
 * @brief 获得矩阵全局行数
 */
void PASE_Matrix_get_global_nrow(PASE_MATRIX A, PASE_INT *nrow);

/**
 * @brief 获得矩阵全局列数
 */
void PASE_Matrix_get_global_ncol(PASE_MATRIX A, PASE_INT *ncol);

/**
 * @brief 获得矩阵通信器
 */
MPI_Comm PASE_Matrix_get_mpi_comm(PASE_MATRIX A);


/**
 * @brief 由矩阵和向量运算集合创建新的向量
 * @param A    输入参数, 矩阵
 * @param ops  输入参数, 向量运算集合, 若为 NULL 则依据矩阵的数据格式生成运算集合
 */
PASE_VECTOR PASE_Vector_create_by_matrix_and_vector_data_operator(PASE_MATRIX A, PASE_VECTOR_DATA_OPERATOR ops);

/**
 * @brief 广义向量内积 *prod = (Ax, y)
 */
void PASE_Vector_inner_product_general(PASE_VECTOR x, PASE_VECTOR y, PASE_MATRIX A, PASE_REAL *prod);

/**
 * @brief 向量 x[start], ..., x[end] 全体 A 内积
 */
void PASE_Vector_inner_product_general_some(PASE_VECTOR *x, PASE_INT start, PASE_INT end, 
                                            PASE_MATRIX A, PASE_REAL **prod);

/**
 * @brief 向量 x[i] 与 x[start], ..., x[end] 在 A 内积下正交化, i 不能属于 [start, end]
 */
void PASE_Vector_orthogonalize_general(PASE_VECTOR *x, 
                                       PASE_INT i, PASE_INT start, PASE_INT end, 
                                       PASE_MATRIX A);

/**
 * @brief 向量 x[0], ..., x[num-1] 全体在 A 内积下正交化
 */
void PASE_Vector_orthogonalize_general_all(PASE_VECTOR *x, PASE_INT num, PASE_MATRIX A);

#endif
