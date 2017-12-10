#include <math.h>
#include "pase_config.h"
#include "pase_matrix.h"

#ifdef PASE_USE_HYPRE
#include "pase_matrix_hypre.h"
#endif

#define DEBUG_PASE_MATRIX 1

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_data_operator_assign"
/**
 * @brief 指定矩阵运算集合
 */
PASE_MATRIX_DATA_OPERATOR 
PASE_Matrix_data_operator_assign
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
     MPI_Comm (*get_mpi_comm)                 (void *A))
{
  PASE_MATRIX_DATA_OPERATOR ops;
  ops = (PASE_MATRIX_DATA_OPERATOR)PASE_Malloc(sizeof(PASE_MATRIX_DATA_OPERATOR_PRIVATE));
  ops->create_by_matrix                = create_by_matrix;
  ops->destroy                         = destroy;
  ops->transpose                       = transpose;
  ops->copy                            = copy;
  ops->multiply_matrix_matrix          = multiply_matrix_matrix;
  ops->multiply_matrixT_matrix         = multiply_matrixT_matrix;
  ops->multiply_matrix_vector          = multiply_matrix_vector;
  ops->multiply_matrixT_vector         = multiply_matrixT_vector;
  ops->multiply_matrix_vector_general  = multiply_matrix_vector_general;
  ops->multiply_matrixT_vector_general = multiply_matrixT_vector_general;
  ops->get_global_nrow                 = get_global_nrow;
  ops->get_global_ncol                 = get_global_ncol;
  ops->get_mpi_comm                    = get_mpi_comm;

  return ops;
}


#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_operator_data_create"
/**
 * @brief 依据指定格式生成矩阵运算集合
 */
PASE_MATRIX_DATA_OPERATOR 
PASE_Matrix_data_operator_create(PASE_INT data_form)
{
  PASE_MATRIX_DATA_OPERATOR ops = NULL;
#if PASE_USE_HYPRE
  if(PACKAGE_HYPRE == data_form) {
    ops = PASE_Matrix_data_operator_assign(PASE_Matrix_create_by_matrix_hypre,
                                           PASE_Matrix_destroy_hypre,
                                           PASE_Matrix_transpose_hypre,
                                           PASE_Matrix_copy_hypre,
                                           PASE_Matrix_multiply_matrix_hypre,
                                           PASE_MatrixT_multiply_matrix_hypre,
                                           PASE_Matrix_multiply_vector_hypre,
                                           PASE_MatrixT_multiply_vector_hypre,
                                           PASE_Matrix_multiply_vector_general_hypre,
                                           PASE_MatrixT_multiply_vector_general_hypre,
                                           PASE_Matrix_get_global_nrow_hypre,
                                           PASE_Matrix_get_global_ncol_hypre,
                                           PASE_Matrix_get_mpi_comm_hypre);
  }
#endif

#if PASE_USE_JXPAMG
  if(PACKAGE_JXPAMG == data_form) {
  }
#endif

  if(NULL == ops) {
    PASE_Error(__FUNCT__": Cannot find data_form %d.\n", data_form);
  }
  return ops;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_data_operator_destroy"
/**
 * @brief 销毁矩阵运算集合
 */
void
PASE_Matrix_data_operator_destroy(PASE_MATRIX_DATA_OPERATOR ops)
{
  PASE_Free(ops);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_assign"
/**
 * @brief 将矩阵数据与运算集合关联至矩阵
 */
PASE_MATRIX 
PASE_Matrix_assign(void *matrix_data, PASE_MATRIX_DATA_OPERATOR ops)
{
#if DEBUG_PASE_MATRIX
  if(NULL == matrix_data) {
    PASE_Error(__FUNCT__": Cannot create PASE MATRIX without matrix data.\n");
  }
  if(NULL == ops) {
    PASE_Error(__FUNCT__": Cannot create PASE MATRIX without matrix data operator.\n");
  }
  if(NULL == ops->get_global_nrow) {
    PASE_Error(__FUNCT__": ops->get_global_nrow is not assigned.\n");
  }
  if(NULL == ops->get_global_ncol) {
    PASE_Error(__FUNCT__": ops->get_global_ncol is not assigned.\n");
  }
#endif
  
  PASE_MATRIX A = (PASE_MATRIX)PASE_Malloc(sizeof(PASE_MATRIX_PRIVATE));
  A->ops = (PASE_MATRIX_DATA_OPERATOR)PASE_Malloc(sizeof(PASE_MATRIX_DATA_OPERATOR_PRIVATE));
  *(A->ops)               = *ops;
  A->matrix_data          = matrix_data;
  A->is_matrix_data_owner = PASE_NO;
  A->data_form            = PASE_USER;
  A->ops->get_global_nrow(A->matrix_data, &A->global_nrow);
  A->ops->get_global_ncol(A->matrix_data, &A->global_ncol);
  return A;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_create"
/**
 * @brief 依据指定格式生成矩阵
 */
PASE_MATRIX 
PASE_Matrix_create(void *matrix_data, PASE_INT data_form)
{
#if DEBUG_PASE_MATRIX
  if(NULL == matrix_data) {
    PASE_Error(__FUNCT__": Cannot create PASE MATRIX without matrix data.\n");
  }
#endif
  
  PASE_MATRIX A = (PASE_MATRIX)PASE_Malloc(sizeof(PASE_MATRIX_PRIVATE));
  A->matrix_data          = matrix_data;
  A->ops                  = PASE_Matrix_data_operator_create(data_form);
  A->is_matrix_data_owner = PASE_NO;
  A->data_form            = data_form;

#if DEBUG_PASE_MATRIX
  if(NULL == A->ops->get_global_nrow) {
    PASE_Error(__FUNCT__": ops->get_global_nrow is not assigned.\n");
  }
  if(NULL == A->ops->get_global_ncol) {
    PASE_Error(__FUNCT__": ops->get_global_ncol is not assigned.\n");
  }
#endif

  A->ops->get_global_nrow(A->matrix_data, &A->global_nrow);
  A->ops->get_global_ncol(A->matrix_data, &A->global_ncol);
  return A;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_destroy"
/**
 * @brief 销毁矩阵
 */
void 
PASE_Matrix_destroy(PASE_MATRIX A)
{
  if(NULL == A) return;

#if DEBUG_PASE_MATRIX
  if((PASE_YES != A->is_matrix_data_owner) &&
     (PASE_NO  != A->is_matrix_data_owner)) {
    PASE_Error(__FUNCT__": Cannot decide whether the owner of matrix is.");
  }
  if(PASE_YES == A->is_matrix_data_owner) {
    if(NULL == A->ops->destroy) {
      PASE_Error(__FUNCT__": ops->destroy is not assigned.\n");
    }
  }
#endif

  if(PASE_YES == A->is_matrix_data_owner) {
    A->ops->destroy(A->matrix_data);
  }
  PASE_Matrix_data_operator_destroy(A->ops);
  
  PASE_Free(A);
  //A = NULL;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_transpose"
/**
 * @brief 矩阵转置
 */
PASE_MATRIX 
PASE_Matrix_transpose(PASE_MATRIX A)
{
#if DEBUG_PASE_MATRIX
  if(NULL == A) {
    PASE_Error(__FUNCT__": Matrix cannot be empty.\n");
  }
  if(NULL == A->ops) {
    PASE_Error(__FUNCT__": A->ops is not assigned.\n");
  }
  if(NULL == A->ops->transpose) {
    PASE_Error(__FUNCT__": A->ops->transpose is not assigned.\n");
  }
  if(NULL == A->ops->get_global_nrow) {
    PASE_Error(__FUNCT__": ops->get_global_nrow is not assigned.\n");
  }
  if(NULL == A->ops->get_global_ncol) {
    PASE_Error(__FUNCT__": ops->get_global_ncol is not assigned.\n");
  }
#endif

  void *matrix_data        = A->ops->transpose(A->matrix_data);
  PASE_MATRIX AT           = PASE_Matrix_assign(matrix_data, A->ops); 
  AT->is_matrix_data_owner = PASE_YES;
  AT->data_form            = A->data_form;
  AT->ops->get_global_nrow(AT->matrix_data, &AT->global_nrow);
  AT->ops->get_global_ncol(AT->matrix_data, &AT->global_ncol);
  return AT;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_copy"
/**
 * @brief 矩阵复制 (TODO 此函数有些问题, 未做测试, 暂时也不需要被调用)
 */
void 
PASE_Matrix_copy(PASE_MATRIX A, PASE_MATRIX B)
{
#if DEBUG_PASE_MATRIX
  if((NULL == A) || (NULL == B)) {
    PASE_Error(__FUNCT__": Neither the tow matrices can be empty.\n");
  }
  if(NULL != B->matrix_data) {
    PASE_Error(__FUNCT__": Cannot copy matrix if matrix data of dst is not NULL.\n");
  }
  if(B->data_form != A->data_form) {
    PASE_Error(__FUNCT__": Cannot copy a matrix to a different form.\n");
  }
  if(NULL == A->ops->copy) {
    PASE_Error(__FUNCT__": ops->copy is not assigned.\n");
  }
  if(NULL == B->ops->get_global_nrow) {
    PASE_Error(__FUNCT__": ops->get_global_nrow is not assigned.\n");
  }
  if(NULL == B->ops->get_global_ncol) {
    PASE_Error(__FUNCT__": ops->get_global_ncol is not assigned.\n");
  }
#endif

  A->ops->copy(A->matrix_data, B->matrix_data);

  B->is_matrix_data_owner = PASE_YES;
  B->ops->get_global_nrow(B->matrix_data, &B->global_nrow);
  B->ops->get_global_ncol(B->matrix_data, &B->global_ncol);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_multiply_matrix"
/**
 * @brief 矩阵与矩阵相乘 A * B
 */
PASE_MATRIX 
PASE_Matrix_multiply_matrix(PASE_MATRIX A, PASE_MATRIX B)
{
#if DEBUG_PASE_MATRIX
  if((NULL == A) || (NULL == B)) {
    PASE_Error(__FUNCT__": Neither the tow matrices can be empty.\n");
  }
  if(A->global_ncol != B->global_nrow) {
    PASE_Error(__FUNCT__": Matrix dimensions must be matched.\n");
  }
  if(NULL == A->ops->multiply_matrix_matrix) {
    PASE_Error(__FUNCT__": ops->multiply_matrix_matrix is not assigned.\n");
  }
  if(NULL == A->ops->get_global_nrow) {
    PASE_Error(__FUNCT__": ops->get_global_nrow is not assigned.\n");
  }
  if(NULL == A->ops->get_global_ncol) {
    PASE_Error(__FUNCT__": ops->get_global_ncol is not assigned.\n");
  }
#endif

  void *matrix_data = A->ops->multiply_matrix_matrix(A->matrix_data, B->matrix_data); 
  PASE_MATRIX C = PASE_Matrix_assign(matrix_data, A->ops); 
  C->is_matrix_data_owner = PASE_YES;
  C->data_form = A->data_form;
  C->ops->get_global_nrow(C->matrix_data, &C->global_nrow);
  C->ops->get_global_ncol(C->matrix_data, &C->global_ncol);
  return C;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_MatrixT_multiply_matrix"
/**
 * @brief 矩阵转置与矩阵相乘 A^T * B
 */
PASE_MATRIX 
PASE_MatrixT_multiply_matrix(PASE_MATRIX A, PASE_MATRIX B)
{
#if DEBUG_PASE_MATRIX
  if((NULL == A) || (NULL == B)) {
    PASE_Error(__FUNCT__": Neither the tow matrices can be empty.\n");
  }
  if(A->global_nrow != B->global_nrow) {
    PASE_Error(__FUNCT__": Matrix dimensions must be matched.\n");
  }
  if((NULL == A->ops->multiply_matrixT_matrix) &&
     ((NULL == A->ops->transpose) || (NULL == A->ops->multiply_matrix_matrix))) {
    PASE_Error(__FUNCT__": Neither ops->multiply_matrixT_matrix nor ops->transpose is assigned.\n");
  }
  if(NULL == A->ops->get_global_nrow) {
    PASE_Error(__FUNCT__": ops->get_global_nrow is not assigned.\n");
  }
  if(NULL == A->ops->get_global_ncol) {
    PASE_Error(__FUNCT__": ops->get_global_ncol is not assigned.\n");
  }
#endif

  PASE_MATRIX C = NULL;
  if(A->ops->multiply_matrixT_matrix) {
      void *matrix_data = A->ops->multiply_matrixT_matrix(A->matrix_data, B->matrix_data);
      C = PASE_Matrix_assign(matrix_data, A->ops);
      C->is_matrix_data_owner = PASE_YES;
      C->data_form            = A->data_form;
      C->ops->get_global_nrow(C->matrix_data, &C->global_nrow);
      C->ops->get_global_ncol(C->matrix_data, &C->global_ncol);
  } else {
      PASE_MATRIX tmp_AT = PASE_Matrix_transpose(A);
      C = PASE_Matrix_multiply_matrix(tmp_AT, B);
      PASE_Matrix_destroy(tmp_AT);
  }

  return C;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_multiply_vector"
/**
 * @brief 矩阵向量相乘 y = A * x (其中 y 不能与 x 相同)
 */
void 
PASE_Matrix_multiply_vector(PASE_MATRIX A, PASE_VECTOR x, PASE_VECTOR y)
{
#if DEBUG_PASE_MATRIX
  if((NULL == A) || (NULL == x) || (NULL == y)) {
    PASE_Error(__FUNCT__": Matrix and vectors cannot be empty.\n");
  }
  if((A->global_ncol != x->global_nrow) ||
     (A->global_nrow != y->global_nrow)) {
    PASE_Error(__FUNCT__": The dimensions of matrix and vector are not matched.\n");
  }
  if((NULL == A->ops->multiply_matrix_vector) &&
     (NULL == A->ops->multiply_matrix_vector_general)) {
    PASE_Error(__FUNCT__": Neither ops->multiply_matrix_vector nor "
                        "ops->multiply_matrix_vector_general is assigned.\n");
  }
  if(y == x) {
    PASE_Error(__FUNCT__": Vectors x and y cannot be the same.\n");
  }
#endif
  
  if(NULL != A->ops->multiply_matrix_vector) {
    A->ops->multiply_matrix_vector(A->matrix_data, x->vector_data, y->vector_data);
  } else {
    PASE_Matrix_multiply_vector_general(1.0, A, x, 0.0, y);
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_multiply_vector_general"
/**
 * @brief 矩阵向量相乘 y = a * A * x  + b * y (y 不能与 x 相同)
 */
void 
PASE_Matrix_multiply_vector_general(PASE_SCALAR a, PASE_MATRIX A, PASE_VECTOR x, 
                                    PASE_SCALAR b, PASE_VECTOR y)
{
#if DEBUG_PASE_MATRIX
  if((NULL == A) || (NULL == x) || (NULL == y)) {
    PASE_Error(__FUNCT__": Matrix and vectors cannot be empty.\n");
  }
  if((A->global_ncol != x->global_nrow) ||
     (A->global_nrow != y->global_nrow)) {
    PASE_Error(__FUNCT__": The dimensions of matrix and vector are not matched.\n");
  }
  if((NULL == A->ops->multiply_matrix_vector_general) &&
     (NULL == A->ops->multiply_matrix_vector)) {
    PASE_Error(__FUNCT__": Neither ops->multiply_matrix_vector_general nor "
                        "ops->multiply_matrix_vector is assigned.\n");
  }
  if(y == x) {
    PASE_Error(__FUNCT__": Vectors x and y cannot be the same.\n");
  }
#endif
  
  if(NULL != A->ops->multiply_matrix_vector_general) {
    A->ops->multiply_matrix_vector_general(a, A->matrix_data, x->vector_data, b, y->vector_data);
  } else {
    PASE_VECTOR tmp_Ax = PASE_Vector_create_by_vector(y);
    PASE_Matrix_multiply_vector(A, x, tmp_Ax);
    PASE_Vector_scale(b, y);
    PASE_Vector_axpy(a, tmp_Ax, y);
    PASE_Vector_destroy(tmp_Ax);
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_MatrixT_multiply_vector"
/**
 * @brief 矩阵转置与向量相乘 y = AT * x (其中 y 不能与 x 相同)
 */
void 
PASE_MatrixT_multiply_vector(PASE_MATRIX A, PASE_VECTOR x, PASE_VECTOR y)
{
#if DEBUG_PASE_MATRIX
  if((NULL == A) || (NULL == x) || (NULL == y)) {
    PASE_Error(__FUNCT__": Matrix and vectors cannot be empty.\n");
  }
  if((A->global_nrow != x->global_nrow) ||
     (A->global_ncol != y->global_nrow)) {
    PASE_Error(__FUNCT__": The dimensions of matrix and vector are not matched.\n");
  }
  if((NULL == A->ops->multiply_matrixT_vector) &&
     (NULL == A->ops->multiply_matrixT_vector_general)) {
    PASE_Error(__FUNCT__": Neither ops->multiply_matrixT_vector nor "
                        "ops->multiply_matrixT_vector_general is assigned.\n");
  }
  if(y == x) {
    PASE_Error(__FUNCT__": Vectors x and y cannot be the same.\n");
  }
#endif
  
  if(NULL != A->ops->multiply_matrixT_vector) {
    A->ops->multiply_matrixT_vector(A->matrix_data, x->vector_data, y->vector_data);
  } else {
    PASE_MatrixT_multiply_vector_general(1.0, A, x, 0.0, y);
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_MatrixT_multiply_vector_general"
/**
 * @brief 矩阵转置与向量相乘 y = a * AT * x  + b * y (其中 y 不能与 x 相同)
 */
void PASE_MatrixT_multiply_vector_general(PASE_SCALAR a, PASE_MATRIX A, PASE_VECTOR x, 
                                          PASE_SCALAR b, PASE_VECTOR y)
{
#if DEBUG_PASE_MATRIX
  if((NULL == A) || (NULL == x) || (NULL == y)) {
    PASE_Error(__FUNCT__": Matrix and vectors cannot be empty.\n");
  }
  if((A->global_nrow != x->global_nrow) ||
     (A->global_ncol != y->global_nrow)) {
    PASE_Error(__FUNCT__": The dimensions of matrix and vector are not matched.\n");
  }
  if((NULL == A->ops->multiply_matrixT_vector) &&
     (NULL == A->ops->multiply_matrixT_vector_general)) {
    PASE_Error(__FUNCT__": Neither ops->multiply_matrixT_vector nor "
                        "ops->multiply_matrixT_vector_general is assigned.\n");
  }
  if(y == x) {
    PASE_Error(__FUNCT__": Vectors x and y cannot be the same.\n");
  }
#endif
  
  if(NULL != A->ops->multiply_matrixT_vector_general) {
    A->ops->multiply_matrixT_vector_general(1.0, A->matrix_data, x->vector_data, 0.0, y->vector_data);
  } else {
    PASE_VECTOR tmp_ATx = PASE_Vector_create_by_vector(y);
    PASE_MatrixT_multiply_vector(A, x, tmp_ATx);
    PASE_Vector_scale(b, y);
    PASE_Vector_axpy(a, tmp_ATx, y);
    PASE_Vector_destroy(tmp_ATx);
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_get_global_nrow"
/**
 * @brief 获得矩阵全局行数
 */
void 
PASE_Matrix_get_global_nrow(PASE_MATRIX A, PASE_INT *nrow)
{
#if DEBUG_PASE_MATRIX
  if(NULL == A) {
    PASE_Error(__FUNCT__": Matrix cannot be empty.\n");
  }
  if(NULL == A->ops->get_global_nrow) {
    PASE_Error(__FUNCT__": ops->get_global_nrow is not assigned.\n");
  }
#endif
  A->ops->get_global_nrow(A->matrix_data, nrow);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_get_global_ncol"
/**
 * @brief 获得矩阵全局行数
 */
void 
PASE_Matrix_get_global_ncol(PASE_MATRIX A, PASE_INT *ncol)
{
#if DEBUG_PASE_MATRIX
  if(NULL == A) {
    PASE_Error(__FUNCT__": Matrix cannot be empty.\n");
  }
  if(NULL == A->ops->get_global_ncol) {
    PASE_Error(__FUNCT__": ops->get_global_ncol is not assigned.\n");
  }
#endif
  A->ops->get_global_ncol(A->matrix_data, ncol);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Matrix_get_mpi_comm"
/**
 * @brief 获得矩阵通信器
 */
MPI_Comm 
PASE_Matrix_get_mpi_comm(PASE_MATRIX A)
{
#if DEBUG_PASE_MATRIX
  if(NULL == A) {
    PASE_Error(__FUNCT__": Matrix cannot be empty.\n");
  }
  if(NULL == A->ops->get_mpi_comm) {
    PASE_Error(__FUNCT__": ops->get_mpi_comm is not assigned.\n");
  }
#endif
  return A->ops->get_mpi_comm(A->matrix_data);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_create_by_matrix_and_vector_data"
/**
 * @brief 由矩阵和向量运算集合创建新的向量
 */
PASE_VECTOR 
PASE_Vector_create_by_matrix_and_vector_data_operator(PASE_MATRIX A, PASE_VECTOR_DATA_OPERATOR ops)
{
#if DEBUG_PASE_MATRIX
  if(NULL == A) {
    PASE_Error(__FUNCT__": Matrix cannot be empty.\n");
  }
#endif

  PASE_VECTOR_DATA_OPERATOR  ops_new = ops;
  if(NULL == ops) {
    ops_new = PASE_Vector_data_operator_create(A->data_form);
  }

#if DEBUG_PASE_MATRIX
  if(NULL == ops_new->create_by_matrix) {
    PASE_Error(__FUNCT__": vector ops->create_by_matrix is not assigned.\n");
  }
#endif

  void        *vector_data = ops_new->create_by_matrix(A->matrix_data);
  PASE_VECTOR  x           = PASE_Vector_assign(vector_data, ops_new);
  x->is_vector_data_owner  = PASE_YES;
  x->data_form             = A->data_form;

  if(NULL == ops) {
    PASE_Vector_data_operator_destroy(ops_new);
  }
  
  return x;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_inner_product_general"
/**
 * @brief 广义向量内积 *prod = x^T * A * y
 */
void 
PASE_Vector_inner_product_general(PASE_VECTOR x, PASE_VECTOR y, PASE_MATRIX A, PASE_REAL *prod)
{
#if DEBUG_PASE_MATRIX
  if((NULL == A) || (NULL == x) || (NULL == y)) {
    PASE_Error(__FUNCT__": Matrix and vectors cannot be empty.\n");
  }
#endif

  PASE_VECTOR tmp_Ax = PASE_Vector_create_by_vector(x);
  PASE_Matrix_multiply_vector(A, x, tmp_Ax);
  PASE_Vector_inner_product(tmp_Ax, y, prod);
  PASE_Vector_destroy(tmp_Ax);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_inner_product_general_some"
/**
 * @brief 向量 x[start], ..., x[end] 全体计算 A 内积
 */
void 
PASE_Vector_inner_product_general_some(PASE_VECTOR *x, PASE_INT start, PASE_INT end, 
                                       PASE_MATRIX A, PASE_REAL **prod)
{
#if DEBUG_PASE_MATRIX
  if((NULL == A) || (NULL == x) || (NULL == prod)) {
    PASE_Error(__FUNCT__": Matrix and vectors and products cannot be empty.\n");
  }
#endif

  PASE_VECTOR tmp_Axi = PASE_Vector_create_by_vector(x[start]);

  PASE_INT i = 0;
  PASE_INT j = 0;
  for(i = start; i <= end; ++i) {
    PASE_Matrix_multiply_vector(A, x[i], tmp_Axi);
    for(j = start; j <= i; ++j) {
      PASE_Vector_inner_product(x[j], tmp_Axi, &prod[i-start][j-start]);
      prod[j-start][i-start] = prod[i-start][j-start];
    }
  }

  PASE_Vector_destroy(tmp_Axi);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_orthogonalize_general"
/**
 * @brief 向量 x[i] 与 x[start], ..., x[end] 在 A 内积下正交化
 */
void PASE_Vector_orthogonalize_general(PASE_VECTOR *x, 
                                       PASE_INT i, PASE_INT start, PASE_INT end, 
                                       PASE_MATRIX A)
{
#if DEBUG_PASE_MATRIX
  if((NULL == A) || (NULL == x)) {
    PASE_Error(__FUNCT__": Matrix and vectors cannot be empty.\n");
  }
  if((i >= start) && (i <= end)) {
    PASE_Error(__FUNCT__": index cannot locate in [%d, %d].\n", start, end);
  }
#endif

  PASE_VECTOR tmp_Axi = PASE_Vector_create_by_vector(x[i]);
  PASE_Matrix_multiply_vector(A, x[i], tmp_Axi);

  PASE_INT  j    = 0;
  PASE_REAL prod = 0.0;
  for(j = start; j <= end; ++j) {
    PASE_Vector_inner_product(x[j], tmp_Axi, &prod);
    PASE_Vector_axpy(-prod, x[j], x[i]);
  }
  PASE_Vector_inner_product(x[i], tmp_Axi, &prod);
  PASE_Vector_scale(1.0/sqrt(prod), x[i]);

  PASE_Vector_destroy(tmp_Axi);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_orthogonalize_general_some"
/**
 * @brief 向量 x[0], ..., x[num-1] 全体在 A 内积下正交化
 */
void 
PASE_Vector_orthogonalize_general_all(PASE_VECTOR *x, PASE_INT num, PASE_MATRIX A)
{
#if DEBUG_PASE_MATRIX
  if((NULL == A) || (NULL == x)) {
    PASE_Error(__FUNCT__": Matrix and vectors cannot be empty.\n");
  }
#endif

  PASE_INT j = 0;
  for(j = 0; j < num; ++j) {
    PASE_Vector_orthogonalize_general(x, j, 0, j-1, A);
  }
}
