#include <stdio.h>
#include <math.h>
#include "pase_config.h"
#include "pase_vector.h"

#if PASE_USE_HYPRE
#include "pase_vector_hypre.h"
#endif

#define DEBUG_PASE_VECTOR 1

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_data_operator_assign"
/**
 * @brief 指定向量运算集合
 */
PASE_VECTOR_DATA_OPERATOR
PASE_Vector_data_operator_assign(void* (*create_by_vector)  (void *x),
                                 void* (*create_by_matrix)  (void *A),
                                 void  (*copy)              (void *x, void *y),
                                 void  (*destroy)           (void *x),
                                 void  (*set_constant_value)(void *x, PASE_SCALAR a),
                                 void  (*set_random_value)  (void *x, PASE_INT seed),
                                 void  (*inner_product)     (void *x, void *y, PASE_REAL *prod),
                                 void  (*axpy)              (PASE_SCALAR a, void *x, void *y),
                                 void  (*scale)             (PASE_SCALAR a, void *x),
                                 void  (*get_global_nrow)   (void *x, PASE_INT *nrow))
{
  PASE_VECTOR_DATA_OPERATOR ops;
  ops = (PASE_VECTOR_DATA_OPERATOR)PASE_Malloc(sizeof(PASE_VECTOR_DATA_OPERATOR_PRIVATE));

  ops->create_by_vector   = create_by_vector;
  ops->create_by_matrix   = create_by_matrix;
  ops->copy               = copy;
  ops->destroy            = destroy;
  ops->set_constant_value = set_constant_value;
  ops->set_random_value   = set_random_value;
  ops->inner_product      = inner_product;
  ops->axpy               = axpy;
  ops->scale              = scale;
  ops->get_global_nrow    = get_global_nrow;

  return ops;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_data_operator_create"
/**
 * @brief 依据指定格式生成向量运算集合
 */
PASE_VECTOR_DATA_OPERATOR
PASE_Vector_data_operator_create(PASE_INT data_form)
{
  PASE_VECTOR_DATA_OPERATOR ops = NULL;
#if PASE_USE_HYPRE
  if(PACKAGE_HYPRE == data_form) {
    ops = PASE_Vector_data_operator_assign(PASE_Vector_data_create_by_vector_hypre,
                                           PASE_Vector_data_create_by_matrix_hypre,
                                           PASE_Vector_data_copy_hypre,
                                           PASE_Vector_data_destroy_hypre,
                                           PASE_Vector_data_set_constant_value_hypre,
                                           PASE_Vector_data_set_random_value_hypre,
                                           PASE_Vector_data_inner_product_hypre,
                                           PASE_Vector_data_axpy_hypre,
                                           PASE_Vector_data_scale_hypre,
                                           PASE_Vector_data_get_global_nrow_hypre);
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
#define __FUNCT__ "PASE_Vector_data_operator_destroy"
/**
 * @brief 销毁向量运算集合
 */
void
PASE_Vector_data_operator_destroy(PASE_VECTOR_DATA_OPERATOR ops)
{
  PASE_Free(ops);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_assign"
/**
 * @brief 将向量数据与运算集合关联至向量
 */
PASE_VECTOR
PASE_Vector_assign(void *vector_data, PASE_VECTOR_DATA_OPERATOR ops)
{
#if DEBUG_PASE_VECTOR
  if(NULL == vector_data) {
    PASE_Error(__FUNCT__": Can not create PAES VECTOR without vector data.\n");
  }
  if(NULL == ops) {
    PASE_Error(__FUNCT__": Can not create PAES VECTOR without vector data operator.\n");
  }
#endif

  PASE_VECTOR x = (PASE_VECTOR)PASE_Malloc(sizeof(PASE_VECTOR_PRIVATE));
  x->ops = (PASE_VECTOR_DATA_OPERATOR)PASE_Malloc(sizeof(PASE_VECTOR_DATA_OPERATOR_PRIVATE));

  x->vector_data          = vector_data;
  x->is_vector_data_owner = PASE_NO;
  x->data_form            = PASE_USER;
  *(x->ops)               = *ops;
  x->ops->get_global_nrow(x->vector_data, &x->global_nrow);
  return x;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_create"
/**
 * @brief 依据指定格式生成向量
 */
PASE_VECTOR
PASE_Vector_create(void *vector_data, PASE_INT data_form)
{
#if DEBUG_PASE_VECTOR
  if(NULL == vector_data) {
    PASE_Error(__FUNCT__": vector_data cannot be NULL.\n");
  }
#endif

  PASE_VECTOR x = (PASE_VECTOR)PASE_Malloc(sizeof(PASE_VECTOR_PRIVATE));
  x->ops = PASE_Vector_data_operator_create(data_form);
  x->vector_data          = vector_data;
  x->is_vector_data_owner = PASE_NO;
  x->data_form            = data_form;
  x->ops->get_global_nrow(x->vector_data, &x->global_nrow);
  return x;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_destroy"
/**
 * @brief 销毁向量
 */
void
PASE_Vector_destroy(PASE_VECTOR x)
{
  if(NULL == x) return;

#if DEBUG_PASE_VECTOR
  if((PASE_YES != x->is_vector_data_owner) &&
     (PASE_NO  != x->is_vector_data_owner)) {
    PASE_Error(__FUNCT__": Cannot decide whether the owner of vector is.");
  }
#endif

  if(PASE_YES == x->is_vector_data_owner) {
    x->ops->destroy(x->vector_data);
  }
  PASE_Vector_data_operator_destroy(x->ops);
  PASE_Free(x);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_create_by_vector"
/**
 * @brief 由向量创建向量
 */
PASE_VECTOR
PASE_Vector_create_by_vector(PASE_VECTOR x)
{
  void        *vector_data = x->ops->create_by_vector(x->vector_data);
  PASE_VECTOR  y           = PASE_Vector_assign(vector_data, x->ops);
  y->is_vector_data_owner  = PASE_YES;
  y->data_form             = x->data_form;
  return y;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_copy"
/**
 * @brief 复制向量 x 到向量 y
 */
void
PASE_Vector_copy(PASE_VECTOR x, PASE_VECTOR y)
{
  if(x->global_nrow != y->global_nrow) {
    PASE_Error(__FUNCT__": Vector dimensions must be matched.\n");
  }
  x->ops->copy(x->vector_data, y->vector_data);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_set_constant_value"
/**
 * @brief 设置向量元素为常数
 */
void
PASE_Vector_set_constant_value(PASE_VECTOR x, PASE_SCALAR a)
{
  x->ops->set_constant_value(x->vector_data, a);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_set_random_value"
/**
 * @brief 设置向量元素为随机值
 */
void
PASE_Vector_set_random_value(PASE_VECTOR x, PASE_INT seed)
{
  x->ops->set_random_value(x->vector_data, seed);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_axpy"
/**
 * @brief 向量校正 y = a * x + y
 */
void
PASE_Vector_axpy(PASE_SCALAR a, PASE_VECTOR x, PASE_VECTOR y)
{
  x->ops->axpy(a, x->vector_data, y->vector_data);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_scale"
/**
 * @brief 向量数乘
 */
void
PASE_Vector_scale(PASE_SCALAR a, PASE_VECTOR x)
{
  x->ops->scale(a, x->vector_data);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_inner_product"
/**
 * @brief 向量内积
 */
void
PASE_Vector_inner_product(PASE_VECTOR x, PASE_VECTOR y, PASE_REAL *prod)
{
  x->ops->inner_product(x->vector_data, y->vector_data, prod);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_inner_product_some"
/**
 * @brief 向量 x[start], ..., x[end] 全体内积
 */
void 
PASE_Vector_inner_product_some(PASE_VECTOR *x, PASE_INT start, PASE_INT end, PASE_REAL **prod)
{
#if DEBUG_PASE_VECTOR
  if((NULL == x) || (NULL == prod)) {
    PASE_Error(__FUNCT__": Vectors and products cannot be empty.\n");
  }
#endif

  PASE_INT i = 0;
  PASE_INT j = 0;
  for(i = start; i <= end; ++i) {
    for(j = start; j <= i; ++j) {
      PASE_Vector_inner_product(x[j], x[i], &prod[i-start][j-start]);
    }
  }

  for(i = start; i <= end; ++i) {
    for(j = i + 1; j <= end; ++j) {
      prod[i-start][j-start] = prod[j-start][i-start];
    }
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_orthogonalize"
/**
 * @brief 向量 x[i] 与 x[start], ..., x[end] 正交化
 */
void PASE_Vector_orthogonalize(PASE_VECTOR *x, PASE_INT i, PASE_INT start, PASE_INT end)
{
#if DEBUG_PASE_VECTOR
  if((i >= start) && (i <= end)) {
    PASE_Error(__FUNCT__": index cannot locate in [%d, %d].\n", start, end);
  }
#endif

  PASE_INT  j    = 0;
  PASE_REAL prod = 0.0;
  for(j = start; j <= end; ++j) {
    PASE_Vector_inner_product(x[j], x[i], &prod);
    PASE_Vector_axpy(-prod, x[j], x[i]);
  }
  PASE_Vector_inner_product(x[i], x[i], &prod);
  PASE_Vector_scale(1.0/sqrt(prod), x[i]);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Vector_orthogonalize_all"
/**
 * @brief 向量 x[0], ..., x[num-1] 全体正交化
 */
void PASE_Vector_orthogonalize_all(PASE_VECTOR *x, PASE_INT num)
{
  PASE_INT j = 0;
  for(j = 0; j < num; ++j) {
    PASE_Vector_orthogonalize(x, j, 0, j-1);
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multi_vector_combination"
/**
 * @brief 向量线性组合: y = \sum_i coef[i] * x[i]
 *
 * @param x       输入向量, 用于做线性组合的向量组
 * @param num_nev 输入向量, 用于做线型组合的向量个数
 * @param coef    输入向量, 用于做线性组合的系数数组
 * @param y       输出向量, 用于存储线性组合完毕得到的向量
 */
void
PASE_Multi_vector_combination(PASE_VECTOR *x, PASE_INT num_vec, PASE_SCALAR *coef, PASE_VECTOR y)
{
#if DEBUG_PASE_VECTOR
  if(NULL == x) {
    PASE_Error(__FUNCT__": Cannot compute the linear combination of an empty PASE VECTOR set.\n");
  }
  if(NULL == coef) {
    PASE_Error(__FUNCT__": Cannot compute the linear combination of an empty coefficient set.\n");
  }
  if(NULL == y) {
    PASE_Error(__FUNCT__": Cannot store the linear combination with an empty PASE VECTOR.\n");
  }
#endif

  PASE_INT j = 0;
  PASE_Vector_set_constant_value(y, 0.0);
  for(j = 0; j < num_vec; j++) {
    PASE_Vector_axpy(coef[j], x[j], y);
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multi_vector_by_matrix"
/**
 * @brief 多重向量线性组合: y = x * mat
 *
 * @param x       输入向量, 用于做线性组合的向量组
 * @param num_nev 输入向量, 用于做线型组合的向量个数
 * @param mat     输入向量, 二维数组, 用于做线性组合的系数, 其维数为 num_vec * num_mat
 * @param num_mat 输入向量, 需做线型组合得到的向量个数
 * @param y       输出向量, 用于存储线性组合完毕得到的向量组
 */
void
PASE_Multi_vector_by_matrix(PASE_VECTOR *x, PASE_INT num_vec, PASE_SCALAR **mat, PASE_INT num_mat, PASE_VECTOR *y)
{
#if DEBUG_PASE_VECTOR
  if(NULL == x) {
    PASE_Error(__FUNCT__": Cannot compute the linear combinations of an empty PASE VECTOR set.\n");
  }
  if(NULL == mat) {
    PASE_Error(__FUNCT__": Cannot compute the linear combinations of an empty coefficient set.\n");
  }
  if(NULL == y) {
    PASE_Error(__FUNCT__": Cannot store the linear combinations with an empty PASE VECTOR set.\n");
  }
#endif

  PASE_INT i = 0;
  for(i = 0; i < num_mat; i++) {
    PASE_Multi_vector_combination(x, num_vec, mat[i], y[i]);
  }
}
