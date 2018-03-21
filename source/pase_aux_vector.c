#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include <math.h>
#include "pase_config.h"
#include "pase_vector.h"
#include "pase_aux_vector.h"
#include "pase_aux_matrix.h"

#if PASE_USE_HYPRE
#include "_hypre_parcsr_mv.h"
#endif

#define DEBUG_PASE_AUX_VECTOR 1

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_create"
/**
 * @brief 创建 PASE_AUX_VECTOR
 */
PASE_AUX_VECTOR
PASE_Aux_vector_create(PASE_VECTOR vec, PASE_SCALAR *block, PASE_INT block_size)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == vec || NULL == block) {
    PASE_Error(__FUNCT__": Cannot create a new PASE AUX VECTOR without vec nor block.\n");
  }
  if(0 >= block_size) {
    PASE_Error(__FUNCT__": Cannot create a new PASE AUX VECTOR with a nonpositive block size %d.\n", block_size);
  }
#endif

  PASE_AUX_VECTOR aux_x = (PASE_AUX_VECTOR)PASE_Malloc(sizeof(PASE_AUX_VECTOR_PRIVATE));
  aux_x->vec            = vec; 
  aux_x->is_vec_owner   = PASE_NO;
  aux_x->block          = block;
  aux_x->block_size     = block_size;
  return aux_x;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_create_by_aux_vector"
/**
 * @brief 根据给定的辅助向量, 创建一个新的辅助向量
 *
 * @param aux_x 输入参数, 给定的辅助向量
 *
 * @return PASE_AUX_VECTOR
 */
PASE_AUX_VECTOR 
PASE_Aux_vector_create_by_aux_vector(PASE_AUX_VECTOR aux_x)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": Cannot create a new PASE AUX VECTOR without a sample PASE AUX VECTOR.\n");
  }
#endif

  PASE_VECTOR  vec        = PASE_Vector_create_by_vector(aux_x->vec);
  PASE_INT     block_size = aux_x->block_size;
  PASE_SCALAR *block      = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
  memset(block, 0, block_size*sizeof(PASE_SCALAR));

  PASE_AUX_VECTOR aux_y   = PASE_Aux_vector_create(vec, block, block_size);
  aux_y->is_vec_owner     = PASE_YES;
  return aux_y;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_destroy"
/**
 * @brief 销毁辅助向量
 */
void 
PASE_Aux_vector_destroy(PASE_AUX_VECTOR aux_x)
{
  if(NULL == aux_x) return;

#if DEBUG_PASE_MATRIX
  if((PASE_YES != aux_x->is_vec_owner) &&
     (PASE_NO  != aux_x->is_vec_owner)) {
    PASE_Error(__FUNCT__": Cannot decide whether the owner of vec is.");
  }
#endif

  if(NULL != aux_x->vec && PASE_YES == aux_x->is_vec_owner) {
    PASE_Vector_destroy(aux_x->vec);
  }
  if(NULL != aux_x->block) {
    PASE_Free(aux_x->block);
  }
  PASE_Free(aux_x);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_copy"
/**
 * @brief 复制辅助向量 aux_x 的数据到辅助向量 aux_y
 *
 * @param aux_x  输入参数, 被复制的向量
 * @param aux_y  输入/输出参数, 复制得到的向量
 */
void
PASE_Aux_vector_copy(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x || NULL == aux_y) {
    PASE_Error(__FUNCT__": Neither the two PASE AUX VECTORs can be empty.\n");
  }
#endif
  PASE_Vector_copy(aux_x->vec, aux_y->vec);
  memcpy(aux_y->block, aux_x->block, aux_x->block_size*sizeof(PASE_SCALAR));
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_set_vec"
/**
 * @brief 设置 aux_x 的 vec
 */
void 
PASE_Aux_vector_set_vec(PASE_AUX_VECTOR aux_x, PASE_VECTOR vec)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": PASE AUX VECTOR being set to cannot be NULL!\n");
  }
  if(NULL == vec) {
    PASE_Error(__FUNCT__": The PASE VECTOR being set to PASE AUX VECTOR is NULL!\n");
  }
#endif

  if(NULL != aux_x->vec && PASE_YES == aux_x->is_vec_owner) {
    PASE_Vector_destroy(aux_x->vec);
  }
  aux_x->vec          = vec;
  aux_x->is_vec_owner = PASE_NO;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_set_constant_value"
/**
 * @brief 设置辅助向量的元素为常数 val
 *
 * @param aux_x  输入/输出参数, 被赋值的辅助向量
 * @param val    输入参数, 赋予向量的常数值
 */
void 
PASE_Aux_vector_set_constant_value(PASE_AUX_VECTOR aux_x, PASE_SCALAR val)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": PASE AUX MATRIX being set constant values can not be NULL.\n");
  }
#endif

  PASE_Vector_set_constant_value(aux_x->vec, val);
  PASE_Aux_vector_set_block_constant(aux_x, val);
} 

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_set_random_value"
/**
 * @brief 设置辅助向量的元素为随机值
 *
 * @param aux_x  输入/输出参数, 被赋值的辅助向量
 * @param seed   输入参数, 产生随机数组的种子
 */
void 
PASE_Aux_vector_set_random_value(PASE_AUX_VECTOR aux_x, PASE_INT seed)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": PASE AUX VECTOR being set random values cannot be NULL.\n");
  }
#endif

  PASE_Vector_set_random_value(aux_x->vec, seed);
  PASE_Aux_vector_set_block_random(aux_x, seed);
} 

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_set_block_constant"
/**
 * @brief 设置辅助向量 [vec  ] 中 block 部分的元素为常数 val
 *                     [block]
 *
 * @param aux_x  输入/输出参数, 被赋值的辅助向量
 * @param val    输入参数, 赋予向量的常数值
 */
void 
PASE_Aux_vector_set_block_constant(PASE_AUX_VECTOR aux_x, PASE_SCALAR val)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": PASE AUX VECTOR being set constant block cannot be NULL.\n");
  }
#endif

  PASE_INT i = 0;
  for(i = 0; i < aux_x->block_size; ++i) {
    aux_x->block[i] = val;
  }
} 

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_set_block_random"
/**
 * @brief 设置辅助向量 [vec  ] 中 block 部分的元素为随机数
 *                     [block]
 *
 * @param aux_x  输入/输出参数, 被赋值的辅助向量
 * @param seed   输入参数, 产生随机数组的种子
 */
void 
PASE_Aux_vector_set_block_random(PASE_AUX_VECTOR aux_x, PASE_INT seed)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": PASE AUX VECTOR being set random block cannot be NULL.\n");
  }
#endif

  PASE_INT i = 0;
  srand(seed);
  for(i=0; i<aux_x->block_size; ++i) {
    aux_x->block[i] = ((double)rand())/2147483647;
  }
} 

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_inner_product"
/**
 * @brief 辅助向量 aux_x 和辅助向量 aux_y 的内积计算 *prod = (aux_x, aux_y)_2 = aux_x^T *aux_y
 *
 * @param aux_x  输入参数
 * @param aux_y  输入参数
 * @param prod   输出参数
 */
void 
PASE_Aux_vector_inner_product(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y, PASE_REAL *prod)
{
#if PASE_AUX_VECTOR
  if(NULL == aux_x || NULL == aux_y || NULL == prod) {
    PASE_Error(__FUNCT__": Neither vectors nor product can be empty.\n");
  }
#endif

#if 0
  PASE_Vector_inner_product(aux_x->vec, aux_y->vec, prod);

  PASE_INT i = 0;
  for(i = 0; i < aux_x->block_size; ++i) {
    *prod += aux_x->block[i] * aux_y->block[i]; 
  }
#else
  MPI_Status  status;
  MPI_Request request;
  *prod = hypre_SeqVectorInnerProd(hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_x->vec->vector_data)), hypre_ParVectorLocalVector((HYPRE_ParVector)(aux_y->vec->vector_data)));
  MPI_Iallreduce(MPI_IN_PLACE, prod, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &request);

  PASE_INT  i   = 0;
  PASE_REAL tmp = 0.0;
  for(i = 0; i < aux_x->block_size; ++i) {
    tmp += aux_x->block[i] * aux_y->block[i]; 
  }
  MPI_Wait(&request, &status);
  *prod += tmp;
#endif
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_inner_product_some"
/**
 * @brief 向量 aux_x[start], ..., aux_x[end] 全体计算内积
 *
 * @param aux_x  输入参数
 * @param start  输入参数
 * @param end    输入参数
 * @param prod   输出参数, prod[i][j] = (aux_x[start+i], aux_x[start+j])
 */
void 
PASE_Aux_vector_inner_product_some(PASE_AUX_VECTOR *aux_x, PASE_INT start, PASE_INT end, PASE_REAL **prod)
{
#if DEBUG_PASE_AUX_VECTOR
  if((NULL == aux_x) || (NULL == prod)) {
    PASE_Error(__FUNCT__": Neither vectors and products can be empty.\n");
  }
#endif

  PASE_INT i = 0;
  PASE_INT j = 0;
  for(i = start; i <= end; ++i) {
    for(j = start; j <= i; ++j) {
      PASE_Aux_vector_inner_product(aux_x[j], aux_x[i], &prod[i-start][j-start]);
      prod[j-start][i-start] = prod[i-start][j-start];
    }
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_norm"
/**
 * @brief 辅助向量 aux_x 的 2-范数的计算
 *
 * @param aux_x  输入参数 
 * @param norm   输出参数
 */
void
PASE_Aux_vector_norm(PASE_AUX_VECTOR aux_x, PASE_REAL *norm)
{
#if DEBUG_PASE_AUX_VECTOR
  if((NULL == aux_x) || (NULL == norm)) {
    PASE_Error(__FUNCT__": Neither vectors and norm can be empty.\n");
  }
#endif

  PASE_Aux_vector_inner_product(aux_x, aux_x, norm);
  *norm = sqrt(*norm);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_axpy"
/**
 * @brief 辅助向量数乘 aux_x = a * aux_x
 *
 * @param a      输入参数, 数乘系数
 * @param aux_x  输入/输出参数
 */
void 
PASE_Aux_vector_axpy(PASE_SCALAR a, PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y)
{
#if DEBUG_PASE_AUX_VECTOR
    if(NULL == aux_x || NULL == aux_y) {
	PASE_Error(__FUNCT__": Neither the two matrix can be empty!\n");
    }
  if(aux_y == aux_x) {
    PASE_Error(__FUNCT__": Vectors aux_x and aux_y cannot be the same.\n");
  }
#endif

  PASE_Vector_axpy(a, aux_x->vec, aux_y->vec);

  PASE_INT i = 0;
  for(i = 0; i < aux_x->block_size; ++i) {
    aux_y->block[i] += a * aux_x->block[i];
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_scale"
/**
 * @brief 辅助向量数乘 aux_x = a * aux_x
 *
 * @param a      输入参数, 数乘系数
 * @param aux_x  输入/输出参数
 */
void 
PASE_Aux_vector_scale(PASE_SCALAR a, PASE_AUX_VECTOR aux_x)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": Cannot scale an empty PASE AUX VECTOR.\n");
  }
#endif 

  PASE_Vector_scale(a, aux_x->vec);

  PASE_INT i = 0;
  for(i = 0; i < aux_x->block_size; ++i) {
    aux_x->block[i] *= a;
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_orthogonalize"
/**
 * @brief 向量 x[i] 与 x[start], ..., x[end] 正交化, i 不能属于 [start, end]
 *
 * @param aux_x  输入/输出参数
 * @param i      输入参数
 * @param start  输入参数
 * @param end    输入参数
 */
void
PASE_Aux_vector_orthogonalize(PASE_AUX_VECTOR *aux_x, PASE_INT i, PASE_INT start, PASE_INT end)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": Cannot orthogonalize an empty PASE AUX VECTOR set.\n");
  }
  if((i >= start) && (i <= end)) {
    PASE_Error(__FUNCT__": index %d cannot locate in [%d, %d].\n", i, start, end);
  }
#endif

  PASE_INT  j    = 0;
  PASE_REAL prod = 0.0;
  PASE_REAL norm = 0.0;
  for(j = start; j <= end; ++j) {
    PASE_Aux_vector_inner_product(aux_x[j], aux_x[i], &prod);
    PASE_Aux_vector_axpy(-prod, aux_x[j], aux_x[i]);
  }
  PASE_Aux_vector_norm(aux_x[i], &norm);
  PASE_Aux_vector_scale(1.0/norm, aux_x[i]);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_orthogonalize_all"
/**
 * @brief 向量 x[0], ..., x[num-1] 全体正交化
 *
 * @param aux_x  输入/输出参数
 * @param num    输入参数
 */
void
PASE_Aux_vector_orthogonalize_all(PASE_AUX_VECTOR *aux_x, PASE_INT num)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": Cannot orthogonalize an empty PASE AUX VECTOR set.\n");
  }
#endif

  PASE_INT j = 0;
  for(j = 0; j < num; ++j) {
    PASE_Aux_vector_orthogonalize(aux_x, j, 0, j-1);
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multi_aux_vector_combination"
/**
 * @brief 辅助向量线性组合: aux_y = \sum_i coef[i] * aux_x[i]
 *
 * @param aux_x   输入向量, 用于做线性组合的辅助向量组
 * @param num_nev 输入向量, 用于做线型组合的向量个数
 * @param coef    输入向量, 用于做线性组合的系数数组
 * @param aux_y   输出向量, 用于存储线性组合完毕得到的辅助向量
 */
void
PASE_Multi_aux_vector_combination(PASE_AUX_VECTOR *aux_x, PASE_INT num_vec, PASE_SCALAR *coef, PASE_AUX_VECTOR aux_y)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": Cannot compute the linear combination of an empty PASE AUX VECTOR set.\n");
  }
  if(NULL == coef) {
    PASE_Error(__FUNCT__": Cannot compute the linear combination of an empty coefficient set.\n");
  }
  if(NULL == aux_y) {
    PASE_Error(__FUNCT__": Cannot store the linear combination with an empty PASE AUX VECTOR.\n");
  }
#endif

  PASE_INT j = 0;
  PASE_Aux_vector_set_constant_value(aux_y, 0.0);
  for(j = 0; j < num_vec; ++j) {
    PASE_Aux_vector_axpy(coef[j], aux_x[j], aux_y);
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multi_aux_vector_by_matrix"
/**
 * @brief 多重辅助向量线性组合: aux_y = aux_x * mat
 *
 * @param aux_x   输入向量, 用于做线性组合的辅助向量组
 * @param num_nev 输入向量, 用于做线型组合的向量个数
 * @param mat     输入向量, 二维数组, 用于做线性组合的系数, 其维数为 num_vec * num_mat
 * @param num_mat 输入向量, 需做线型组合得到的向量个数
 * @param aux_y   输出向量, 用于存储线性组合完毕得到的辅助向量组
 */
void
PASE_Multi_aux_vector_by_matrix(PASE_AUX_VECTOR *aux_x, PASE_INT num_vec, PASE_SCALAR **mat, PASE_INT num_mat, PASE_AUX_VECTOR *aux_y)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": Cannot compute the linear combinations of an empty PASE AUX VECTOR set.\n");
  }
  if(NULL == mat) {
    PASE_Error(__FUNCT__": Cannot compute the linear combinations of an empty coefficient set.\n");
  }
  if(NULL == aux_y) {
    PASE_Error(__FUNCT__": Cannot store the linear combinations with an empty PASE AUX VECTOR set.\n");
  }
#endif

  PASE_INT i = 0;
  for(i = 0; i < num_mat; ++i) {
    PASE_Multi_aux_vector_combination(aux_x, num_vec, mat[i], aux_y[i]);
  }
}

