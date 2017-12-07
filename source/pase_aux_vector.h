#ifndef __PASE_AUX_VECTOR_H__
#define __PASE_AUX_VECTOR_H__

#include "pase_vector.h"

/*
 * aux vector = [vec  ]
 *              [block]
 */
typedef struct PASE_AUX_VECTOR_PRIVATE_ {
  PASE_VECTOR  vec;
  PASE_SCALAR *block;
  PASE_INT     block_size;   //辅助空间的维数
  PASE_INT     is_vec_owner; //是否为 vec 属主
} PASE_AUX_VECTOR_PRIVATE;
typedef PASE_AUX_VECTOR_PRIVATE * PASE_AUX_VECTOR;

//typedef struct PASE_MULTI_AUX_VECTOR_PRIVATE_ {
//  PASE_INT size;
//  PASE_AUX_VECTOR **aux_vector;
//} PASE_MULTI_AUX_VECTOR_PRIVATE;
//typedef PASE_MULTI_AUX_VECTOR_PRIVATE * PASE_MULTI_AUX_VECTOR;


/**
 * @brief 创建 PASE_AUX_VECTOR
 */
PASE_AUX_VECTOR PASE_Aux_Vector_create(PASE_VECTOR vec, PASE_SCALAR *block, PASE_INT block_size);

/**
 * @brief 根据给定的辅助向量, 创建一个新的辅助向量
 *
 * @param aux_x 输入参数, 给定的辅助向量
 *
 * @return PASE_AUX_VECTOR
 */
PASE_AUX_VECTOR PASE_Aux_vector_create_by_aux_vector(PASE_AUX_VECTOR aux_x);

/**
 * @brief 销毁辅助向量
 */
void PASE_Aux_vector_destroy(PASE_AUX_VECTOR aux_x);

/**
 * @brief 复制辅助向量 aux_x 的数据到辅助向量 aux_y
 *
 * @param aux_x  输入参数, 被复制的向量
 * @param aux_y  输入/输出参数, 复制得到的向量
 */
void PASE_Aux_vector_copy(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y);

/**
 * @brief 
 */
void PASE_Aux_vector_set_vec(PASE_AUX_VECTOR aux_x, PASE_VECTOR vec);

/**
 * @brief 设置辅助向量的元素为常数 val
 *
 * @param aux_x  输入/输出参数, 被赋值的辅助向量
 * @param val    输入参数, 赋予向量的常数值
 */
void PASE_Aux_vector_set_constant_value(PASE_AUX_VECTOR aux_x, PASE_SCALAR val);  

/**
 * @brief 设置辅助向量的元素为随机值
 *
 * @param aux_x  输入/输出参数, 被赋值的辅助向量
 * @param seed   输入参数, 产生随机数组的种子
 */
void PASE_Aux_vector_set_random_value(PASE_AUX_VECTOR aux_x, PASE_INT seed);  

/**
 * @brief
 */
void PASE_Aux_vector_set_block_constant(PASE_AUX_VECTOR aux_x, PASE_SCALAR val);  

/**
 * @brief
 */
void PASE_Aux_vector_set_block_random(PASE_AUX_VECTOR aux_x, PASE_INT seed);  

/**
 * @brief 辅助向量 aux_x 和辅助向量 aux_y 的内积计算 *prod = (aux_x, aux_y)_2 = aux_x^T *aux_y
 *
 * @param aux_x  输入参数
 * @param aux_y  输入参数
 * @param prod   输出参数
 */
void PASE_Aux_vector_inner_product(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y, PASE_REAL *prod);


/**
 * @brief 辅助向量 aux_x 的 2-范数的计算
 *
 * @param aux_x  输入参数 
 * @param norm   输出参数
 */
void PASE_Aux_vector_norm(PASE_AUX_VECTOR aux_x, PASE_REAL *norm);

/**
 * @brief 辅助向量校正 aux_y = a * aux_x + aux_y
 *
 * @param a      输入参数, 数乘系数
 * @param aux_x  输入参数
 * @param aux_y  输入/输出参数
 */
void PASE_Aux_vector_axpy(PASE_SCALAR a, PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y);

/**
 * @brief 辅助向量数乘 aux_x = a * aux_x
 *
 * @param a      输入参数, 数乘系数
 * @param aux_x  输入/输出参数
 */
void PASE_Aux_vector_scale(PASE_SCALAR a, PASE_AUX_VECTOR aux_x);

/**
 * @brief
 */
void PASE_Aux_vector_orthogonalize(PASE_AUX_VECTOR *aux_x, PASE_INT num);

/**
 * @brief
 */
void PASE_Multi_aux_vector_combination(PASE_AUX_VECTOR *aux_x, PASE_INT num_vec, PASE_SCALAR *coef, PASE_AUX_VECTOR aux_y);

/**
 * @brief
 */
void PASE_Multi_aux_vector_by_matrix(PASE_AUX_VECTOR *aux_x, PASE_INT num_vec, PASE_SCALAR **mat, PASE_INT num_mat, PASE_AUX_VECTOR *aux_y);

#endif
