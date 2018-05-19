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
PASE_AUX_VECTOR PASE_Aux_vector_create(PASE_VECTOR vec, PASE_SCALAR *block, PASE_INT block_size);

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
 * @brief 设置 aux_x 的 vec
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
 * @brief 设置辅助向量 [vec  ] 中 block 部分的元素为常数 val
 *                     [block]
 *
 * @param aux_x  输入/输出参数, 被赋值的辅助向量
 * @param val    输入参数, 赋予向量的常数值
 */
void PASE_Aux_vector_set_block_constant(PASE_AUX_VECTOR aux_x, PASE_SCALAR val);  

/**
 * @brief 设置辅助向量 [vec  ] 中 block 部分的元素为随机数
 *                     [block]
 *
 * @param aux_x  输入/输出参数, 被赋值的辅助向量
 * @param seed   输入参数, 产生随机数组的种子
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
 * @brief 向量 aux_x[start], ..., aux_x[end] 全体计算内积
 *
 * @param aux_x  输入参数
 * @param start  输入参数
 * @param end    输入参数
 * @param prod   输出参数, prod[i][j] = (aux_x[start+i], aux_x[start+j])
 */
void PASE_Aux_vector_inner_product_some(PASE_AUX_VECTOR *aux_x, PASE_INT start, PASE_INT end, PASE_REAL **prod);

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
 * @brief 向量 x[i] 与 x[start], ..., x[end] 正交化, i 不能属于 [start, end]
 *
 * @param aux_x  输入/输出参数
 * @param i      输入参数
 * @param start  输入参数
 * @param end    输入参数
 */
void PASE_Aux_vector_orthogonalize(PASE_AUX_VECTOR *aux_x, PASE_INT i, PASE_INT start, PASE_INT end);

/**
 * @brief 向量 x[0], ..., x[num-1] 全体正交化
 */
void PASE_Aux_vector_orthogonalize_all(PASE_AUX_VECTOR *aux_x, PASE_INT num);

/**
 * @brief 辅助向量线性组合: aux_y = \sum_i coef[i] * aux_x[i]
 *
 * @param aux_x   输入向量, 用于做线性组合的辅助向量组
 * @param num_nev 输入向量, 用于做线型组合的向量个数
 * @param coef    输入向量, 用于做线性组合的系数数组
 * @param aux_y   输出向量, 用于存储线性组合完毕得到的辅助向量
 */
void PASE_Multi_aux_vector_combination(PASE_AUX_VECTOR *aux_x, PASE_INT num_vec, PASE_SCALAR *coef, PASE_AUX_VECTOR aux_y);

/**
 * @brief 多重辅助向量线性组合: aux_y = aux_x * mat
 *
 * @param aux_x   输入向量, 用于做线性组合的辅助向量组
 * @param num_nev 输入向量, 用于做线型组合的向量个数
 * @param mat     输入向量, 二维数组, 用于做线性组合的系数, 其维数为 num_vec * num_mat
 * @param num_mat 输入向量, 需做线型组合得到的向量个数
 * @param aux_y   输出向量, 用于存储线性组合完毕得到的辅助向量组
 */
void PASE_Multi_aux_vector_by_matrix(PASE_AUX_VECTOR *aux_x, PASE_INT num_vec, PASE_SCALAR **mat, PASE_INT num_mat, PASE_AUX_VECTOR *aux_y);

#endif
