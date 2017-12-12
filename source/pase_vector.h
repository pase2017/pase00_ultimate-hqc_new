#ifndef __PASE_VECTOR_H__
#define __PASE_VECTOR_H__

#include "pase_config.h"

/*********************************************************************
 * PASE_VECTOR 将外部软件包实现的向量类型封装为其内部的 vector_data, *
 * 对应的向量运算封装为其内部的 ops,                                 *
 * 从而在抽象层面上统一对真正的向量进行操作.                         *
 *********************************************************************/

/**
 * 对用户来说, 在实现下列运算时,
 * 所操作的都是其可直接感知的向量/矩阵类型 (虽然被转换为 void * 类型).
 */
typedef struct PASE_VECTOR_DATA_OPERATOR_PRIVATE_ {
  /**
   * @brief 由向量创建向量
   */
  void * (*create_by_vector)(void *x);

  /**
   * @brief 由矩阵创建向量
   */
  void * (*create_by_matrix)(void *A);

  /**
   * @brief 复制向量
   * @param src  输入参数, 被复制的向量
   * @param dst  输出参数, 复制得到的向量
   */
  void (*copy)(void *src, void *dst);

  /**
   * @brief 销毁向量数据
   */
  void (*destroy)(void *x);

  /**
   * @brief 设置向量元素为常数
   * @param x  输入/输出参数, 元素被设置为常数的向量
   * @param a  输入参数, 常数
   */
  void (*set_constant_value)(void *x, PASE_SCALAR a);

  /**
   * @brief 设置向量元素为随机数
   * @param x    输入/输出参数, 元素被设置为随机数的向量
   * @param seed 输入参数，随机数种子
   */
  void (*set_random_value)(void *x, PASE_INT seed);

  /**
   * @brief 计算向量内积
   * @param x    输入参数, 待计算内积的向量之一
   * @param y    输入参数, 待计算内积的向量之一
   * @param prod 输出参数, PASE_REAL 指针, 指向计算所得内积
   */
  void (*inner_product)(void *x, void *y, PASE_REAL *prod);

  /**
   * @brief 向量校正 y = a * x + y
   * @param a  输入参数, 校正参数
   * @param x  输入参数, 校正参数
   * @param y  输入/输出参数, 待校正向量
   */
  void (*axpy)(PASE_SCALAR a, void *x, void *y);

  /**
   * @brief 向量数乘 x = a * x
   * @param a  输入参数, 数乘因子
   * @param x  输入/输出参数, 数乘向量
   */
  void (*scale)(PASE_SCALAR a, void *x);

  /**
   * @brief 获得向量全局长度
   * @param x    输入参数, 待获取全局长度的向量
   * @param nrow 输出参数, 向量全局长度
   */
  void (*get_global_nrow)(void *x, PASE_INT *nrow);
} PASE_VECTOR_DATA_OPERATOR_PRIVATE;

typedef PASE_VECTOR_DATA_OPERATOR_PRIVATE * PASE_VECTOR_DATA_OPERATOR;

typedef struct PASE_VECTOR_PRIVATE_ {
    void                      *vector_data; // 向量数据
    PASE_INT                   global_nrow; // 向量全局长度
    PASE_VECTOR_DATA_OPERATOR  ops;         // 向量运算集合
    PASE_INT                   is_vector_data_owner; // 是否为向量数据属主
    PASE_INT                   data_form;   // 向量数据格式
} PASE_VECTOR_PRIVATE;

typedef PASE_VECTOR_PRIVATE * PASE_VECTOR;

//#include "pase_matrix.h"

/**
 * @brief 指定向量运算集合
 *
 * @param create_by_vector    输入参数, 函数指针, 由向量生成向量
 * @param create_by_matrix    输入参数, 函数指针, 由矩阵生成向量
 * @param copy                输入参数, 函数指针, 向量数据复制
 * @param destroy             输入参数, 函数指针, 销毁向量数据
 * @param set_constant_value  输入参数, 函数指针, 设置向量元素为常数
 * @param set_random_value    输入参数, 函数指针, 设置向量元素为随机数
 * @param inner_product       输入参数, 函数指针, 向量内积
 * @param axpy                输入参数, 函数指针, 向量校正
 * @param scale               输入参数, 函数指针, 向量数乘
 * @param get_global_nrow     输入参数, 函数指针, 获取向量全局行数
 *
 * @return PASE_VECTOR_DATA_OPERATOR
 */
PASE_VECTOR_DATA_OPERATOR PASE_Vector_data_operator_assign
        (void* (*create_by_vector)  (void *x),
         void* (*create_by_matrix)  (void *A),
         void  (*copy)              (void *x, void *y),
         void  (*destroy)           (void *x),
         void  (*set_constant_value)(void *x, PASE_SCALAR a),
         void  (*set_random_value)  (void *x, PASE_INT seed),
         void  (*inner_product)     (void *x, void *y, PASE_REAL *prod),
         void  (*axpy)              (PASE_SCALAR a, void *x, void *y),
         void  (*scale)             (PASE_SCALAR a, void *x),
         void  (*get_global_nrow)   (void *x, PASE_INT *nrow));

/**
 * @brief 依据指定格式生成向量运算集合, 需要在编译时指定可用的格式
 *
 * @param data_form  输入参数, 数据格式类型
 *                   1: PACKAGE_HYPRE
 *                   2: PACKAGE_JXPAMG
 *
 * @return PASE_VECTOR_DATA_OPERATOR
 */
PASE_VECTOR_DATA_OPERATOR PASE_Vector_data_operator_create(PASE_INT data_form);

/**
 * @brief 销毁向量运算集合
 */
void PASE_Vector_data_operator_destroy(PASE_VECTOR_DATA_OPERATOR ops);


/**
 * @brief 将向量数据与运算集合关联至向量
 *
 * @param vector_data  输入参数, void * 指针, 指向向量数据
 * @param ops          输入参数, 向量运算集合
 *
 * @return PASE_VECTOR
 */
PASE_VECTOR PASE_Vector_assign(void *vector_data, PASE_VECTOR_DATA_OPERATOR ops);

/**
 * @brief 依据指定格式生成向量
 *
 * @param vector_data  输入参数, void * 指针, 指向向量数据
 * @param data_form    输入参数, 数据格式类型
 *
 * @return PASE_VECTOR
 */
PASE_VECTOR PASE_Vector_create(void *vector_data, PASE_INT data_form);

/**
 * @brief 销毁向量
 */
void PASE_Vector_destroy(PASE_VECTOR x);

/**
 * @brief 由向量创建向量
 */
PASE_VECTOR PASE_Vector_create_by_vector(PASE_VECTOR x);

/**
 * @brief 复制向量 x 到向量 y
 */
void PASE_Vector_copy(PASE_VECTOR x, PASE_VECTOR y);

/**
 * @brief 设置向量元素为常数
 */
void PASE_Vector_set_constant_value(PASE_VECTOR x, PASE_SCALAR a);

/**
 * @brief 设置向量元素为随机值
 */
void PASE_Vector_set_random_value(PASE_VECTOR x, PASE_INT seed);

/**
 * @brief 向量校正 y = a * x + y
 */
void PASE_Vector_axpy(PASE_SCALAR a, PASE_VECTOR x, PASE_VECTOR y);

/**
 * @brief 向量数乘
 */
void PASE_Vector_scale(PASE_SCALAR a, PASE_VECTOR x);

/**
 * @brief 广义向量内积
 */
//void PASE_Vector_inner_product_general(PASE_VECTOR x, PASE_VECTOR y, PASE_MATRIX A, PASE_REAL *prod);

/**
 * @brief 向量内积 *prod = (x, y)
 */
void PASE_Vector_inner_product(PASE_VECTOR x, PASE_VECTOR y, PASE_REAL *prod);

/**
 * @brief 向量 x[start], ..., x[end] 全体内积
 */
void PASE_Vector_inner_product_some(PASE_VECTOR *x, PASE_INT start, PASE_INT end, PASE_REAL **prod);

/**
 * @brief 向量 x[i] 与 x[start], ..., x[end] 正交化, i 不能属于 [start, end]
 */
void PASE_Vector_orthogonalize(PASE_VECTOR *x, PASE_INT i, PASE_INT start, PASE_INT end);

/**
 * @brief 向量 x[0], ..., x[num-1] 全体正交化
 */
void PASE_Vector_orthogonalize_all(PASE_VECTOR *x, PASE_INT num);

/**
 * @brief 向量线性组合: y = \sum_i coef[i] * x[i]
 *
 * @param x       输入向量, 用于做线性组合的向量组
 * @param num_nev 输入向量, 用于做线型组合的向量个数
 * @param coef    输入向量, 用于做线性组合的系数数组
 * @param y       输出向量, 用于存储线性组合完毕得到的向量
 */
void PASE_Multi_vector_combination(PASE_VECTOR *x, PASE_INT num_vec, PASE_SCALAR *coef, PASE_VECTOR y);

/**
 * @brief 多重向量线性组合: y = x * mat
 *
 * @param x       输入向量, 用于做线性组合的向量组
 * @param num_nev 输入向量, 用于做线型组合的向量个数
 * @param mat     输入向量, 二维数组, 用于做线性组合的系数, 其维数为 num_vec * num_mat
 * @param num_mat 输入向量, 需做线型组合得到的向量个数
 * @param y       输出向量, 用于存储线性组合完毕得到的向量组
 */
void PASE_Multi_vector_by_matrix(PASE_VECTOR *x, PASE_INT num_vec, PASE_SCALAR **mat, PASE_INT num_mat, PASE_VECTOR *y);
#endif
