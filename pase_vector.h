#ifndef __PASE_VECTOR_H__
#define __PASE_VECTOR_H__ 

#include <stdio.h>
#include "pase_config.h"
#include "pase_param.h"

//extern struct PASE_MATRIX_PRIVATE_ * PASE_MATRIX;
 
/* 向量相关操作函数集合 */
typedef struct PASE_VECTOR_OPERATOR_PRIVATE_ {
    void*      (*create_by_vector)  (void *x);
    void*      (*create_by_matrix)  (void *A);
    void       (*copy_vector)       (void *x, void *y);
    void       (*destroy_vector)    (void *x);
    void       (*set_constant_value)(void *x, PASE_SCALAR a);
    void       (*set_random_value)  (void *x, PASE_INT seed);
    void       (*inner_product)     (void *x, void *y, PASE_REAL *prod);
    /**
     * @brief 向量相加 y = x + y
     */
    void       (*add_vector)        (PASE_SCALAR a, void *x, void *y);
    /**
     * @brief 向量数乘 x = a * x
     */
    void       (*scale_vector)      (PASE_SCALAR a, void *x);
    PASE_INT   (*get_global_nrow)   (void *x);
} PASE_VECTOR_OPERATOR_PRIVATE;
typedef PASE_VECTOR_OPERATOR_PRIVATE * PASE_VECTOR_OPERATOR;

typedef struct PASE_VECTOR_PRIVATE_ {
    void                 *vector_data;
    PASE_INT             global_nrow; // 行数. PASE_VECTOR 均为列向量
    PASE_VECTOR_OPERATOR ops;
    PASE_INT             is_vector_data_owner; 
    PASE_INT             data_struct;
    //PASE_INT             is_ops_owner;
} PASE_VECTOR_PRIVATE;
typedef PASE_VECTOR_PRIVATE * PASE_VECTOR;

//typedef struct PASE_MULTI_VECTOR_PRIVATE_ {
//    PASE_INT size; // 多重向量个数
//    //PASE_VECTOR *vector; // 指向 size 个 PASE_VECTOR 变量
//    PASE_VECTOR **vector; // 指向 size 个 PASE_VECTOR * 变量
//} PASE_MULTI_VECTOR_PRIVATE;
//typedef PASE_MULTI_VECTOR_PRIVATE * PASE_MULTI_VECTOR;
#include "pase_matrix.h"

PASE_VECTOR_OPERATOR PASE_Vector_operator_create
    (void*      (*create_by_vector)  (void *x),
     void*      (*create_by_matrix)  (void *A),
     void       (*copy_vector)       (void *x, void *y),
     void       (*destroy_vector)    (void *x),
     void       (*set_constant_value)(void *x, PASE_SCALAR a),
     void       (*set_random_value)  (void *x, PASE_INT seed),
     void       (*inner_product)     (void *x, void *y, PASE_REAL *prod),
     void       (*add_vector)        (PASE_SCALAR a, void *x, void *y),
     void       (*scale_vector)      (PASE_SCALAR a, void *x),
     PASE_INT   (*get_global_nrow)   (void *x));
PASE_VECTOR_OPERATOR PASE_Vector_operator_create_default(PASE_INT data_struct);
void PASE_Vector_operator_destroy(PASE_VECTOR_OPERATOR ops);
PASE_VECTOR PASE_Vector_create_by_vector(PASE_VECTOR x);
PASE_VECTOR PASE_Vector_create_by_matrix(PASE_MATRIX A);
PASE_VECTOR PASE_Vector_create_by_operator(void *vector_data, PASE_VECTOR_OPERATOR ops);
PASE_VECTOR PASE_Vector_create_default(void *vector_data, PASE_INT data_struct);
void PASE_Vector_destroy(PASE_VECTOR x);
void PASE_Vector_copy(PASE_VECTOR x, PASE_VECTOR y);
void PASE_Vector_set_constant_value(PASE_VECTOR x, PASE_SCALAR a);
void PASE_Vector_set_random_value(PASE_VECTOR x, PASE_INT seed);
void PASE_Vector_inner_product(PASE_VECTOR x, PASE_VECTOR y, PASE_REAL *prod);
void PASE_Vector_add_vector(PASE_SCALAR a, PASE_VECTOR x, PASE_VECTOR y);
void PASE_Vector_scale(PASE_SCALAR a, PASE_VECTOR x);

void* PASE_Vector_create_by_vector_hypre(void *x);
void* PASE_Vector_create_by_matrix_hypre(void *A);
void  PASE_Vector_copy_hypre(void *x, void *y);
void  PASE_Vector_destroy_hypre(void *x);
void  PASE_Vector_set_constant_value_hypre(void *x, PASE_SCALAR a);
void  PASE_Vector_set_random_value_hypre(void *x, PASE_INT seed);
void  PASE_Vector_inner_product_hypre(void *x, void *y, PASE_REAL *prod);
void  PASE_Vector_add_vector_hypre(PASE_SCALAR a, void *x, void *y);
void  PASE_Vector_scale_hypre(PASE_SCALAR a, void *x);
PASE_INT PASE_Vector_get_global_nrow_hypre(void *x);



#endif
