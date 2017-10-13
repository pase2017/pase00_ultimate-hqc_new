#ifndef __PASE_VECTOR_H__
#define __PASE_VECTOR_H__ 

/* 向量相关操作函数集合 */
typedef struct PASE_VECTOR_OPERATOR_PRIVATE_ {
    void       (*destroy_vector)    (void *x);
    void       (*set_constant_value)(void *x, PASE_SCALAR a);
    void       (*set_random_value)  (void *x, PASE_INT seed);
    void       (*inner_product)     (void *x, void *y, PASE_REAL prod);
    /**
     * @brief 向量相加 z = x + y
     */
    void       (*add_vector_vector) (void *x, void *y, void *z);
    /**
     * @brief 向量数乘 y = a * x
     */
    void       (*scale_vector)      (PASE_SCALAR a, void *x, void *y);
    void       (*copy_vector)       (void *x, void *y);
    PASE_INT   (*get_global_nrow)   (void *x);
} PASE_VECTOR_OPERATOR_PRIVATE;

typedef PASE_VECTOR_OPERATOR_PRIVATE * PASE_VECTOR_OPERATOR;

typedef struct PASE_VECTOR_PRIVATE_ {
    void                 *vector_data;
    PASE_INT             global_nrow; // 行数. PASE_VECTOR 均为列向量
    PASE_VECTOR_OPERATOR ops;
    PASE_INT             is_vector_data_owner; 
    PASE_INT             is_ops_owner;
} PASE_VECTOR_PRIVATE;
typedef PASE_VECTOR_PRIVATE * PASE_VECTOR;

//typedef struct PASE_MULTI_VECTOR_PRIVATE_ {
//    PASE_INT size; // 多重向量个数
//    //PASE_VECTOR *vector; // 指向 size 个 PASE_VECTOR 变量
//    PASE_VECTOR **vector; // 指向 size 个 PASE_VECTOR * 变量
//} PASE_MULTI_VECTOR_PRIVATE;
//typedef PASE_MULTI_VECTOR_PRIVATE * PASE_MULTI_VECTOR;

PASE_VECTOR PASE_Create_vector(void *vector_data, PASE_PARAMETER param, PASE_VECTOR_OPERATOR ops);
PASE_VECTOR_OPERATOR PASE_Create_vector_operator(
    void       (*destroy_vector)    (void *x),
    void       (*set_constant_value)(void *x, PASE_SCALAR a),
    void       (*set_random_value)  (void *x, PASE_INT seed),
    void       (*inner_product)     (void *x, void *y, PASE_REAL prod),
    void       (*add_vector_vector) (void *x, void *y, void *z),
    void       (*scale_vector)      (PASE_SCALAR a, void *x, void *y),
    void       (*copy_vector)       (void *x, void *y),
    PASE_INT   (*get_global_nrow)   (void *x)
	);
PASE_VECTOR_OPERATOR PASE_Create_vector_operator_default( PASE_PARAMETER param);
void PASE_Destroy_vector_operator(PASE_VECTOR_OPERATOR ops)
void PASE_Destroy_vector(PASE_VECTOR x);

void PASE_Vector_add_vector(PASE_VECTOR x, PASE_VECTOR y, PASE_VECTOR);
void PASE_Vector_scale(PASE_SCALAR a, PASE_VECTOR x, PASE_VECTOR y);
void PASE_Vector_copy(PASE_VECTOR x, PASE_VECTOR y);





#endif
