#include "pase_vector.h"

/**
 * @brief Create PASE_VECTOR
 */
PASE_VECTOR PASE_Create_vector(void *vector_data, PASE_PARAMETER param, PASE_VECTOR_OPERATOR ops)
{
    PASE_VECTOR x = (PASE_VECTOR) PASE_Malloc(sizeof(PASE_VECTOR));

    if(NULL != ops) {
        x->ops = ops;
	x->is_ops_owner = 0;
    } else if(NULL != param) {
	x->ops = PASE_Create_vector_operator_default( param);
	x->is_ops_owner = 1;
    }

    if( NULL != param){
        if(param.copy_matrix > 0) {
            x->vector_data = x->ops->create_vector_by_vector(vector_data);
            x->ops->copy_vector(vector_data, x->vector_data);
            x->is_vector_data_owner = 1;
        }
    } else {
        x->vector_data = vector_data;
        x->is_vector_data_owner = 0;
    }

    x->global_nrow = x->ops->get_global_nrow(x->vector_data);
    return x;
}


PASE_VECTOR_OPERATOR PASE_Create_vector_operator(
    void       (*destroy_vector)    (void *x);
    void       (*set_constant_value)(void *x, PASE_SCALAR a),
    void       (*set_random_value)  (void *x, PASE_INT seed),
    void       (*inner_product)     (void *x, void *y, PASE_REAL prod),
    void       (*add_vector_vector) (void *x, void *y, void *z),
    void       (*scale_vector)      (PASE_SCALAR a, void *x, void *y),
    void       (*copy_vector)       (void *x, void *y),
    PASE_INT   (*get_global_nrow)   (void *x)
	)
{
    PASE_VECTOR_OPERATOR ops;
    ops = (PASE_VECTOR_OPERATOR)PASE_Malloc(sizeof(PASE_VECTOR_OPERATOR);

    ops->destroy_vector     = destroy_vector;
    ops->set_constant_value = set_constant_value;
    ops->set_random_value   = set_random_value;
    ops->inner_product      = inner_product;
    ops->add_vector_vector  = add_vector_vector;
    ops->scale_vector       = scale_vector;
    ops->copy_vector        = copy_vector;
    ops->get_global_nrow    = get_global_nrow;

    return ops;
}

PASE_VECTOR_OPERATOR PASE_Create_vector_operator_default( PASE_PARAMETER param)
{
    PASE_VECTOR_OPERATOR ops;
    if( param.external_package == 1) {
	//填上hypre的函数
	ops = PASE_Create_matrix_operator(
            HYPRE_ParVectorDestroy,

	    HYPRE_ParVectorCopy,
	    
	    );
    }
    return ops;
}

void PASE_Destroy_vector_operator(PASE_VECTOR_OPERATOR ops)
{
    PASE_Free(ops);
}

void PASE_Destroy_vector(PASE_VECTOR x)
{
    if(x->is_vector_data_owner > 0) {
        x->ops->destroy_vector(x->vector_data);
        x->vector_data = NULL;
    }
    if(x->is_ops_owner > 0) {
	PASE_Destroy_vector_operator(x->ops);
	x->ops = NULL;
    }
    PASE_Free(x);
    x = NULL;
}

void PASE_Vector_add_vector(PASE_VECTOR x, PASE_VECTOR y, PASE_VECTOR)
{

}

void PASE_Vector_scale(PASE_SCALAR a, PASE_VECTOR x, PASE_VECTOR y)
{

}

void PASE_Vector_copy(PASE_VECTOR x, PASE_VECTOR y)
{

}


