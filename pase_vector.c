#include <stdio.h>
#include <stdlib.h>
//#include "pase_matrix.h"
#include "pase_vector.h"
#include "pase_config.h"
#include "pase_param.h"

//#include "HYPRE_seq_mv.h"
//#include "HYPRE.h"
//#include "HYPRE_parcsr_ls.h"
//#include "interpreter.h"
//#include "HYPRE_MatvecFunctions.h"
#include "_hypre_parcsr_mv.h"
//#include "_hypre_parcsr_ls.h"
//#include "temp_multivector.h"
//#include "HYPRE_utilities.h"
	   

/**
 * @brief Create PASE_VECTOR
 */
//PASE_VECTOR
//PASE_Vector_create_by_pase_vector(PASE_VECTOR x)
//{
//    void        *vector_data = x->ops->create_by_vector(x->vector_data);
//    PAES_VECTOR  y           = PASE_Vector_create_by_operator(vector_data, x->ops);
//    y->is_vector_data_owner  = 1;
//    return y;
//}

//PASE_VECTOR
//PASE_Vector_create_by_pase_matrix(PASE_MATRIX A)
//{
//    void        *vector_data = x->ops->create_by_matrix(A->matrix_data);
//    PAES_VECTOR  y           = PASE_Vector_create_default(vector_data, A->data_struct);
//    y->is_vector_data_owner  = 1;
//    return y;
//}

PASE_VECTOR 
PASE_Vector_create_by_operator(void *vector_data, PASE_VECTOR_OPERATOR ops)
{
    if(NULL == vector_data) {
	printf("PASE ERROR: Can not create PAES VECTOR without vector data.\n");
	exit(-1);
    }
    if(NULL == ops) {
	printf("PASE ERROR: Can not create PAES VECTOR without vector operator.\n");
	exit(-1);
    }

    PASE_VECTOR x = (PASE_VECTOR) PASE_Malloc(sizeof(PASE_VECTOR));

    x->ops = (PASE_VECTOR_OPERATOR)PASE_Malloc(sizeof(PASE_VECTOR_OPERATOR_PRIVATE));
    *(x->ops)               = *ops;
    x->vector_data          = vector_data;
    x->is_vector_data_owner = 0;
    x->global_nrow = x->ops->get_global_nrow(x->vector_data);
    x->data_struct = 0;
    return x;
}

PASE_VECTOR 
PASE_Vector_create_default(void *vector_data, PASE_INT data_struct)
{
    if(NULL == vector_data) {
	printf("PASE ERROR: Can not create PAES VECTOR without vector data.\n");
	exit(-1);
    }
    //if( data_struct < 0 || data_struct > 2) {
    //    printf("PASE ERROR: Data struct can not be %d.\n", data_struct);
    //    exit(-1);
    //}

    PASE_VECTOR x = (PASE_VECTOR) PASE_Malloc(sizeof(PASE_VECTOR_PRIVATE));

    x->ops = PASE_Vector_operator_create_default(data_struct);
    x->vector_data          = vector_data;
    x->is_vector_data_owner = 0;
    x->global_nrow          = x->ops->get_global_nrow(x->vector_data);
    x->data_struct          = data_struct;
    return x;
}

PASE_VECTOR_OPERATOR 
PASE_Vector_operator_create(void*      (*create_by_vector)  (void *x),
                            void*      (*create_by_matrix)  (void *A),
                            void       (*copy_vector)       (void *x, void *y),
	                    void       (*destroy_vector)    (void *x),
                            void       (*set_constant_value)(void *x, PASE_SCALAR a),
                            void       (*set_random_value)  (void *x, PASE_INT seed),
                            void       (*inner_product)     (void *x, void *y, PASE_REAL *prod),
                            void       (*add_vector) (void *x, void *y),
                            void       (*scale_vector)      (PASE_SCALAR a, void *x),
                            PASE_INT   (*get_global_nrow)   (void *x))
{
    PASE_VECTOR_OPERATOR ops;
    ops = (PASE_VECTOR_OPERATOR)PASE_Malloc(sizeof(PASE_VECTOR_OPERATOR_PRIVATE));

    ops->create_by_vector   = create_by_vector;
    ops->create_by_matrix   = create_by_matrix;
    ops->copy_vector        = copy_vector;
    ops->destroy_vector     = destroy_vector;
    ops->set_constant_value = set_constant_value;
    ops->set_random_value   = set_random_value;
    ops->inner_product      = inner_product;
    ops->add_vector         = add_vector;
    ops->scale_vector       = scale_vector;
    ops->get_global_nrow    = get_global_nrow;

    return ops;
}

PASE_VECTOR_OPERATOR 
PASE_Vector_operator_create_default(PASE_INT data_struct)
{
    PASE_VECTOR_OPERATOR ops;
    if( data_struct == 1) {
	//填上hypre的函数
	ops = PASE_Vector_operator_create
	    (PASE_Vector_create_by_vector_hypre,
	     PASE_Vector_create_by_matrix_hypre,
	     PASE_Vector_copy_hypre,
             PASE_Vector_destroy_hypre,
	     PASE_Vector_set_constant_value_hypre,
	     PASE_Vector_set_random_value_hypre,
	     PASE_Vector_inner_product_hypre,
	     PASE_Vector_add_vector_hypre,
	     PASE_Vector_scale_hypre,
	     PASE_Vector_get_global_nrow_hypre);

    }
        
    return ops;
}

void 
PASE_Vector_operator_destroy(PASE_VECTOR_OPERATOR ops)
{
    PASE_Free(ops);
}

void 
PASE_Vector_destroy(PASE_VECTOR x)
{
    if(x->is_vector_data_owner > 0) {
        x->ops->destroy_vector(x->vector_data);
        x->vector_data = NULL;
    }
    PASE_Vector_operator_destroy(x->ops);
    x->ops = NULL;
    PASE_Free(x);
    x = NULL;
}

void 
PASE_Vector_copy(PASE_VECTOR x, PASE_VECTOR y)
{
    if(x->global_nrow != y->global_nrow) {
	printf("PASE ERROR: Vector dimensions must be matched.\n");
	exit(-1);
    }
    x->ops->copy_vector(x->vector_data, y->vector_data);
}

void
PASE_Vector_set_constant_value(PASE_VECTOR x, PASE_SCALAR a)
{
    x->ops->set_constant_value(x->vector_data, a);
}

void
PASE_Vector_set_random_value(PASE_VECTOR x, PASE_INT seed)
{
    x->ops->set_random_value(x->vector_data, seed);
}

void
PASE_Vector_inner_product(PASE_VECTOR x, PASE_VECTOR y, PASE_REAL *prod)
{
    x->ops->inner_product(x->vector_data, y->vector_data, prod);
}

void 
PASE_Vector_add_vector(PASE_VECTOR x, PASE_VECTOR y)
{
    x->ops->add_vector(x->vector_data, y->vector_data);
}

void 
PASE_Vector_scale(PASE_SCALAR a, PASE_VECTOR x)
{
    x->ops->scale_vector(a, x->vector_data);
}

void*
PASE_Vector_create_by_vector_hypre(void *x)
{
    HYPRE_ParVector  x_hypre      = (HYPRE_ParVector)x;
    MPI_Comm         comm         = hypre_ParVectorComm(x_hypre); 
    PASE_INT         global_size  = hypre_ParVectorGlobalSize(x_hypre);
    PASE_INT        *partitioning = hypre_ParVectorPartitioning(x_hypre);

    HYPRE_ParVector  y            = hypre_ParVectorCreate(comm, global_size, partitioning);
    HYPRE_ParVectorInitialize(y);
    hypre_ParVectorSetPartitioningOwner(y, 0);
    return (void*)y;
}

void*
PASE_Vector_create_by_matrix_hypre(void *A)
{
    HYPRE_ParCSRMatrix A_hypre      = (HYPRE_ParCSRMatrix)A;
    MPI_Comm           comm         = hypre_ParCSRMatrixComm(A_hypre);
    PASE_INT           global_size  = hypre_ParCSRMatrixGlobalNumRows(A_hypre);
    PASE_INT          *partitioning = NULL;
    HYPRE_ParCSRMatrixGetRowPartitioning(A_hypre, &partitioning);

    HYPRE_ParVector    y            = hypre_ParVectorCreate(comm, global_size, partitioning);
    HYPRE_ParVectorInitialize(y);
    hypre_ParVectorSetPartitioningOwner(y, 1);
    return (void*)y;
}

void 
PASE_Vector_copy_hypre(void *x, void *y)
{
    HYPRE_ParVectorCopy((HYPRE_ParVector)x, (HYPRE_ParVector)y);
}

void 
PASE_Vector_destroy_hypre(void *x)
{
    HYPRE_ParVectorDestroy((HYPRE_ParVector)x);
}

void 
PASE_Vector_set_constant_value_hypre(void *x, PASE_SCALAR a)
{
    HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)x, a);
}

void
PASE_Vector_set_random_value_hypre(void *x, PASE_INT seed)
{
    HYPRE_ParVectorSetRandomValues((HYPRE_ParVector)x, seed);
}

void
PASE_Vector_inner_product_hypre(void *x, void *y, PASE_REAL *prod)
{
    HYPRE_ParVectorInnerProd((HYPRE_ParVector)x, (HYPRE_ParVector)y, prod);
}

void 
PASE_Vector_add_vector_hypre(void *x, void *y)
{
    HYPRE_ParVectorAxpy(1.0, (HYPRE_ParVector)x, (HYPRE_ParVector)y);
}

void 
PASE_Vector_scale_hypre(PASE_SCALAR a, void *x)
{
    HYPRE_ParVectorScale(a, (HYPRE_ParVector)x);
}

PASE_INT 
PASE_Vector_get_global_nrow_hypre(void *x)
{
    return hypre_ParVectorGlobalSize((HYPRE_ParVector)x);
}

