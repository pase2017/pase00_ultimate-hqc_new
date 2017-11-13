#include "pase_matrix.h"
#include "pase_vector.h"
#include <stdio.h>
#include <stdlib.h>

#include "_hypre_parcsr_mv.h"

/**
 * @brief 通过此函数进行外部矩阵类型到 PASE_MATRIX 的转换.
 *        例如对于 HYPRE 矩阵, 可设置 external_package 为 HYPRE.
 */
PASE_MATRIX 
PASE_Matrix_create_by_operator(void *matrix_data, PASE_MATRIX_OPERATOR ops)
{
    if(NULL == matrix_data) {
        printf("PASE ERROR: Can not create PASE MATRIX without matrix data.\n");
	exit(-1);
    }
    if(NULL == ops) {
        printf("PASE ERROR: Can not create PASE MATRIX without matrix operator.\n");
	exit(-1);
    }

    PASE_MATRIX A = (PASE_MATRIX)PASE_Malloc(sizeof(PASE_MATRIX_PRIVATE));

    A->ops = (PASE_MATRIX_OPERATOR)PASE_Malloc(sizeof(PASE_MATRIX_OPERATOR_PRIVATE));
    *(A->ops)               = *ops;
    A->matrix_data          = matrix_data;
    A->is_matrix_data_owner = 0;
    A->data_struct          = 0;
    A->global_nrow          = A->ops->get_global_nrow(A->matrix_data);
    A->global_ncol          = A->ops->get_global_ncol(A->matrix_data);
    return A;
}

PASE_MATRIX 
PASE_Matrix_create_default(void *matrix_data, PASE_INT data_struct)
{
    if(NULL == matrix_data) {
        printf("PASE ERROR: Can not create PASE MATRIX without matrix data.\n");
	exit(-1);
    }
    //if(NULL == param) {
    //    printf("PASE ERROR: Can not create PASE MATRIX without parameter.\n");
    //    exit(-1);
    //}

    PASE_MATRIX A = (PASE_MATRIX)PASE_Malloc(sizeof(PASE_MATRIX_PRIVATE));

    A->ops = PASE_Matrix_operator_create_default(data_struct);
    A->matrix_data = matrix_data;
    A->is_matrix_data_owner = 0;
    A->data_struct = data_struct;

    A->global_nrow = A->ops->get_global_nrow(A->matrix_data);
    A->global_ncol = A->ops->get_global_ncol(A->matrix_data);
    return A;
}

PASE_MATRIX_OPERATOR 
PASE_Matrix_operator_create(void *   (*create_matrix_by_matrix) (void *A),
                            void     (*copy_matrix)             (void *A, void *B),
                            void     (*destroy_matrix)          (void *A),
                            void *   (*multiply_matrix_matrix)  (void *A, void *B),
                            void     (*multiply_matrix_vector)  (void *A, void *x, void *y),
                            PASE_INT (*get_global_nrow)         (void *A),
                            PASE_INT (*get_global_ncol)         (void *A),
                            MPI_Comm (*get_comm_info)           (void *A))
{
    PASE_MATRIX_OPERATOR ops;
    ops = (PASE_MATRIX_OPERATOR)PASE_Malloc(sizeof(PASE_MATRIX_OPERATOR_PRIVATE));
    ops->create_matrix_by_matrix = create_matrix_by_matrix; 
    ops->copy_matrix             = copy_matrix;
    ops->destroy_matrix          = destroy_matrix;
    ops->multiply_matrix_matrix  = multiply_matrix_matrix;
    ops->multiply_matrix_vector  = multiply_matrix_vector;  
    ops->get_global_nrow         = get_global_nrow;
    ops->get_global_ncol         = get_global_ncol;
    ops->get_comm_info           = get_comm_info;

    return ops;
}

PASE_MATRIX_OPERATOR 
PASE_Matrix_operator_create_default(PASE_INT data_struct)
{
    PASE_MATRIX_OPERATOR ops = NULL;
    if( data_struct == 1) {
	//填上hypre的函数
	ops = PASE_Matrix_operator_create
	    (PASE_Matrix_create_by_matrix_hypre,
	     PASE_Matrix_copy_hypre, 
	     PASE_Matrix_destroy_hypre,
             PASE_Matrix_multiply_matrix_hypre,
	     PASE_Matrix_multiply_vector_hypre,
	     PASE_Matrix_get_global_nrow_hypre,
	     PASE_Matrix_get_global_ncol_hypre,
	     PASE_Matrix_get_comm_info_hypre);
    }
    return ops;
}

void 
PASE_Matrix_operator_destroy(PASE_MATRIX_OPERATOR ops)
{
    PASE_Free(ops);
}


void 
PASE_Matrix_destroy(PASE_MATRIX A)
{
    if(A) {
        if(A->is_matrix_data_owner > 0) {
            A->ops->destroy_matrix(A->matrix_data);
            A->matrix_data = NULL;
        }
            
        PASE_Matrix_operator_destroy(A->ops);
        A->ops = NULL;

        PASE_Free(A);
        A = NULL;
    }
}

/**
 * @brief copies A to B
 */
void 
PASE_Matrix_copy(PASE_MATRIX A, PASE_MATRIX B)
{
    A->ops->copy_matrix( A->matrix_data, B->matrix_data);
}

/**
 * @brief C = A * B
 */
PASE_MATRIX 
PASE_Matrix_multiply_matrix(PASE_MATRIX A, PASE_MATRIX B)
{
    if(A->global_ncol == B->global_nrow) {
        void *matrix_data = A->ops->multiply_matrix_matrix(A->matrix_data, B->matrix_data); 
	PASE_MATRIX C = PASE_Matrix_create_by_operator(matrix_data, A->ops); 
	C->is_matrix_data_owner = 1;
	C->data_struct = A->data_struct;
	return C;
    } else {
        printf("PASE ERROR: Matrix dimensions must be matched.\n");
        exit(-1);
    }
}


/**
 * @brief y = A * x 
 */
void 
PASE_Matrix_multiply_vector(PASE_MATRIX A, PASE_VECTOR x, PASE_VECTOR y)
{
    if( A->global_ncol == x->global_nrow) {
	A->ops->multiply_matrix_vector(A->matrix_data, x->vector_data, y->vector_data);
    } else {
        printf("PASE ERROR: Matrix dimension must be matched with vector.\n");
        exit(-1);
    }
}

void     
PASE_Matrix_multiply_vector_general(PASE_SCALAR a, PASE_MATRIX A, PASE_VECTOR x, PASE_SCALAR b, PASE_VECTOR y)
{
    PASE_VECTOR z = PASE_Vector_create_by_vector(y);
    PASE_Matrix_multiply_vector(A, x, z);
    PASE_Vector_scale(b, y);
    PASE_Vector_add_vector(a, z, y);

    PASE_Vector_destroy(z);
}

MPI_Comm
PASE_Matrix_get_comm_info(PASE_MATRIX A)
{
    return A->ops->get_comm_info(A->matrix_data);
}

//get_global_nrow get_global_ncol 写函数还是宏比较好？






void* 
PASE_Matrix_create_by_matrix_hypre(void *A)
{
    HYPRE_ParCSRMatrix B = hypre_ParCSRMatrixCompleteClone((HYPRE_ParCSRMatrix)A);
    return (void*)B;
}

void 
PASE_Matrix_copy_hypre(void *A, void *B)
{
    hypre_ParCSRMatrixCopy((HYPRE_ParCSRMatrix)A, (HYPRE_ParCSRMatrix)B, 1);
}

void 
PASE_Matrix_destroy_hypre(void *A)
{
    HYPRE_ParCSRMatrixDestroy((HYPRE_ParCSRMatrix)A);
}

void* 
PASE_Matrix_multiply_matrix_hypre(void *A, void *B)
{
    HYPRE_ParCSRMatrix C = hypre_ParMatmul((HYPRE_ParCSRMatrix)A, (HYPRE_ParCSRMatrix)B);
    MPI_Comm comm = hypre_ParCSRMatrixComm((HYPRE_ParCSRMatrix)A);
    PASE_INT num_procs;
    MPI_Comm_size(comm, &num_procs);
    if (num_procs > 1) {
    	hypre_MatvecCommPkgCreate(C);
    }

    return (void*)C;
}

void 
PASE_Matrix_multiply_vector_hypre(void *A, void *x, void *y)
{
    hypre_ParCSRMatrixMatvec(1.0, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)x, 0.0, (HYPRE_ParVector)y);
}

PASE_INT 
PASE_Matrix_get_global_nrow_hypre(void *A)
{
    return hypre_ParCSRMatrixGlobalNumRows((HYPRE_ParCSRMatrix)A);
}

PASE_INT 
PASE_Matrix_get_global_ncol_hypre(void *A)
{
    return hypre_ParCSRMatrixGlobalNumCols((HYPRE_ParCSRMatrix)A);
}

MPI_Comm
PASE_Matrix_get_comm_info_hypre(void *A)
{
    return hypre_ParCSRMatrixComm((HYPRE_ParCSRMatrix)A);
}
