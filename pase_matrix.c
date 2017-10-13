#include "pase_matrix.h"

/**
 * @brief 通过此函数进行外部矩阵类型到 PASE_MATRIX 的转换.
 *        例如对于 HYPRE 矩阵, 可设置 external_package 为 HYPRE.
 */
PASE_MATRIX PASE_Create_matrix(void *matrix_data, PASE_PARAMETER param, PASE_MATRIX_OPERATOR ops)
{
    PASE_MATRIX A = (PASE_MATRIX)PASE_Malloc(sizeof(PASE_MATRIX));

    if(NULL != ops) {
        A->ops = ops;
	A->is_ops_owner = 0;
    } else if(NULL != param) {
	A->ops = PASE_Create_matrix_operator_default( param);
	A->is_ops_owner = 1;
    }

    if( NULL != param){
        if(param.copy_matrix > 0) {
            A->matrix_data = A->ops->create_matrix_by_matrix(matrix_data);
            A->ops->copy_matrix(matrix_data, A->matrix_data);
            A->is_matrix_data_owner = 1;
        }
    } else {
        A->matrix_data = matrix_data;
        A->is_matrix_data_owner = 0;
    }

    A->global_nrow = A->ops->get_global_nrow(A->matrix_data);
    A->global_ncol = A->ops->get_global_ncol(A->matrix_data);
    return A;
}

PASE_MATRIX_OPERATOR PASE_Create_matrix_operator(
    void *   (*create_matrix_by_matrix) (void *A),
    void     (*copy_matrix)             (void *A, void *B),
    void     (*destroy_matrix)          (void *A),
    void *   (*multiply_matrix_matrix)  (void *A, void *B),
    void     (*multiply_matrix_vector)  (PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y),
    PASE_INT (*get_global_nrow)         (void *A),
    PASE_INT (*get_global_ncol)         (void *A)
	)
{
    PASE_MATRIX_OPERATOR ops;
    ops = (PASE_MATRIX_OPERATOR)PASE_Malloc(sizeof(PASE_MATRIX_OPERATOR);
    ops->create_matrix_by_matrix = create_matrix_by_matrix; 
    ops->copy_matrix             = copy_matrix;
    ops->destroy_matrix          = destroy_matrix;
    ops->multiply_matrix_matrix  = multiply_matrix_matrix;
    ops->multiply_matrix_vector  = multiply_matrix_vector;  
    ops->get_global_nrow         = get_global_nrow;
    ops->get_global_ncol         = get_global_ncol;

    return ops;
}

PASE_MATRIX_OPERATOR PASE_Create_matrix_operator_default( PASE_PARAMETER param)
{
    PASE_MATRIX_OPERATOR ops;
    if( param.external_package == 1) {
	//填上hypre的函数
	ops = PASE_Create_matrix_operator(
            PASE_Create_matrix_by_matrix_hypre,
	    PASE_Copy_matrix_hypre, 
	    HYPRE_ParCSRMatrixDestroy,
	    hypre_ParMatmul,
	    hypre_ParCSRMatrixMatvec,
	    PASE_Get_matrix_global_nrow_hypre,
	    PASE_Get_matrix_global_ncol_hypre
	    );
    }
    return ops;
}

void PASE_Destroy_matrix_operator(PASE_MATRIX_OPERATOR ops)
{
    PASE_Free(ops);
}

void* PASE_Hypre_create_matrix_by_matrix(void *A)
{

}

void PASE_Hypre_copy_matrix(void *A, void *B)
{
    hypre_ParCSRMatrixCopy( (hypre_ParCSRMatrix*)A, (hypre_ParCSRMatrix*)B, 1);
}

PASE_Int PASE_Hypre_get_matrix_global_nrow(void *A)
{
    return hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix*)A);
}

PASE_Int PASE_Hypre_get_matrix_global_ncol(void *A)
{
    return hypre_ParCSRMatrixGlobalNumCols((hypre_ParCSRMatrix*)A);
}

void PASE_Destroy_matrix(PASE_MATRIX A)
{
    if(A->is_matrix_data_owner > 0) {
        A->ops->destroy_matrix(A->matrix_data);
        A->matrix_data = NULL;
    }
    if(A->is_ops_owner > 0) {
	PASE_Destroy_matrix_operator(A->ops);
	A->ops = NULL;
    }
    PASE_Free(A);
    A = NULL;
}

/**
 * @brief copies A to B
 */
void PASE_Copy_matrix(PASE_MATRIX A, PASE_MATRIX B)
{
    A->ops->copy_matrix( A->matrix_data, B->matrix_data);
}

/**
 * @brief C = A * B
 */
PASE_Matrix PASE_Matrix_multiply_matrix(PASE_MATRIX A, PASE_MATRIX B)
{
    if(A->global_ncol == B->global_nrow) {
        void *matrix_data = A->ops->multiply_matrix_matrix(A->matrix_data, B->matrix_data); 
	PASE_Matrix C = PASE_Create_matrix(matrix_data, NULL, ops); 
	return C;
    } else {
        printf("PASE ERROR: Matrix dimensions must be matched.\n");
        exit(-1);
    }
}


/**
 * @brief y = a * A * x + b * y
 */
void PASE_Matrix_multiply_vector(PASE_SCALAR a, PASE_MATRIX A, PASE_VECTOR x, PASE_SCALAR b, PASE_VECTOR y)
{
    if( A->global_ncol == x->global_nrow) {
	A->ops->multiply_matrix_vector(a, A->matrix_data, x->vector_data, b, y->vector_data);
    } else {
        printf("PASE ERROR: Matrix dimension must be matched with vector.\n");
        exit(-1);
    }
}

//get_global_nrow get_global_ncol 写函数还是宏比较好？
