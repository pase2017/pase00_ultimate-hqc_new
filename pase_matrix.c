#include "pase_matrix.h"
#include "pase_vector.h"
#include <stdio.h>
#include <stdlib.h>

#include "_hypre_parcsr_mv.h"

/**
 * @brief 通过此函数进行外部矩阵类型到 PASE_MATRIX 的转换.
 *        需输入矩阵数据与矩阵相关操作集.
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

/**
 * @brief 通过此函数进行缺省类型的外部矩阵到 PASE_MATRIX 的转换.
 *        需输入向量数据与缺省类型参数data_struct, 如HYPRE的类型参数为 1.
 */
PASE_MATRIX 
PASE_Matrix_create_default(void *matrix_data, PASE_INT data_struct)
{
    if(NULL == matrix_data) {
        printf("PASE ERROR: Can not create PASE MATRIX without matrix data.\n");
	exit(-1);
    }

    PASE_MATRIX A = (PASE_MATRIX)PASE_Malloc(sizeof(PASE_MATRIX_PRIVATE));

    A->ops = PASE_Matrix_operator_create_default(data_struct);
    A->matrix_data = matrix_data;
    A->is_matrix_data_owner = 0;
    A->data_struct = data_struct;

    A->global_nrow = A->ops->get_global_nrow(A->matrix_data);
    A->global_ncol = A->ops->get_global_ncol(A->matrix_data);
    return A;
}

/**
 * @brief 通过此函数, 用户可生成自定义的矩阵操作集.
 */
PASE_MATRIX_OPERATOR 
PASE_Matrix_operator_create(void *   (*create_matrix_by_matrix) (void *A),
                            void     (*copy_matrix)             (void *A, void *B),
                            void     (*destroy_matrix)          (void *A),
			    void *   (*transpose_matrix)        (void *A),
                            void *   (*multiply_matrix_matrix)  (void *A, void *B),
                            void *   (*multiply_matrixT_matrix) (void *A, void *B),
                            void     (*multiply_matrix_vector)  (void *A, void *x, void *y),
                            void     (*multiply_matrix_vector_general)  (PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y),
                            void     (*multiply_matrixT_vector) (void *A, void *x, void *y),
                            void     (*multiply_matrixT_vector_general) (PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y),
                            PASE_INT (*get_global_nrow)         (void *A),
                            PASE_INT (*get_global_ncol)         (void *A),
                            MPI_Comm (*get_comm_info)           (void *A))
{
    PASE_MATRIX_OPERATOR ops;
    ops = (PASE_MATRIX_OPERATOR)PASE_Malloc(sizeof(PASE_MATRIX_OPERATOR_PRIVATE));
    ops->create_matrix_by_matrix = create_matrix_by_matrix; 
    ops->copy_matrix             = copy_matrix;
    ops->destroy_matrix          = destroy_matrix;
    ops->transpose_matrix        = transpose_matrix;
    ops->multiply_matrix_matrix  = multiply_matrix_matrix;
    ops->multiply_matrixT_matrix = multiply_matrixT_matrix;
    ops->multiply_matrix_vector  = multiply_matrix_vector;  
    ops->multiply_matrix_vector_general  = multiply_matrix_vector_general;  
    ops->multiply_matrixT_vector = multiply_matrixT_vector;  
    ops->multiply_matrixT_vector_general = multiply_matrixT_vector_general;  
    ops->get_global_nrow         = get_global_nrow;
    ops->get_global_ncol         = get_global_ncol;
    ops->get_comm_info           = get_comm_info;

    return ops;
}

/**
 * @brief 通过此函数, 可生成缺省类型的矩阵操作集.
 */
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
	     PASE_Matrix_transpose_hypre,
             PASE_Matrix_multiply_matrix_hypre,
	     PASE_Matrix_transposition_multiply_matrix_hypre,
	     PASE_Matrix_multiply_vector_hypre,
	     PASE_Matrix_multiply_vector_general_hypre,
	     PASE_Matrix_transposition_multiply_vector_hypre,
	     PASE_Matrix_transposition_multiply_vector_general_hypre,
	     PASE_Matrix_get_global_nrow_hypre,
	     PASE_Matrix_get_global_ncol_hypre,
	     PASE_Matrix_get_comm_info_hypre);
    }
    return ops;
}

/**
 * @brief 销毁操作集ops, 释放内存空间. 
 */
void 
PASE_Matrix_operator_destroy(PASE_MATRIX_OPERATOR ops)
{
    PASE_Free(ops);
}

/**
 * @brief Destroy matrix A and free memory. 
 */
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
 * @brief Copies A to B.
 */
void 
PASE_Matrix_copy(PASE_MATRIX A, PASE_MATRIX B)
{
    A->ops->copy_matrix( A->matrix_data, B->matrix_data);
}

/**
 * @brief Perform B = A^T.
 */
PASE_MATRIX
PASE_Matrix_transpose(PASE_MATRIX A)
{
    void *matrix_data = NULL;
    if(A->ops->transpose_matrix){
        matrix_data = A->ops->transpose_matrix(A->matrix_data);
    } else {
	printf("PASE ERROR in PASE_Matrix_transpose: Can not perform B = A^T without A->ops->transpose_matrix.\n");
	exit(-1);
    }
    PASE_MATRIX B = PASE_Matrix_create_by_operator(matrix_data, A->ops); 
    B->is_matrix_data_owner = 1;
    B->data_struct = A->data_struct;
    return B;
}


/**
 * @brief Perform C = A * B.
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
        printf("PASE ERROR in PASE_Matrix_multiply_matrix: Matrix dimensions must be matched.\n");
        exit(-1);
    }
}

/**
 * @brief Perform C = A^T * B.
 */
PASE_MATRIX
PASE_Matrix_transposition_multiply_matrix(PASE_MATRIX A, PASE_MATRIX B)
{
    if(A->global_nrow == B->global_nrow) {
        void *matrix_data = NULL; 
	if(A->ops->multiply_matrixT_matrix) {
	    matrix_data = A->ops->multiply_matrixT_matrix(A->matrix_data, B->matrix_data);
	} else if(A->ops->transpose_matrix) {
	    void *workspace = A->ops->transpose_matrix(A->matrix_data);
            matrix_data = A->ops->multiply_matrix_matrix(workspace, B->matrix_data); 
	    A->ops->destroy_matrix(workspace);
	} else {
	    printf("PASE ERROR in PASE_Matrix_transposition_multiply_matrix: Can not perform C=A^T*B without A->ops->multiply_matrixT_matrix nor A->ops->transpose_matrix.\n");
	    exit(-1);
	}
	PASE_MATRIX C = PASE_Matrix_create_by_operator(matrix_data, A->ops); 
	C->is_matrix_data_owner = 1;
	C->data_struct = A->data_struct;
	return C;
    } else {
        printf("PASE ERROR in PASE_Matrix_multiply_matrix: Matrix dimensions must be matched.\n");
        exit(-1);
    }
}

/**
 * @brief Perform y = A * x. 
 */
void 
PASE_Matrix_multiply_vector(PASE_MATRIX A, PASE_VECTOR x, PASE_VECTOR y)
{
    if(A->global_ncol == x->global_nrow && A->global_nrow == y->global_nrow) {
	if(A->ops->multiply_matrix_vector) {
	    A->ops->multiply_matrix_vector(A->matrix_data, x->vector_data, y->vector_data);
	} else if(A->ops->multiply_matrix_vector_general) {
	    A->ops->multiply_matrix_vector_general(1.0, A->matrix_data, x->vector_data, 0.0, y->vector_data);
	} else {
	    printf("PASE ERROR in PASE_Matrix_multiply_vector: Can not perform y=A*x without A->ops->multiply_matrix_vector nor A->ops->multiply_matrix_vector_general.\n");
	    exit(-1);
	}
    } else {
        printf("PASE ERROR in PASE_Matrix_multiply_vector: Matrix dimension must be matched with vector.\n");
        exit(-1);
    }
}

/**
 * @brief Perform y = a * A * x + b * y.
 */
void     
PASE_Matrix_multiply_vector_general(PASE_SCALAR a, PASE_MATRIX A, PASE_VECTOR x, PASE_SCALAR b, PASE_VECTOR y)
{
    if(A->global_ncol == x->global_nrow && A->global_nrow == y->global_nrow) {
	if(A->ops->multiply_matrix_vector_general) {
	    A->ops->multiply_matrix_vector_general(a, A->matrix_data, x->vector_data, b, y->vector_data);
	} else if(A->ops->multiply_matrix_vector) {
            void *z = x->ops->create_by_vector(y->vector_data);
            A->ops->multiply_matrix_vector(A->matrix_data, x->vector_data, z);
            x->ops->scale_vector(b, y->vector_data);
            x->ops->add_vector(a, z, y->vector_data);
            x->ops->destroy_vector(z);
	} else {
	    printf("PASE ERROR in PASE_Matrix_multiply_vector_general: Can not perform y=a*A*x+b*y without A->ops->multiply_matrix_vector nor A->ops->multiply_matrix_vector_general.\n");
	    exit(-1);
	}
    } else {
        printf("PASE ERROR in PASE_Matrix_multiply_vector_general: Matrix dimension must be matched with vector.\n");
        exit(-1);
    }	
}

/**
 * @brief Perform y = A^T * x. 
 */
void 
PASE_Matrix_transposition_multiply_vector(PASE_MATRIX A, PASE_VECTOR x, PASE_VECTOR y)
{
    if(A->global_nrow == x->global_nrow && A->global_ncol == y->global_nrow) {
	if(A->ops->multiply_matrixT_vector) {
	    A->ops->multiply_matrixT_vector(A->matrix_data, x->vector_data, y->vector_data);
	} else if(A->ops->multiply_matrixT_vector_general) {
	    A->ops->multiply_matrixT_vector_general(1.0, A->matrix_data, x->vector_data, 0.0, y->vector_data);
	} else {
	    printf("PASE ERROR in PASE_Matrix_transposition_multiply_vector: Can not perform y=A^T*x without A->ops->multiply_matrixT_vector nor A->ops->multiply_matrixT_vector_general.\n");
	    exit(-1);
	}
    } else {
        printf("PASE ERROR in PASE_Matrix_transposition_multiply_vector: Matrix dimension must be matched with vector.\n");
        exit(-1);
    }
}

/**
 * @brief Perform y = a * A^T * x + b * y.
 */
void     
PASE_Matrix_transposition_multiply_vector_general(PASE_SCALAR a, PASE_MATRIX A, PASE_VECTOR x, PASE_SCALAR b, PASE_VECTOR y)
{
    if(A->global_nrow == x->global_nrow && A->global_ncol == y->global_nrow) {
	if(A->ops->multiply_matrixT_vector_general) {
	    A->ops->multiply_matrixT_vector_general(a, A->matrix_data, x->vector_data, b, y->vector_data);
	} else if(A->ops->multiply_matrixT_vector) {
            void *z = x->ops->create_by_vector(y->vector_data);
            A->ops->multiply_matrixT_vector(A->matrix_data, x->vector_data, z);
            x->ops->scale_vector(b, y->vector_data);
            x->ops->add_vector(a, z, y->vector_data);
            x->ops->destroy_vector(z);
	} else {
	    printf("PASE ERROR in PASE_Matrix_transposition_multiply_vector_general: Can not perform y=a*A^T*x+b*y without A->ops->multiply_matrixT_vector nor A->ops->multiply_matrixT_vector_general.\n");
	    exit(-1);
	}
    } else {
        printf("PASE ERROR in PASE_Matrix_transposition_multiply_vector_general: Matrix dimension must be matched with vector.\n");
        exit(-1);
    }	
}

/**
 * @brief Get MPI_Comm of A.
 */
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
PASE_Matrix_transpose_hypre(void *A)
{
    HYPRE_ParCSRMatrix AT;
    hypre_ParCSRMatrixTranspose(A, &AT, 1);
    return (void*)AT;
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

void* 
PASE_Matrix_transposition_multiply_matrix_hypre(void *A, void *B)
{
    HYPRE_ParCSRMatrix C = hypre_ParTMatmul((HYPRE_ParCSRMatrix)A, (HYPRE_ParCSRMatrix)B);
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

void 
PASE_Matrix_multiply_vector_general_hypre(PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y)
{
    hypre_ParCSRMatrixMatvec(a, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)x, b, (HYPRE_ParVector)y);
}

void 
PASE_Matrix_transposition_multiply_vector_hypre(void *A, void *x, void *y)
{
    hypre_ParCSRMatrixMatvecT(1.0, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)x, 0.0, (HYPRE_ParVector)y);
}

void 
PASE_Matrix_transposition_multiply_vector_general_hypre(PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y)
{
    hypre_ParCSRMatrixMatvecT(a, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)x, b, (HYPRE_ParVector)y);
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
