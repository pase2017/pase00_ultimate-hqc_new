#ifndef __PASE_MATRIX_HYPRE_H__
#define __PASE_MATRIX_HYPRE_H__

#include "pase_config.h"

#if PASE_USE_HYPRE

void * PASE_Matrix_create_by_matrix_hypre(void *A);
void   PASE_Matrix_destroy_hypre(void *A);
void * PASE_Matrix_transpose_hypre(void *A);
void   PASE_Matrix_copy_hypre(void *A, void *B);
void * PASE_Matrix_multiply_matrix_hypre(void *A, void *B);
void * PASE_MatrixT_multiply_matrix_hypre(void *A, void *B);
void   PASE_Matrix_multiply_vector_hypre(void *A, void *x, void *y);
void   PASE_MatrixT_multiply_vector_hypre(void *A, void *x, void *y);
void   PASE_Matrix_multiply_vector_general_hypre(PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y);
void   PASE_MatrixT_multiply_vector_general_hypre(PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y);
void   PASE_Matrix_get_global_nrow_hypre(void *A, PASE_INT *nrow);
void   PASE_Matrix_get_global_ncol_hypre(void *A, PASE_INT *ncol);
MPI_Comm PASE_Matrix_get_mpi_comm_hypre(void *A);

#endif

#endif
