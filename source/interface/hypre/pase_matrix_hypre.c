#include "pase_matrix_hypre.h"

#if PASE_USE_HYPRE
#include "_hypre_parcsr_mv.h"

void*
PASE_Matrix_create_by_matrix_hypre(void *A)
{
  HYPRE_ParCSRMatrix B = hypre_ParCSRMatrixCompleteClone((HYPRE_ParCSRMatrix)A);
  return (void*)B;
}

void
PASE_Matrix_copy_hypre(void *A, void *B)
{
  hypre_ParCSRMatrixCopy((HYPRE_ParCSRMatrix)A, (HYPRE_ParCSRMatrix)B, 1); // 1 代表 copy_data
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
  if(num_procs > 1) {
    hypre_MatvecCommPkgCreate(C);
  }
  return (void*)C;
}

void*
PASE_MatrixT_multiply_matrix_hypre(void *A, void *B)
{
  HYPRE_ParCSRMatrix C = hypre_ParTMatmul((HYPRE_ParCSRMatrix)A, (HYPRE_ParCSRMatrix)B);
  MPI_Comm comm = hypre_ParCSRMatrixComm((HYPRE_ParCSRMatrix)A);
  PASE_INT num_procs;
  MPI_Comm_size(comm, &num_procs);
  if(num_procs > 1) {
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
PASE_MatrixT_multiply_vector_hypre(void *A, void *x, void *y)
{
  hypre_ParCSRMatrixMatvecT(1.0, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)x, 0.0, (HYPRE_ParVector)y);
}

void
PASE_MatrixT_multiply_vector_general_hypre(PASE_SCALAR a, void *A, void *x, PASE_SCALAR b, void *y)
{
  hypre_ParCSRMatrixMatvecT(a, (HYPRE_ParCSRMatrix)A, (HYPRE_ParVector)x, b, (HYPRE_ParVector)y);
}

void
PASE_Matrix_get_global_nrow_hypre(void *A, PASE_INT *nrow)
{
  *nrow = hypre_ParCSRMatrixGlobalNumRows((HYPRE_ParCSRMatrix)A);
}

void
PASE_Matrix_get_global_ncol_hypre(void *A, PASE_INT *ncol)
{
  *ncol = hypre_ParCSRMatrixGlobalNumCols((HYPRE_ParCSRMatrix)A);
}

MPI_Comm
PASE_Matrix_get_mpi_comm_hypre(void *A)
{
  return hypre_ParCSRMatrixComm((HYPRE_ParCSRMatrix)A);
}

#endif

