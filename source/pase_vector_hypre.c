#include "pase_vector_hypre.h"

#if PASE_USE_HYPRE
#include "_hypre_parcsr_mv.h"

void*
PASE_Vector_data_create_by_vector_hypre(void *x)
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
PASE_Vector_data_create_by_matrix_hypre(void *A)
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
PASE_Vector_data_copy_hypre(void *x, void *y)
{
  HYPRE_ParVectorCopy((HYPRE_ParVector)x, (HYPRE_ParVector)y);
}

void
PASE_Vector_data_destroy_hypre(void *x)
{
  HYPRE_ParVectorDestroy((HYPRE_ParVector)x);
  x = NULL;
}

void
PASE_Vector_data_set_constant_value_hypre(void *x, PASE_SCALAR a)
{
  HYPRE_ParVectorSetConstantValues((HYPRE_ParVector)x, a);
}

void
PASE_Vector_data_set_random_value_hypre(void *x, PASE_INT seed)
{
  HYPRE_ParVectorSetRandomValues((HYPRE_ParVector)x, seed);
}

void
PASE_Vector_data_inner_product_hypre(void *x, void *y, PASE_REAL *prod)
{
  HYPRE_ParVectorInnerProd((HYPRE_ParVector)x, (HYPRE_ParVector)y, prod);
}

void
PASE_Vector_data_axpy_hypre(PASE_SCALAR a, void *x, void *y)
{
  HYPRE_ParVectorAxpy(a, (HYPRE_ParVector)x, (HYPRE_ParVector)y);
}

void
PASE_Vector_data_scale_hypre(PASE_SCALAR a, void *x)
{
  HYPRE_ParVectorScale(a, (HYPRE_ParVector)x);
}

void
PASE_Vector_data_get_global_nrow_hypre(void *x, PASE_INT *nrow)
{
  *nrow = hypre_ParVectorGlobalSize((HYPRE_ParVector)x);
}

#endif
