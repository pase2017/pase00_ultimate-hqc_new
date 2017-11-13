/*
 * =====================================================================================
 *
 *       Filename:  pase_pcg.c
 *
 *    Description:  PASE_ParCSRMatrix下PCG求解线性方程组
 *
 *        Version:  1.0
 *        Created:  2017年09月08日 15时41分38秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include <math.h>
#include "pase_pcg.h"
#include "pase_vector.h"
#include "pase_matrix.h"
#include "pase_aux_vector.h"
#include "pase_aux_matrix.h"

PASE_INT
PASE_Pcg_comm_info(void *A, PASE_INT *my_id, PASE_INT *num_procs)
{
   MPI_Comm comm = PASE_Matrix_get_comm_info((PASE_MATRIX)A);
   hypre_MPI_Comm_size(comm, num_procs);
   hypre_MPI_Comm_rank(comm, my_id);
   return 0;
}

void*
PASE_Pcg_create_vector(void *x )
{
   return (void*)PASE_Vector_create_by_vector((PASE_VECTOR)x);
}

PASE_INT
PASE_Pcg_destroy_vector(void *x)
{
   PASE_Vector_destroy((PASE_VECTOR)x);
   return 0;
}

PASE_INT
PASE_Pcg_matvec(void        *matvec_data,
                PASE_SCALAR  alpha,
                void        *A,
                void        *x,
                PASE_SCALAR beta,
                void        *y)
{
    PASE_Matrix_multiply_vector_general(alpha, (PASE_MATRIX)A, (PASE_VECTOR)x, beta, (PASE_VECTOR)y);
    return 0;
}

PASE_REAL
PASE_Pcg_inner_product(void *x, void *y)
{
   PASE_REAL prod;
   PASE_Vector_inner_product((PASE_VECTOR)x, (PASE_VECTOR)y, &prod); 
   return prod;
}

PASE_INT
PASE_Pcg_copy_vector(void *x, void *y)
{
    PASE_Vector_copy((PASE_VECTOR)x, (PASE_VECTOR)y);
    return 0;
}

PASE_INT
PASE_Pcg_clear_vector(void *x)
{
    PASE_Vector_set_constant_value((PASE_VECTOR)x, 0.0);
    return 0;
}

PASE_INT
PASE_Pcg_scale_vector(PASE_SCALAR alpha, void *x)
{
    PASE_Vector_scale(alpha, (PASE_VECTOR)x);
    return 0;
}

PASE_INT
PASE_Pcg_add_vector(PASE_SCALAR alpha, void *x, void *y )
{
    PASE_Vector_add_vector(alpha, (PASE_VECTOR)x, (PASE_VECTOR)y);
    return 0;
}

PASE_INT
PASE_Pcg_identity( void *vdata, void *A, void *b, void *x )
{
   return PASE_Pcg_copy_vector(b, x);
}

PASE_INT
PASE_Pcg_set_random_value( void* v, PASE_INT seed ) 
{
    PASE_Vector_set_random_value((PASE_VECTOR)v, seed);
    return 0;
}

/* 这里是否需要都将函数名封装成pase呢? */
PASE_INT
PASE_Pcg_create(MPI_Comm comm, HYPRE_Solver *solver)
{
   hypre_PCGFunctions * pcg_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree, PASE_Pcg_comm_info,
         PASE_Pcg_create_vector, PASE_Pcg_destroy_vector, 
	 hypre_ParKrylovMatvecCreate,
         PASE_Pcg_matvec, hypre_ParKrylovMatvecDestroy,
         PASE_Pcg_inner_product, PASE_Pcg_copy_vector,
         PASE_Pcg_clear_vector,
         PASE_Pcg_scale_vector, PASE_Pcg_add_vector,
         hypre_ParKrylovIdentitySetup, PASE_Pcg_identity);
   *solver = ( (HYPRE_Solver) hypre_PCGCreate( pcg_functions ) );

   return hypre_error_flag;
}

PASE_INT 
PASE_Lobpcg_setup_interpreter( mv_InterfaceInterpreter* i)
{
  /* Vector part */

  i->CreateVector    = PASE_Pcg_create_vector;
  i->DestroyVector   = PASE_Pcg_destroy_vector; 
  i->InnerProd       = PASE_Pcg_inner_product; 
  i->CopyVector      = PASE_Pcg_copy_vector;
  i->ClearVector     = PASE_Pcg_clear_vector;
  i->SetRandomValues = PASE_Pcg_set_random_value;
  i->ScaleVector     = PASE_Pcg_scale_vector;
  i->Axpy            = PASE_Pcg_add_vector;

  /* Multivector part */

  i->CreateMultiVector = mv_TempMultiVectorCreateFromSampleVector;
  i->CopyCreateMultiVector = mv_TempMultiVectorCreateCopy;
  i->DestroyMultiVector = mv_TempMultiVectorDestroy;

  i->Width = mv_TempMultiVectorWidth;
  i->Height = mv_TempMultiVectorHeight;
  i->SetMask = mv_TempMultiVectorSetMask;
  i->CopyMultiVector = mv_TempMultiVectorCopy;
  i->ClearMultiVector = mv_TempMultiVectorClear;
  i->SetRandomVectors = mv_TempMultiVectorSetRandom;
  i->MultiInnerProd = mv_TempMultiVectorByMultiVector;
  i->MultiInnerProdDiag = mv_TempMultiVectorByMultiVectorDiag;
  i->MultiVecMat = mv_TempMultiVectorByMatrix;
  i->MultiVecMatDiag = mv_TempMultiVectorByDiagonal;
  i->MultiAxpy = mv_TempMultiVectorAxpy;
  i->MultiXapy = mv_TempMultiVectorXapy;
  i->Eval = mv_TempMultiVectorEval;

  return 0;
}

PASE_INT 
PASE_Lobpcg_setup_matvec(HYPRE_MatvecFunctions* mv)
{
  mv->MatvecCreate       = hypre_ParKrylovMatvecCreate;
  mv->Matvec             = PASE_Pcg_matvec;
  mv->MatvecDestroy      = hypre_ParKrylovMatvecDestroy;

  mv->MatMultiVecCreate  = NULL;
  mv->MatMultiVec        = NULL;
  mv->MatMultiVecDestroy = NULL;

  return 0;
}






//AUX_PCG
PASE_INT
PASE_Pcg_comm_info_aux(void *A, PASE_INT *my_id, PASE_INT *num_procs)
{
   MPI_Comm comm;
   PASE_Aux_matrix_get_comm_info((PASE_AUX_MATRIX)A, &comm);
   hypre_MPI_Comm_size(comm, num_procs);
   hypre_MPI_Comm_rank(comm, my_id);
   return 0;
}

void*
PASE_Pcg_create_vector_aux(void *x)
{
   return (void*)PASE_Aux_vector_create_by_aux_vector((PASE_AUX_VECTOR)x);
}

PASE_INT
PASE_Pcg_destroy_vector_aux(void *x)
{
   PASE_Aux_vector_destroy((PASE_AUX_VECTOR)x);
   return 0;
}

PASE_INT
PASE_Pcg_matvec_aux(void        *matvec_data,
                     PASE_SCALAR  alpha,
                     void        *A,
                     void        *x,
                     PASE_SCALAR beta,
                     void        *y)
{
    PASE_Aux_matrix_multiply_aux_vector_general(alpha, (PASE_AUX_MATRIX)A, (PASE_AUX_VECTOR)x, beta, (PASE_AUX_VECTOR)y);
    return 0;
}

PASE_REAL
PASE_Pcg_inner_product_aux(void *x, void *y)
{
   PASE_REAL prod;
   PASE_Aux_vector_inner_product((PASE_AUX_VECTOR)x, (PASE_AUX_VECTOR)y, &prod); 
   return prod;
}

PASE_INT
PASE_Pcg_copy_vector_aux(void *x, void *y)
{
    PASE_Aux_vector_copy((PASE_AUX_VECTOR)x, (PASE_AUX_VECTOR)y);
    return 0;
}

PASE_INT
PASE_Pcg_clear_vector_aux(void *x)
{
    PASE_Aux_vector_set_constant_value((PASE_AUX_VECTOR)x, 0.0);
    return 0;
}

PASE_INT
PASE_Pcg_scale_vector_aux(PASE_SCALAR alpha, void *x)
{
    PASE_Aux_vector_scale(alpha, (PASE_AUX_VECTOR)x);
    return 0;
}

PASE_INT
PASE_Pcg_add_vector_aux(PASE_SCALAR alpha, void *x, void *y )
{
    PASE_Aux_vector_add(alpha, (PASE_AUX_VECTOR)x, (PASE_AUX_VECTOR)y);
    return 0;
}

PASE_INT
PASE_Pcg_identity_aux(void *vdata, void *A, void *b, void *x )
{
   return PASE_Pcg_copy_vector_aux(b, x);
}

PASE_INT
PASE_Pcg_set_random_value_aux( void* v, PASE_INT seed ) 
{
    PASE_Aux_vector_set_random_value((PASE_AUX_VECTOR)v, seed);
    return 0;
}



/* 这里是否需要都将函数名封装成pase呢? */
PASE_INT
PASE_Pcg_create_aux(MPI_Comm comm, HYPRE_Solver *solver)
{
   hypre_PCGFunctions * pcg_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree, PASE_Pcg_comm_info_aux,
         PASE_Pcg_create_vector_aux, PASE_Pcg_destroy_vector_aux, 
	 hypre_ParKrylovMatvecCreate,
         PASE_Pcg_matvec_aux, hypre_ParKrylovMatvecDestroy,
         PASE_Pcg_inner_product_aux, PASE_Pcg_copy_vector_aux,
         PASE_Pcg_clear_vector_aux,
         PASE_Pcg_scale_vector_aux, PASE_Pcg_add_vector_aux,
         hypre_ParKrylovIdentitySetup, 
	 PASE_Pcg_identity_aux);
   *solver = ( (HYPRE_Solver) hypre_PCGCreate( pcg_functions ) );

   return hypre_error_flag;
}

PASE_INT 
PASE_Lobpcg_setup_interpreter_aux( mv_InterfaceInterpreter* i)
{
  /* Vector part */

  i->CreateVector    = PASE_Pcg_create_vector_aux;
  i->DestroyVector   = PASE_Pcg_destroy_vector_aux; 
  i->InnerProd       = PASE_Pcg_inner_product_aux; 
  i->CopyVector      = PASE_Pcg_copy_vector_aux;
  i->ClearVector     = PASE_Pcg_clear_vector_aux;
  i->SetRandomValues = PASE_Pcg_set_random_value_aux;
  i->ScaleVector     = PASE_Pcg_scale_vector_aux;
  i->Axpy            = PASE_Pcg_add_vector_aux;

  /* Multivector part */

  i->CreateMultiVector = mv_TempMultiVectorCreateFromSampleVector;
  i->CopyCreateMultiVector = mv_TempMultiVectorCreateCopy;
  i->DestroyMultiVector = mv_TempMultiVectorDestroy;

  i->Width = mv_TempMultiVectorWidth;
  i->Height = mv_TempMultiVectorHeight;
  i->SetMask = mv_TempMultiVectorSetMask;
  i->CopyMultiVector = mv_TempMultiVectorCopy;
  i->ClearMultiVector = mv_TempMultiVectorClear;
  i->SetRandomVectors = mv_TempMultiVectorSetRandom;
  i->MultiInnerProd = mv_TempMultiVectorByMultiVector;
  i->MultiInnerProdDiag = mv_TempMultiVectorByMultiVectorDiag;
  i->MultiVecMat = mv_TempMultiVectorByMatrix;
  i->MultiVecMatDiag = mv_TempMultiVectorByDiagonal;
  i->MultiAxpy = mv_TempMultiVectorAxpy;
  i->MultiXapy = mv_TempMultiVectorXapy;
  i->Eval = mv_TempMultiVectorEval;

  return 0;
}

PASE_INT 
PASE_Lobpcg_setup_matvec_aux(HYPRE_MatvecFunctions* mv)
{
  mv->MatvecCreate       = hypre_ParKrylovMatvecCreate;
  mv->Matvec             = PASE_Pcg_matvec_aux;
  mv->MatvecDestroy      = hypre_ParKrylovMatvecDestroy;

  mv->MatMultiVecCreate  = NULL;
  mv->MatMultiVec        = NULL;
  mv->MatMultiVecDestroy = NULL;

  return 0;
}

