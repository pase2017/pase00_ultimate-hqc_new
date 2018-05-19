#include <stdlib.h>
#include <math.h>
#include "pase_lobpcg_hypre.h"
#include "pase_vector.h"
#include "pase_matrix.h"
#include "pase_aux_vector.h"
#include "pase_aux_matrix.h"

#if PASE_USE_HYPRE

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

#endif
