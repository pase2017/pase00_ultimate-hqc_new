#include <stdlib.h>
#include <math.h>
#include "pase_pcg_hypre.h"
#include "pase_vector.h"
#include "pase_matrix.h"
#include "pase_aux_vector.h"
#include "pase_aux_matrix.h"

#if PASE_USE_HYPRE

PASE_INT
PASE_Pcg_comm_info(void *A, PASE_INT *my_id, PASE_INT *num_procs)
{
   MPI_Comm comm = PASE_Matrix_get_mpi_comm((PASE_MATRIX)A);
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
    PASE_Vector_axpy(alpha, (PASE_VECTOR)x, (PASE_VECTOR)y);
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



//AUX_PCG
PASE_INT
PASE_Pcg_comm_info_aux(void *A, PASE_INT *my_id, PASE_INT *num_procs)
{
   MPI_Comm comm;
   PASE_Aux_matrix_get_mpi_comm((PASE_AUX_MATRIX)A, &comm);
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
    PASE_Aux_vector_axpy(alpha, (PASE_AUX_VECTOR)x, (PASE_AUX_VECTOR)y);
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

#endif
