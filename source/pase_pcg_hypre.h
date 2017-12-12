#ifndef __PASE_PCG_HYPRE_H__
#define __PASE_PCG_HYPRE_H__

#include <mpi.h>
#include "pase_config.h"

#if PASE_USE_HYPRE
#include "_hypre_parcsr_ls.h"
#include "HYPRE_MatvecFunctions.h"
#include "temp_multivector.h"

PASE_INT PASE_Pcg_comm_info(void   *A, PASE_INT *my_id, PASE_INT *num_procs);
void *PASE_Pcg_create_vector(void *vvector );
PASE_INT PASE_Pcg_destroy_vector(void *vvector );
PASE_INT PASE_Pcg_matvec(void *matvec_data, 
	                 PASE_SCALAR alpha, 
			 void *A, 
			 void *x, 
			 PASE_SCALAR  beta, 
			 void *y );
PASE_REAL PASE_Pcg_inner_product(void *x, void *y );
PASE_INT PASE_Pcg_copy_vector(void *x, void *y );
PASE_INT PASE_Pcg_clear_vector(void *x );
PASE_INT PASE_Pcg_scale_vector(PASE_SCALAR  alpha, void *x );
PASE_INT PASE_Pcg_add_vector(PASE_SCALAR alpha, void *x, void *y );
PASE_INT PASE_Pcg_identity(void *vdata, void *A, void *b, void *x );
PASE_INT PASE_Pcg_set_random_value(void* v, PASE_INT seed );
PASE_INT PASE_Pcg_create(MPI_Comm comm, HYPRE_Solver *solver );

PASE_INT PASE_Pcg_comm_info_aux(void *A, PASE_INT *my_id, PASE_INT *num_procs);
void* PASE_Pcg_create_vector_aux(void *x);
PASE_INT PASE_Pcg_destroy_vector_aux(void *x);
PASE_INT PASE_Pcg_matvec_aux(void        *matvec_data,
                             PASE_SCALAR  alpha,
                             void        *A,
                             void        *x,
                             PASE_SCALAR beta,
                             void        *y);
PASE_REAL PASE_Pcg_inner_product_aux(void *x, void *y);
PASE_INT PASE_Pcg_copy_vector_aux(void *x, void *y);
PASE_INT PASE_Pcg_clear_vector_aux(void *x);
PASE_INT PASE_Pcg_scale_vector_aux(PASE_SCALAR alpha, void *x);
PASE_INT PASE_Pcg_add_vector_aux(PASE_SCALAR alpha, void *x, void *y );
PASE_INT PASE_Pcg_identity_aux(void *vdata, void *A, void *b, void *x );
PASE_INT PASE_Pcg_set_random_value_aux( void* v, PASE_INT seed);

PASE_INT PASE_Pcg_create_aux(MPI_Comm comm, HYPRE_Solver *solver);


#endif

#endif
