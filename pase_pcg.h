/*
 * =====================================================================================
 *
 *       Filename:  pash.h
 *
 *    Description:  后期可以考虑将行参类型都变成void *, 以方便修改和在不同计算机上调试
 *                  一般而言, 可以让用户调用的函数以PASE_开头, 内部函数以pase_开头
 *
 *        Version:  1.0
 *        Created:  2017年08月29日 14时15分22秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  LIYU 
 *   Organization:  LSEC
 *
 * =====================================================================================
 */

#ifndef _pase_pcg_h_
#define _pase_pcg_h_

#include <mpi.h>
#include "pase_config.h"

#include "_hypre_parcsr_ls.h"
#include "HYPRE_MatvecFunctions.h"
#include "temp_multivector.h"

#ifdef __cplusplus
extern "C" {
#endif

PASE_INT PASE_Pcg_comm_info( void   *A, PASE_INT *my_id, PASE_INT *num_procs);
void *PASE_Pcg_create_vector( void *vvector );
PASE_INT PASE_Pcg_destroy_vector( void *vvector );
PASE_INT PASE_Pcg_matvec(void *matvec_data, 
	                 PASE_SCALAR alpha, 
			 void *A, 
			 void *x, 
			 PASE_SCALAR  beta, 
			 void *y );
PASE_REAL PASE_Pcg_inner_product( void *x, void *y );
PASE_INT PASE_Pcg_copy_vector( void *x, void *y );
PASE_INT PASE_Pcg_clear_vector( void *x );
PASE_INT PASE_Pcg_scale_vector( PASE_SCALAR  alpha, void *x );
PASE_INT PASE_Pcg_add_vector( PASE_SCALAR alpha, void *x, void *y );
PASE_INT PASE_Pcg_identity( void *vdata, void *A, void *b, void *x );
PASE_INT PASE_Pcg_set_random_value( void* v, PASE_INT seed );

PASE_INT PASE_Pcg_create( MPI_Comm comm, HYPRE_Solver *solver );
PASE_INT PASE_Lobpcg_setup_interpreter( mv_InterfaceInterpreter* i);
PASE_INT PASE_Lobpcg_setup_matvec(HYPRE_MatvecFunctions* mv);


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
PASE_INT PASE_Lobpcg_setup_interpreter_aux( mv_InterfaceInterpreter* i);
PASE_INT PASE_Lobpcg_setup_matvec_aux(HYPRE_MatvecFunctions* mv);

PASE_INT hypre_LOBPCGSetup( void *pcg_vdata, void *A, void *b, void *x );
PASE_INT hypre_LOBPCGSetupB( void *pcg_vdata, void *B, void *x );



#ifdef __cplusplus
}
#endif

#endif
