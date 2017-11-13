#ifndef __PASE_MG_SOLVER_H__
#define __PASE_MG_SOLVER_H__

#include "pase_multigrid.h" 
#include "pase_config.h"

typedef struct PASE_MG_FUNCTION_PRIVATE_ { 
    PASE_INT    (*solve_directly)      (void *solver); 
    PASE_INT    (*presmoothing)      (void *solver); 
    PASE_INT    (*postsmoothing)     (void *solver); 
    PASE_INT    (*presmoothing_aux) (void *solver); 
    PASE_INT    (*postsmoothing_aux)(void *solver); 
} PASE_MG_FUNCTION_PRIVATE;
typedef PASE_MG_FUNCTION_PRIVATE * PASE_MG_FUNCTION;

typedef struct PASE_MG_SOLVER_PRIVATE_ {
   PASE_INT 	block_size;
   PASE_INT     pre_iter;
   PASE_INT     post_iter;
   PASE_INT 	max_iter;
   PASE_INT     max_level;
   PASE_INT     cur_level;
   PASE_REAL   	rtol;
   PASE_REAL   	atol;
   PASE_REAL    r_norm;
   PASE_INT     num_converged;
   PASE_INT     num_iter;
   PASE_INT     print_level; 

   PASE_SCALAR *eigenvalues;
   PASE_SCALAR *exact_eigenvalues;
   PASE_VECTOR *u;
   PASE_AUX_VECTOR    	**aux_u;

   PASE_MULTIGRID    multigrid;
   PASE_MG_FUNCTION  function;
} PASE_MG_SOLVER_PRIVATE; 
typedef PASE_MG_SOLVER_PRIVATE * PASE_MG_SOLVER;

PASE_MG_SOLVER PASE_Mg_solver_create_by_multigrid(PASE_MULTIGRID multigrid);
PASE_INT PASE_Mg_solver_destroy(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_set_up(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_solve(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_iteration(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_error_estimate(PASE_MG_SOLVER solver);
//PASE_INT PASE_Mg_restart(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_presmoothing(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_pre_set_up(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_postsmoothing(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_post_correction(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_set_aux_matrix(PASE_MG_SOLVER solver);
PASE_INT PASE_Get_initial_vector(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_presmoothing_by_pcg_hypre(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_presmoothing_by_pcg_aux_hypre(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_solve_directly_by_lobpcg_aux_hypre(PASE_MG_SOLVER solver);








#endif
