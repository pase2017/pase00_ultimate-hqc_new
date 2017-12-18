#ifndef __PASE_MG_SOLVER_H__
#define __PASE_MG_SOLVER_H__

#include "pase_multigrid.h" 
#include "pase_config.h"

typedef struct PASE_MG_FUNCTION_PRIVATE_ { 
    PASE_INT    (*get_initial_vector) (void *solver);
    PASE_INT    (*direct_solve)       (void *solver); 
    PASE_INT    (*presmoothing)       (void *solver); 
    PASE_INT    (*postsmoothing)      (void *solver); 
    PASE_INT    (*presmoothing_aux)   (void *solver); 
    PASE_INT    (*postsmoothing_aux)  (void *solver); 
} PASE_MG_FUNCTION_PRIVATE;
typedef PASE_MG_FUNCTION_PRIVATE * PASE_MG_FUNCTION;

typedef struct PASE_MG_SOLVER_PRIVATE_ {
   PASE_INT 	block_size;
   PASE_INT     actual_block_size;
   PASE_INT     pre_iter;
   PASE_INT     post_iter;
   PASE_INT 	max_iter;
   PASE_INT     max_level;
   PASE_INT     cur_level;
   PASE_REAL   	rtol;
   PASE_REAL   	atol;
   PASE_REAL    *r_norm;
   PASE_INT     num_converged;
   PASE_INT     last_num_converged;
   PASE_INT     num_iter;
   PASE_INT     print_level; 

   PASE_REAL    set_up_time;
   PASE_REAL    smooth_time;
   PASE_REAL    set_aux_time;
   PASE_REAL    prolong_time;
   PASE_REAL    direct_solve_time;
   PASE_REAL    total_solve_time;

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
PASE_INT PASE_Mg_iteration_two_gird(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_set_aux_matrix_two_grid(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_prolong_two_grid(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_prolong_general(PASE_MG_SOLVER solver,  PASE_INT i, PASE_VECTOR u_i, PASE_INT j, PASE_VECTOR u_j);
PASE_INT PASE_Mg_error_estimate(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_print(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_presmoothing(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_pre_set_up(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_postsmoothing(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_direct_solve(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_prolong(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_set_aux_matrix(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_orthogonalize(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_set_exact_eigenvalues(PASE_MG_SOLVER solver, PASE_SCALAR *exact_eigenvalues);

PASE_INT PASE_Mg_set_block_size(PASE_MG_SOLVER solver, PASE_INT block_size);
PASE_INT PASE_Mg_set_max_iteration(PASE_MG_SOLVER solver, PASE_INT max_iter);
PASE_INT PASE_Mg_set_max_pre_iteration(PASE_MG_SOLVER solver, PASE_INT pre_iter);
PASE_INT PASE_Mg_set_max_post_iteration(PASE_MG_SOLVER solver, PASE_INT post_iter);
PASE_INT PASE_Mg_set_atol(PASE_MG_SOLVER solver, PASE_REAL atol);
PASE_INT PASE_Mg_set_rtol(PASE_MG_SOLVER solver, PASE_REAL rtol);
PASE_INT PASE_Mg_set_print_level(PASE_MG_SOLVER solver, PASE_INT print_level);

PASE_INT PASE_Mg_presmoothing_by_cg(void *mg_solver);
PASE_INT PASE_Mg_presmoothing_by_cg_aux(void *mg_solver);
PASE_INT PASE_Linear_solve_by_cg(PASE_MATRIX A, PASE_VECTOR b, PASE_VECTOR x, PASE_REAL tol, PASE_INT max_iter);
PASE_INT PASE_Linear_solve_by_cg_aux(PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_b, PASE_AUX_VECTOR aux_x, PASE_REAL tol, PASE_INT max_iter);
//PASE_INT PASE_Mg_solve_directly_by_IRA(void *mg_solver);

PASE_INT PASE_Mg_coarsest_aux_matrix_set(PASE_MG_SOLVER solver);
PASE_INT PASE_Mg_coarsest_aux_matrix_create(PASE_MG_SOLVER solver, PASE_AUX_MATRIX *aux_A, PASE_MATRIX A_H);
PASE_INT PASE_Mg_restrict(PASE_MG_SOLVER solver, PASE_INT i, PASE_VECTOR u_i, PASE_INT j, PASE_VECTOR u_j);






#endif
