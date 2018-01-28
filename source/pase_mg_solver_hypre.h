#ifndef __PASE_MG_SOLVER_HYPRE_H__
#define __PASE_MG_SOLVER_HYPRE_H__

#include "pase_config.h"
#include "pase_mg_solver.h"

#if PASE_USE_HYPRE

PASE_INT PASE_Mg_get_initial_vector_by_coarse_grid_hypre(void *mg_solver);
PASE_INT PASE_Mg_get_initial_vector_by_coarse_grid_lobpcg_amg_hypre(void *mg_solver);
PASE_INT PASE_Mg_get_initial_vector_by_full_multigrid_hypre(void *mg_solver);
PASE_INT PASE_Mg_direct_solve_by_lobpcg_aux_hypre(void *mg_solver);

PASE_INT PASE_Mg_smoothing_by_pcg_hypre(void *mg_solver, char *PreOrPost);
PASE_INT PASE_Mg_smoothing_by_pcg_amg_hypre(void *mg_solver, char *PreOrPost);
PASE_INT PASE_Mg_smoothing_by_amg_hypre(void *mg_solver, char *PreOrPost);
PASE_INT PASE_Mg_smoothing_by_pcg_aux_hypre(void *mg_solver, char *PreOrPost);
PASE_INT PASE_Mg_presmoothing_by_pcg_hypre(void *mg_solver);
PASE_INT PASE_Mg_postsmoothing_by_pcg_hypre(void *mg_solver);
PASE_INT PASE_Mg_presmoothing_by_pcg_amg_hypre(void *mg_solver);
PASE_INT PASE_Mg_postsmoothing_by_pcg_amg_hypre(void *mg_solver);
PASE_INT PASE_Mg_presmoothing_by_amg_hypre(void *mg_solver);
PASE_INT PASE_Mg_postsmoothing_by_amg_hypre(void *mg_solver);
PASE_INT PASE_Mg_presmoothing_by_pcg_aux_hypre(void *mg_solver);
PASE_INT PASE_Mg_postsmoothing_by_pcg_aux_hypre(void *mg_solver);

PASE_INT PASE_Linear_solve_by_amg_hypre(PASE_MATRIX A, PASE_VECTOR *b, PASE_VECTOR *x, PASE_INT n, PASE_REAL tol, PASE_INT max_iter, void *amg_data);

PASE_INT PASE_Mg_get_initial_vector_by_full_multigrid_hypre_for_guangji(void *mg_solver);
PASE_INT PASE_Mg_smoothing_by_pcg_amg_hypre_for_guangji(void *mg_solver);
#endif

#endif
