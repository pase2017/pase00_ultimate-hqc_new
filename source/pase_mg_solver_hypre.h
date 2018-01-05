#ifndef __PASE_MG_SOLVER_HYPRE_H__
#define __PASE_MG_SOLVER_HYPRE_H__

#include "pase_config.h"
#include "pase_mg_solver.h"

#if PASE_USE_HYPRE

PASE_INT PASE_Mg_get_initial_vector_by_coarse_grid_hypre(void *mg_solver);
PASE_INT PASE_Mg_get_initial_vector_by_coarse_grid_lobpcg_amg_hypre(void *mg_solver);
PASE_INT PASE_Mg_get_initial_vector_by_full_multigrid_hypre(void *mg_solver);
PASE_INT PASE_Mg_presmoothing_by_pcg_hypre(void *mg_solver);
PASE_INT PASE_Mg_presmoothing_by_pcg_amg_hypre(void *mg_solver);
PASE_INT PASE_Mg_presmoothing_by_amg_hypre(void *mg_solver);
PASE_INT PASE_Mg_presmoothing_by_pcg_aux_hypre(void *mg_solver);
PASE_INT PASE_Mg_direct_solve_by_lobpcg_aux_hypre(void *mg_solver);

#endif

#endif
