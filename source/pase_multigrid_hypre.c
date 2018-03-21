#include "pase_multigrid_hypre.h"

#if PASE_USE_HYPRE
#include "HYPRE_utilities.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"

void         
PASE_Multigrid_get_amg_array_hypre
    (void *A, PASE_PARAMETER param, 
     void ***A_array, 
     void ***P_array, 
     void ***R_array, 
     PASE_INT *num_level, 
     void **amg_data)
{            
  HYPRE_Solver amg_solver;
  HYPRE_BoomerAMGCreate(&amg_solver);

  /* Set some parameters (See Reference Manual for more parameters) */
  HYPRE_BoomerAMGSetPrintLevel(amg_solver, 1);         /* print solve info + parameters */
  //HYPRE_BoomerAMGSetInterpType(amg_solver, 0 );
  //HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0 );
  HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
  HYPRE_BoomerAMGSetOldDefault(amg_solver);
  HYPRE_BoomerAMGSetRelaxType(amg_solver, 6);    
  //HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);         /* uses C/F relaxation */
  HYPRE_BoomerAMGSetNumSweeps(amg_solver, 1);          /* Sweeeps on each level */
  HYPRE_BoomerAMGSetTol(amg_solver, 0.0);             /* conv. tolerance */
  HYPRE_BoomerAMGSetMaxIter(amg_solver, 1);
  HYPRE_BoomerAMGSetMaxLevels(amg_solver, 16);  /* maximum number of levels */
  //HYPRE_BoomerAMGSetMinCoarseSize(amg_solver, param->min_coarse_size);

  /* Now setup */
  HYPRE_ParCSRMatrix parcsr_A     = (HYPRE_ParCSRMatrix)A;
  MPI_Comm           comm         = hypre_ParCSRMatrixComm(parcsr_A);
  PASE_INT           global_size  = hypre_ParCSRMatrixGlobalNumRows(parcsr_A);
  PASE_INT          *partitioning = NULL;
  HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_A, &partitioning);

  HYPRE_ParVector    par_b        = hypre_ParVectorCreate(comm, global_size, partitioning);
  HYPRE_ParVectorInitialize(par_b);
  hypre_ParVectorSetPartitioningOwner(par_b, 1);
  HYPRE_ParVector    par_x        = hypre_ParVectorCreate(comm, global_size, partitioning);
  HYPRE_ParVectorInitialize(par_x);
  hypre_ParVectorSetPartitioningOwner(par_x, 0);

  HYPRE_BoomerAMGSetup(amg_solver, parcsr_A, par_b, par_x);
  hypre_ParAMGData *amg_data_hypre = (hypre_ParAMGData*) amg_solver;
  HYPRE_ParCSRMatrix *A_hypre, *P_hypre;
  A_hypre = hypre_ParAMGDataAArray(amg_data_hypre);
  P_hypre = hypre_ParAMGDataPArray(amg_data_hypre);

  PASE_INT amg_levels = hypre_ParAMGDataNumLevels(amg_data_hypre);
  *num_level = (amg_levels < param->max_level)? amg_levels : param->max_level;
  PASE_INT i   = 0;
  PASE_INT N_H = hypre_ParCSRMatrixGlobalNumRows(A_hypre[*num_level-1]);
  while( N_H < param->min_coarse_size) {
    *num_level = *num_level - 1;
    N_H = hypre_ParCSRMatrixGlobalNumRows(A_hypre[*num_level-1]);
  }
  PASE_Printf(MPI_COMM_WORLD, "The number of levels of AMG   = %d\n", amg_levels);
  PASE_Printf(MPI_COMM_WORLD, "The number of levels for PASE = %d\n", *num_level);
  PASE_Printf(MPI_COMM_WORLD, "The dim of the coarsest space is %d.\n", N_H );

  HYPRE_ParCSRMatrix *R_hypre = hypre_CTAlloc(HYPRE_ParCSRMatrix, *num_level-1);
  for(i = 0; i < *num_level-1; i++) {
    hypre_ParCSRMatrixTranspose(P_hypre[i], &(R_hypre[i]), 1);
  }

  *A_array  = (void**)A_hypre;
  *P_array  = (void**)P_hypre;
  *R_array  = (void**)R_hypre;
  *amg_data = (void*)amg_solver;

  HYPRE_ParVectorDestroy(par_b);
  HYPRE_ParVectorDestroy(par_x);
}            

void
PASE_Multigrid_destroy_amg_data_hypre(void *amg_data)
{
  HYPRE_BoomerAMGDestroy((HYPRE_Solver)amg_data);
}

#endif
