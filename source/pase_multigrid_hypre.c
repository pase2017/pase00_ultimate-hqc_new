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
  HYPRE_BoomerAMGSetInterpType(amg_solver, 0 );
  HYPRE_BoomerAMGSetPMaxElmts(amg_solver, 0 );
  HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
  //HYPRE_BoomerAMGSetMaxLevels(amg_solver, param->max_level);  /* maximum number of levels */
  //HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);          /* G-S/Jacobi hybrid relaxation */
  //HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);         /* uses C/F relaxation */
  //HYPRE_BoomerAMGSetNumSweeps(amg_solver, 1);          /* Sweeeps on each level */
  //HYPRE_BoomerAMGSetTol(amg_solver, 1e-7);             /* conv. tolerance */
  HYPRE_BoomerAMGSetMinCoarseSize(amg_solver, param->min_coarse_size);

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

  *num_level = hypre_ParAMGDataNumLevels(amg_data_hypre);
  HYPRE_ParCSRMatrix *A_hypre, *P_hypre;
  A_hypre = hypre_ParAMGDataAArray(amg_data_hypre);
  P_hypre = hypre_ParAMGDataPArray(amg_data_hypre);
  HYPRE_ParCSRMatrix *R_hypre = hypre_CTAlloc(HYPRE_ParCSRMatrix, *num_level-1);

  PASE_INT N_H = hypre_ParCSRMatrixGlobalNumRows(A_hypre[*num_level-1]);
  PASE_INT myid = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if(myid == 0) {
    printf ( "The number of levels = %d\n", *num_level );
    printf("The dim of the coarsest space is %d.\n", N_H );
  }

  PASE_INT i;
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
