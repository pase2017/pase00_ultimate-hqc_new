#include "pase_mg_solver_hypre.h"
#include "pase_mg_solver.h"
#include "pase_pcg_hypre.h"
#include "pase_lobpcg_hypre.h"

#if PASE_USE_HYPRE
#include "HYPRE_utilities.h"
#include "HYPRE_lobpcg.h"
#include "lobpcg.h"

PASE_INT
PASE_Mg_get_initial_vector_by_coarse_grid_hypre(void *mg_solver)
{
  PASE_MG_SOLVER solver        = (PASE_MG_SOLVER)mg_solver;
  HYPRE_Solver   lobpcg_solver = NULL; 
  PASE_INT       maxIterations = 50; 	        /* maximum number of iterations */
  PASE_INT       pcgMode       = 1;    	        /* use rhs as initial guess for inner pcg iterations */
  PASE_INT       verbosity     = 0;    	        /* print iterations info */
  PASE_REAL      atol 	       = solver->atol;	/* absolute tolerance (all eigenvalues) */
  PASE_REAL      rtol          = 1e-50;
  PASE_INT       lobpcgSeed    = 77;

  PASE_INT       myid          = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if(myid != 0) {
    verbosity = 0;
  }

  PASE_INT     max_level      = solver->max_level;        
  PASE_INT     block_size     = solver->block_size;       
  PASE_MATRIX  A              = solver->multigrid->A[max_level];
  PASE_MATRIX  B              = solver->multigrid->B[max_level];
  PASE_VECTOR  u_temp = PASE_Vector_create_by_matrix_and_vector_data_operator(A, solver->u[0]->ops); 

  PASE_VECTOR  x              = PASE_Vector_create_by_vector(u_temp);
  PASE_INT     max_block_size = solver->max_block_size;
  PASE_SCALAR *eigenvalues    = (PASE_SCALAR*)PASE_Malloc(max_block_size*sizeof(PASE_SCALAR));

  mv_InterfaceInterpreter *interpreter_Hh  = hypre_CTAlloc(mv_InterfaceInterpreter, 1);
  HYPRE_MatvecFunctions    matvec_fn_Hh;
  PASE_Lobpcg_setup_interpreter(interpreter_Hh);
  PASE_Lobpcg_setup_matvec(&matvec_fn_Hh);

  mv_MultiVectorPtr        eigenvectors_Hh = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, max_block_size, u_temp);
  mv_MultiVectorPtr        constraints_Hh  = NULL;
  mv_MultiVectorSetRandom(eigenvectors_Hh, lobpcgSeed);

  HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
  HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
  HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
  HYPRE_LOBPCGSetTol(lobpcg_solver, atol);
  HYPRE_LOBPCGSetRTol(lobpcg_solver, rtol);
  HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

  hypre_LOBPCGSetup(lobpcg_solver, A, u_temp, x);
  hypre_LOBPCGSetupB(lobpcg_solver, B, u_temp);
  HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);

  /* 根据初始的特征值分布，调整实际求解的特征值个数来提高收敛速度 */
  PASE_REAL *rate     = (PASE_REAL*)PASE_Malloc((max_block_size-block_size)*sizeof(PASE_REAL));
  PASE_INT   index    = 0;
  PASE_REAL  max_rate = 1.0;
  PASE_INT   i        = 0;
  while(i < (max_block_size-block_size) && index > -1) {
    rate[i] = fabs(eigenvalues[i+block_size-1]/eigenvalues[i+block_size]);
    if(rate[i] < 0.9) {
      solver->block_size = block_size + i;
      index = -1;
    } else if(max_rate > rate[i]) {
      max_rate = rate[i];
      index = i;
    }
    i++;
  }

  if(index > 0) {
    solver->block_size = block_size + index;
  }
  PASE_Printf(MPI_COMM_WORLD, "modified block_size = %d\n\n", solver->block_size);

#if 0
  if(solver->block_size != block_size) {
    PASE_VECTOR *u = (PASE_VECTOR*)PASE_Malloc(solver->block_size*sizeof(PASE_VECTOR));
    for(i = 0; i < solver->block_size; i++) {
      if(i < block_size) {
	u[i] = solver->u[i];
      } else {
	u[i] = PASE_Vector_create_by_vector(solver->u[0]);
      }
    }
    PASE_Free(solver->u);
    solver->u = u;
    PASE_Free(solver->eigenvalues);
    solver->eigenvalues = (PASE_SCALAR*)PASE_Malloc(solver->block_size*sizeof(PASE_SCALAR));
  }
#endif

  block_size              = solver->block_size;
  mv_TempMultiVector *tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
  PASE_VECTOR        *u_H = (PASE_VECTOR*)(tmp->vector);
  for(i = 0; i < block_size; i++) {
    PASE_Mg_prolong_general(solver, max_level, u_H[i], 0, solver->u[i]);
    solver->eigenvalues[i] = eigenvalues[i];
    PASE_Vector_destroy(u_H[i]);
  }
  for(i = block_size; i < max_block_size; i++) {
    PASE_Vector_destroy(solver->u[i]);
    PASE_Vector_destroy(u_H[i]);
  }

  if(solver->print_level > 1) {
    PASE_Printf(MPI_COMM_WORLD, "Get initial vector: ");
  }
  PASE_Mg_print_eigenvalue_of_current_level(solver);

  PASE_Free(rate);
  free((mv_TempMultiVector*)mv_MultiVectorGetData(eigenvectors_Hh));
  PASE_Free(eigenvectors_Hh);
  PASE_Free(interpreter_Hh);
  PASE_Vector_destroy(u_temp);
  PASE_Vector_destroy(x);

  PASE_Free(eigenvalues);
  PASE_Free(u_H);
  HYPRE_LOBPCGDestroy( lobpcg_solver);
  return 0;
}

PASE_INT
PASE_Mg_get_initial_vector_by_full_multigrid_hypre(void *mg_solver)
{
  PASE_MG_SOLVER solver        = (PASE_MG_SOLVER)mg_solver;
  HYPRE_Solver   lobpcg_solver = NULL; 
  PASE_INT       maxIterations = 100; 	        /* maximum number of iterations */
  PASE_INT       pcgMode       = 1;    	        /* use rhs as initial guess for inner pcg iterations */
  PASE_INT       verbosity     = 0;    	        /* print iterations info */
  PASE_REAL      atol 	       = solver->atol * 1e-2;	/* absolute tolerance (all eigenvalues) */
  PASE_REAL      rtol          = 1e-50;
  PASE_INT       lobpcgSeed    = 77;

  PASE_INT       myid          = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if(myid != 0) {
    verbosity = 0;
  }

  PASE_INT     max_level      = solver->max_level;        
  PASE_INT     block_size     = solver->block_size;       
  PASE_MATRIX  A              = solver->multigrid->A[max_level];
  PASE_MATRIX  B              = solver->multigrid->B[max_level];
  PASE_VECTOR  u_temp = PASE_Vector_create_by_matrix_and_vector_data_operator(A, solver->u[0]->ops); 

  PASE_VECTOR  x              = PASE_Vector_create_by_vector(u_temp);
  PASE_INT     max_block_size = ((2*block_size)<(block_size+5))?(2*block_size):(block_size+5);
  PASE_SCALAR *eigenvalues    = (PASE_SCALAR*)PASE_Malloc(max_block_size*sizeof(PASE_SCALAR));

  mv_InterfaceInterpreter *interpreter_Hh  = hypre_CTAlloc(mv_InterfaceInterpreter, 1);
  HYPRE_MatvecFunctions    matvec_fn_Hh;
  PASE_Lobpcg_setup_interpreter(interpreter_Hh);
  PASE_Lobpcg_setup_matvec(&matvec_fn_Hh);

  mv_MultiVectorPtr        eigenvectors_Hh = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, max_block_size, u_temp);
  mv_MultiVectorPtr        constraints_Hh  = NULL;
  mv_MultiVectorSetRandom(eigenvectors_Hh, lobpcgSeed);

  HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
  HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
  HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
  HYPRE_LOBPCGSetTol(lobpcg_solver, atol);
  HYPRE_LOBPCGSetRTol(lobpcg_solver, rtol);
  HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

  hypre_LOBPCGSetup(lobpcg_solver, A, u_temp, x);
  hypre_LOBPCGSetupB(lobpcg_solver, B, u_temp);
  HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);

  /* 根据初始的特征值分布，调整实际求解的特征值个数来提高收敛速度 */
  PASE_REAL *rate     = (PASE_REAL*)PASE_Malloc((max_block_size-block_size)*sizeof(PASE_REAL));
  PASE_INT   index    = 0;
  PASE_REAL  max_rate = 1.0;
  PASE_INT   i        = 0;
  while(i < (max_block_size-block_size) && index > -1) {
    rate[i] = fabs(eigenvalues[i+block_size-1]/eigenvalues[i+block_size]);
    if(rate[i] < 0.9) {
      solver->block_size = block_size + i;
      index = -1;
    } else if(max_rate > rate[i]) {
      max_rate = rate[i];
      index = i;
    }
    i++;
  }

  if(index > 0) {
    solver->block_size = block_size + index;
  }
  PASE_Printf(MPI_COMM_WORLD, "modified block_size = %d\n\n", solver->block_size);

  if(solver->block_size != block_size) {
    PASE_VECTOR *u = (PASE_VECTOR*)PASE_Malloc(solver->block_size*sizeof(PASE_VECTOR));
    for(i = 0; i < solver->block_size; i++) {
      if(i < block_size) {
	u[i] = solver->u[i];
      } else {
	u[i] = PASE_Vector_create_by_vector(solver->u[0]);
      }
    }
    PASE_Free(solver->u);
    solver->u = u;
    PASE_Free(solver->eigenvalues);
    solver->eigenvalues = (PASE_SCALAR*)PASE_Malloc(solver->block_size*sizeof(PASE_SCALAR));
  }

  block_size              = solver->block_size;
  mv_TempMultiVector *tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
  PASE_VECTOR        *u_H = (PASE_VECTOR*)(tmp->vector);
  for(i = 0; i < block_size; i++) {
    PASE_Mg_prolong_general(solver, max_level, u_H[i], 0, solver->u[i]);
    solver->eigenvalues[i] = eigenvalues[i];
  }
  for(i = 0; i < max_block_size; i++) {
    PASE_Vector_destroy(u_H[i]);
  }

  if(solver->print_level > 1) {
    PASE_Printf(MPI_COMM_WORLD, "Get initial vector: ");
  }
  PASE_Mg_print_eigenvalue_of_current_level(solver);

  PASE_Free(rate);
  free((mv_TempMultiVector*)mv_MultiVectorGetData(eigenvectors_Hh));
  PASE_Free(eigenvectors_Hh);
  PASE_Free(interpreter_Hh);
  PASE_Vector_destroy(u_temp);
  PASE_Vector_destroy(x);

  PASE_Free(eigenvalues);
  PASE_Free(u_H);
  HYPRE_LOBPCGDestroy( lobpcg_solver);
  return 0;
}

PASE_INT 
PASE_Mg_presmoothing_by_pcg_hypre(void *mg_solver)
{
  PASE_SCALAR    inner_A, inner_B;
  PASE_MG_SOLVER solver      = (PASE_MG_SOLVER)mg_solver;
  PASE_INT       cur_level   = solver->cur_level;
  PASE_INT       block_size  = solver->block_size;
  PASE_INT       nconv       = solver->nconv; 
  PASE_INT       i	     = 0;

  PASE_MATRIX    A           = solver->multigrid->A[cur_level];            
  PASE_MATRIX    B           = solver->multigrid->B[cur_level];            
  PASE_VECTOR   *u           = solver->u;
  PASE_SCALAR   *eigenvalues = solver->eigenvalues;
  PASE_VECTOR    rhs 	     = PASE_Vector_create_by_vector(u[0]);

  HYPRE_Solver   cg_solver   = NULL;
  PASE_Pcg_create(MPI_COMM_WORLD, &cg_solver);
  HYPRE_PCGSetMaxIter(cg_solver, solver->max_pre_iter); /* max iterations */
  HYPRE_PCGSetTol(cg_solver, 1.0e-50); 
  HYPRE_PCGSetTwoNorm(cg_solver, 1);                    /* use the two norm as the stopping criteria */
  HYPRE_PCGSetPrintLevel(cg_solver, 0); 
  HYPRE_PCGSetLogging(cg_solver, 1);                    /* needed to get run info later */
  hypre_PCGSetup(cg_solver, A, rhs, u[0]);

  for(i = nconv; i < block_size; i++) {
    PASE_Matrix_multiply_vector(B, u[i], rhs);
    PASE_Vector_scale(eigenvalues[i], rhs);
    hypre_PCGSolve(cg_solver, A, rhs, u[i]);

    PASE_Vector_inner_product_general(rhs, u[i], A, &inner_A);
    PASE_Vector_inner_product_general(rhs, u[i], B, &inner_B);
    eigenvalues[i] = inner_A / inner_B;
  }
  PASE_Mg_print_eigenvalue_of_current_level(solver);

  PASE_Vector_destroy(rhs);
  HYPRE_ParCSRPCGDestroy(cg_solver);
  return 0;
}

PASE_INT 
PASE_Mg_presmoothing_by_pcg_amg_hypre(void *mg_solver)
{
  PASE_SCALAR     inner_A, inner_B;
  PASE_MG_SOLVER  solver        = (PASE_MG_SOLVER)mg_solver;
  HYPRE_Solver    cg_solver     = NULL;
  PASE_INT        block_size	= solver->block_size;
  PASE_INT        nconv         = solver->nconv; 
  PASE_INT        i		= 0;

  PASE_MATRIX     A             = solver->multigrid->A[0];            
  PASE_MATRIX     B             = solver->multigrid->B[0];            
  PASE_VECTOR    *u             = solver->u;
  PASE_SCALAR    *eigenvalues   = solver->eigenvalues;
  PASE_VECTOR     rhs 	        = PASE_Vector_create_by_vector(u[0]);

  HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &cg_solver);
  HYPRE_PCGSetMaxIter(cg_solver, solver->max_pre_iter); /* max iterations */
  HYPRE_PCGSetTol(cg_solver, 1.0e-15); 
  HYPRE_PCGSetTwoNorm(cg_solver, 1); /* use the two norm as the st    opping criteria */
  HYPRE_PCGSetPrintLevel(cg_solver, 0); 
  HYPRE_PCGSetLogging(cg_solver, 1); /* needed to get run info lat    er */

  HYPRE_ParCSRPCGSetup(cg_solver, (HYPRE_ParCSRMatrix)(A->matrix_data), (HYPRE_ParVector)(rhs->vector_data), (HYPRE_ParVector)(u[i]->vector_data));
  HYPRE_PCGSetPrecond(cg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, solver->multigrid->amg_data);

  for(i = nconv; i < block_size; i++) {
    PASE_Matrix_multiply_vector(B, u[i], rhs);
    PASE_Vector_scale(eigenvalues[i], rhs);
    HYPRE_ParCSRPCGSolve(cg_solver, (HYPRE_ParCSRMatrix)(A->matrix_data), (HYPRE_ParVector)(rhs->vector_data), (HYPRE_ParVector)(u[i]->vector_data));

    PASE_Vector_inner_product_general(rhs, u[i], A, &inner_A);
    PASE_Vector_inner_product_general(rhs, u[i], B, &inner_B);
    eigenvalues[i] = inner_A / inner_B;
  }
  PASE_Mg_print_eigenvalue_of_current_level(solver);

  PASE_Vector_destroy(rhs);
  HYPRE_ParCSRPCGDestroy(cg_solver);
  return 0;
}

PASE_INT 
PASE_Mg_presmoothing_by_pcg_aux_hypre(void *mg_solver)
{
  PASE_SCALAR      inner_A, inner_B;
  PASE_MG_SOLVER   solver        = (PASE_MG_SOLVER)mg_solver;
  PASE_INT         cur_level 	 = solver->cur_level;
  PASE_INT         block_size	 = solver->block_size;
  PASE_INT         nconv         = solver->nconv; 
  PASE_INT         i		 = 0;

  PASE_AUX_MATRIX  aux_A         = solver->multigrid->aux_A[cur_level];            
  PASE_AUX_MATRIX  aux_B         = solver->multigrid->aux_B[cur_level];            
  PASE_AUX_VECTOR *aux_u         = solver->aux_u[cur_level];
  PASE_SCALAR     *eigenvalues   = solver->eigenvalues;
  PASE_AUX_VECTOR  rhs           = PASE_Aux_vector_create_by_aux_vector(aux_u[0]);

  HYPRE_Solver     cg_solver     = NULL;
  PASE_Pcg_create_aux(MPI_COMM_WORLD, &cg_solver);
  HYPRE_PCGSetMaxIter(cg_solver, solver->max_pre_iter); /* max iterations */
  HYPRE_PCGSetTol(cg_solver, 1.0e-50); 
  HYPRE_PCGSetTwoNorm(cg_solver, 1);                    /* use the two norm as the stopping criteria */
  HYPRE_PCGSetPrintLevel(cg_solver, 0); 
  HYPRE_PCGSetLogging(cg_solver, 1);                    /* needed to get run info later */
  hypre_PCGSetup(cg_solver, aux_A, rhs, aux_u[0]);

  for( i = nconv; i < block_size; i++) {
    PASE_Aux_matrix_multiply_aux_vector(aux_B, aux_u[i], rhs);
    PASE_Aux_vector_scale(eigenvalues[i], rhs);
    hypre_PCGSolve(cg_solver, aux_A, rhs, aux_u[i]);

    PASE_Aux_vector_inner_product_general(rhs, aux_u[i], aux_A, &inner_A);
    PASE_Aux_vector_inner_product_general(rhs, aux_u[i], aux_B, &inner_B);
    eigenvalues[i] = inner_A / inner_B;
  }
  PASE_Mg_print_eigenvalue_of_current_level(solver);

  PASE_Aux_vector_destroy(rhs);
  HYPRE_ParCSRPCGDestroy(cg_solver);
  return 0;
}

PASE_INT 
PASE_Mg_direct_solve_by_lobpcg_aux_hypre(void *mg_solver)
{
  PASE_MG_SOLVER solver         = (PASE_MG_SOLVER)mg_solver;
  HYPRE_Solver   lobpcg_solver 	= NULL; 
  PASE_INT       maxIterations 	= 20; 	              /* maximum number of iterations */
  PASE_INT       pcgMode        = 1;    	      /* use rhs as initial guess for inner pcg iterations */
  PASE_INT       verbosity 	= 0;    	      /* print iterations info */
  PASE_REAL      atol 		= solver->atol;  /* absolute tolerance (all eigenvalues) */
  PASE_REAL      rtol           = 1e-50;

  PASE_INT myid = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if(myid != 0) {
    verbosity = 0;
  }

  PASE_INT         i           = 0;
  PASE_INT         block_size  = solver->block_size;
  PASE_INT         cur_level   = solver->cur_level;

  PASE_AUX_MATRIX  aux_A       = solver->multigrid->aux_A[cur_level];            
  PASE_AUX_MATRIX  aux_B       = solver->multigrid->aux_B[cur_level];            
  PASE_AUX_VECTOR *aux_u       = solver->aux_u[cur_level];
  PASE_SCALAR     *eigenvalues = solver->eigenvalues;
  PASE_AUX_VECTOR  x           = PASE_Aux_vector_create_by_aux_vector(aux_u[0]);

  mv_MultiVectorPtr eigenvectors_Hh = NULL;
  mv_MultiVectorPtr constraints_Hh  = NULL;
  mv_InterfaceInterpreter* interpreter_Hh;
  HYPRE_MatvecFunctions matvec_fn_Hh;
  interpreter_Hh = hypre_CTAlloc( mv_InterfaceInterpreter, 1);
  PASE_Lobpcg_setup_interpreter_aux( interpreter_Hh);
  PASE_Lobpcg_setup_matvec_aux( &matvec_fn_Hh);

  eigenvectors_Hh          = mv_MultiVectorCreateFromSampleVector( interpreter_Hh, block_size, aux_u[0]);
  mv_TempMultiVector* tmp  = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
  solver->aux_u[cur_level] = (PASE_AUX_VECTOR*)(tmp -> vector);
  for(i = 0; i < block_size; i++) {
    PASE_Aux_vector_destroy(solver->aux_u[cur_level][i]);
    solver->aux_u[cur_level][i] = aux_u[i];
  }
  //mv_MultiVectorSetRandom( eigenvectors_Hh, 77);

  HYPRE_LOBPCGCreate( interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
  HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
  HYPRE_LOBPCGSetPrecondUsageMode( lobpcg_solver, pcgMode);
  HYPRE_LOBPCGSetTol(lobpcg_solver, atol);
  HYPRE_LOBPCGSetRTol(lobpcg_solver, rtol);
  HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

  hypre_LOBPCGSetup(lobpcg_solver, aux_A, aux_u[0], x);
  hypre_LOBPCGSetupB(lobpcg_solver, aux_B, aux_u[0]);
  HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);
  PASE_Mg_print_eigenvalue_of_current_level(solver);

  PASE_Free(aux_u);
  free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh));
  PASE_Free( eigenvectors_Hh);
  PASE_Free( interpreter_Hh);
  PASE_Aux_vector_destroy(x);
  HYPRE_LOBPCGDestroy( lobpcg_solver);
  return 0;
}

#endif