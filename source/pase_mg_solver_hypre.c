#include "pase_mg_solver_hypre.h"
#include "pase_mg_solver.h"
#include "pase_pcg_hypre.h"
#include "pase_lobpcg_hypre.h"
#include "time.h"

#if PASE_USE_HYPRE
#include "HYPRE_utilities.h"
#include "HYPRE_lobpcg.h"
#include "lobpcg.h"

#define CLK_TCK 1000000

PASE_INT
PASE_Mg_get_initial_vector_by_coarse_grid_hypre(void *mg_solver)
{
  PASE_MG_SOLVER solver        = (PASE_MG_SOLVER)mg_solver;
  solver->method_init          = "solve eigenvalue problem by lobpcg without preconditioner on coarsest grid";
  HYPRE_Solver   lobpcg_solver = NULL; 
  PASE_INT       maxIterations = 1000; 	        /* maximum number of iterations */
  PASE_INT       pcgMode       = 1;    	        /* use rhs as initial guess for inner pcg iterations */
  PASE_INT       verbosity     = 1;    	        /* print iterations info */
  PASE_REAL      atol 	       = solver->atol*1e-6;	/* absolute tolerance (all eigenvalues) */
  PASE_REAL      rtol          = 1e-50;
  PASE_INT       lobpcgSeed    = 77;

  PASE_INT       myid          = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if(myid != 0) {
    verbosity = 0;
  }

  PASE_INT     i              = 0;
  PASE_INT     cur_level      = solver->idx_cycle_level[solver->max_cycle_level];        
  PASE_INT     block_size     = solver->block_size;       
  PASE_MATRIX  A              = solver->multigrid->A[cur_level];
  PASE_MATRIX  B              = solver->multigrid->B[cur_level];
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

#if 0
  /* 根据初始的特征值分布，调整实际求解的特征值个数来提高收敛速度 */
  PASE_REAL *rate     = (PASE_REAL*)PASE_Malloc((max_block_size-block_size)*sizeof(PASE_REAL));
  PASE_INT   index    = 0;
  PASE_REAL  max_rate = 1.0;
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
  PASE_Free(rate);
#else
  solver->block_size = max_block_size;
#endif
  PASE_Printf(MPI_COMM_WORLD, "modified block_size = %d\n\n", solver->block_size);

  block_size              = solver->block_size;
  mv_TempMultiVector *tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
  PASE_VECTOR        *u_H = (PASE_VECTOR*)(tmp->vector);
  for(i = 0; i < block_size; i++) {
    PASE_Mg_prolong(solver, cur_level, u_H[i], 0, solver->u[i]);
    solver->eigenvalues[i] = eigenvalues[i];
    PASE_Vector_destroy(u_H[i]);
  }
  for(i = block_size; i < max_block_size; i++) {
    PASE_Vector_destroy(solver->u[i]);
    PASE_Vector_destroy(u_H[i]);
  }
  //PASE_Mg_smoothing_by_pcg_amg_hypre_for_guangji(mg_solver);
  PASE_Mg_smoothing_by_pcg_amg_hypre(mg_solver, "NeitherPreNorPost");

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
PASE_Mg_get_initial_vector_by_coarse_grid_lobpcg_amg_hypre(void *mg_solver)
{
  PASE_MG_SOLVER solver        = (PASE_MG_SOLVER)mg_solver;
  solver->method_init          = "solve eigenvalue problem by lobpcg with amg preconditioner on coarsest grid";
  HYPRE_Solver   lobpcg_solver = NULL; 
  HYPRE_Solver   precond       = NULL; 
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

  PASE_INT     i              = 0;
  PASE_INT     cur_level      = solver->idx_cycle_level[solver->max_cycle_level];        
  PASE_INT     block_size     = solver->block_size;       
  PASE_MATRIX  A              = solver->multigrid->A[cur_level];
  PASE_MATRIX  B              = solver->multigrid->B[cur_level];
  PASE_VECTOR  u_temp = PASE_Vector_create_by_matrix_and_vector_data_operator(A, solver->u[0]->ops); 
  PASE_VECTOR  x              = PASE_Vector_create_by_vector(u_temp);

  PASE_INT     max_block_size = solver->max_block_size;
  PASE_SCALAR *eigenvalues    = (PASE_SCALAR*)PASE_Malloc(max_block_size*sizeof(PASE_SCALAR));

  mv_InterfaceInterpreter *interpreter_Hh  = hypre_CTAlloc(mv_InterfaceInterpreter, 1);
  HYPRE_MatvecFunctions    matvec_fn_Hh;
  HYPRE_ParCSRSetupInterpreter(interpreter_Hh);
  HYPRE_ParCSRSetupMatvec(&matvec_fn_Hh);

  mv_MultiVectorPtr        eigenvectors_Hh = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, max_block_size, (HYPRE_ParVector)u_temp->vector_data);
  mv_MultiVectorPtr        constraints_Hh  = NULL;
  mv_MultiVectorSetRandom(eigenvectors_Hh, lobpcgSeed);

  HYPRE_BoomerAMGCreate(&precond);
  HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
  HYPRE_BoomerAMGSetNumSweeps(precond, 1); /* 2 sweeps of smoothing */
  HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
  HYPRE_BoomerAMGSetCoarsenType(precond, 6);
  HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

  HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
  HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
  HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
  HYPRE_LOBPCGSetTol(lobpcg_solver, atol);
  HYPRE_LOBPCGSetRTol(lobpcg_solver, rtol);
  HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);
  HYPRE_LOBPCGSetPrecond(lobpcg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

  hypre_LOBPCGSetup(lobpcg_solver, (HYPRE_ParCSRMatrix)A->matrix_data, (HYPRE_ParVector)u_temp->vector_data, (HYPRE_ParVector)x->vector_data);
  hypre_LOBPCGSetupB(lobpcg_solver, (HYPRE_ParCSRMatrix)B->matrix_data, (HYPRE_ParVector)u_temp->vector_data);
  HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);

#if 0
  /* 根据初始的特征值分布，调整实际求解的特征值个数来提高收敛速度 */
  PASE_REAL *rate     = (PASE_REAL*)PASE_Malloc((max_block_size-block_size)*sizeof(PASE_REAL));
  PASE_INT   index    = 0;
  PASE_REAL  max_rate = 1.0;
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
  PASE_Free(rate);
#else
  solver->block_size      = max_block_size;
#endif
  PASE_Printf(MPI_COMM_WORLD, "modified block_size = %d\n\n", solver->block_size);

  block_size              = solver->block_size;
  mv_TempMultiVector *tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
  HYPRE_ParVector    *u_H = (HYPRE_ParVector*)(tmp->vector);
  PASE_VECTOR         pase_u_H = NULL;
  for(i = 0; i < block_size; i++) {
    pase_u_H = PASE_Vector_assign(u_H[i], u_temp->ops);
    PASE_Mg_prolong(solver, cur_level, pase_u_H, 0, solver->u[i]);
    solver->eigenvalues[i] = eigenvalues[i];
    PASE_Vector_destroy(pase_u_H);
    HYPRE_ParVectorDestroy(u_H[i]);
  }

  for(i = block_size; i < max_block_size; i++) {
    PASE_Vector_destroy(solver->u[i]);
    HYPRE_ParVectorDestroy(u_H[i]);
  }
  PASE_Mg_presmoothing_by_pcg_amg_hypre(solver);

  free((mv_TempMultiVector*)mv_MultiVectorGetData(eigenvectors_Hh));
  PASE_Free(eigenvectors_Hh);
  PASE_Free(interpreter_Hh);
  PASE_Vector_destroy(u_temp);
  PASE_Vector_destroy(x);

  PASE_Free(eigenvalues);
  PASE_Free(u_H);
  HYPRE_BoomerAMGDestroy(precond);
  HYPRE_LOBPCGDestroy( lobpcg_solver);
  return 0;
}

PASE_INT
PASE_Mg_get_initial_vector_by_full_multigrid_hypre(void *mg_solver)
{
  PASE_MG_SOLVER solver        = (PASE_MG_SOLVER)mg_solver;
  solver->method_init          = "full multigrid";
  HYPRE_Solver   lobpcg_solver = NULL; 
  HYPRE_Solver   precond       = NULL; 
  PASE_INT       maxIterations = 30; 	        /* maximum number of iterations */
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
  clock_t      start, end;
  PASE_REAL    direct_time    = 0.0;
  PASE_REAL    aux_direct_time    = 0.0;
  PASE_REAL    smooth_time    = 0.0;
  PASE_REAL    set_aux_time   = 0.0;
  PASE_REAL    prolong_time   = 0.0;

  start = clock();
  PASE_INT     i              = 0;
  PASE_INT     cur_level      = solver->idx_cycle_level[solver->max_cycle_level];        
  PASE_INT     block_size     = solver->block_size;       
  PASE_MATRIX *A              = solver->multigrid->A;
  PASE_MATRIX *B              = solver->multigrid->B;
  PASE_VECTOR  u_temp = PASE_Vector_create_by_matrix_and_vector_data_operator(A[cur_level], solver->u[0]->ops); 
  PASE_VECTOR  x              = PASE_Vector_create_by_vector(u_temp);

  PASE_INT     max_block_size = solver->max_block_size;
  PASE_SCALAR *eigenvalues    = (PASE_SCALAR*)PASE_Malloc(max_block_size*sizeof(PASE_SCALAR));

  mv_InterfaceInterpreter *interpreter_Hh  = hypre_CTAlloc(mv_InterfaceInterpreter, 1);
  HYPRE_MatvecFunctions    matvec_fn_Hh;
  HYPRE_ParCSRSetupInterpreter(interpreter_Hh);
  HYPRE_ParCSRSetupMatvec(&matvec_fn_Hh);

  mv_MultiVectorPtr        eigenvectors_Hh = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, max_block_size, (HYPRE_ParVector)u_temp->vector_data);
  mv_MultiVectorPtr        constraints_Hh  = NULL;
  mv_MultiVectorSetRandom(eigenvectors_Hh, lobpcgSeed);

  HYPRE_BoomerAMGCreate(&precond);
  HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
  HYPRE_BoomerAMGSetInterpType( precond,  0 );
  HYPRE_BoomerAMGSetPMaxElmts( precond,  0 );
  HYPRE_BoomerAMGSetNumSweeps(precond, 1); /* 2 sweeps of smoothing */
  HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
  HYPRE_BoomerAMGSetCoarsenType(precond, 6);
  HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

  HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
  HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
  HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
  HYPRE_LOBPCGSetTol(lobpcg_solver, atol);
  HYPRE_LOBPCGSetRTol(lobpcg_solver, rtol);
  HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);
  HYPRE_LOBPCGSetPrecond(lobpcg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

  hypre_LOBPCGSetup(lobpcg_solver, (HYPRE_ParCSRMatrix)A[cur_level]->matrix_data, (HYPRE_ParVector)u_temp->vector_data, (HYPRE_ParVector)x->vector_data);
  hypre_LOBPCGSetupB(lobpcg_solver, (HYPRE_ParCSRMatrix)B[cur_level]->matrix_data, (HYPRE_ParVector)u_temp->vector_data);
  HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);
  end = clock();
  direct_time += ((double)(end-start))/CLK_TCK;
  start = clock();

#if 0
  /* 根据初始的特征值分布，调整实际求解的特征值个数来提高收敛速度 */
  PASE_REAL *rate     = (PASE_REAL*)PASE_Malloc((max_block_size-block_size)*sizeof(PASE_REAL));
  PASE_INT   index    = 0;
  PASE_REAL  max_rate = 1.0;
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
  PASE_Free(rate);
#else
  solver->block_size      = max_block_size;
#endif
  PASE_Printf(MPI_COMM_WORLD, "modified block_size = %d\n\n", solver->block_size);

  block_size              = solver->block_size;
  mv_TempMultiVector *tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
  HYPRE_ParVector    *u_H = (HYPRE_ParVector*)(tmp->vector);
  PASE_VECTOR         pase_u_H = NULL;
  PASE_VECTOR        *pase_u_h = (PASE_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_VECTOR));
  for(i = 0; i < block_size; i++) {
    pase_u_H = PASE_Vector_assign(u_H[i], u_temp->ops);
    pase_u_h[i] = PASE_Vector_create_by_matrix_and_vector_data_operator(A[cur_level-1], u_temp->ops);
    PASE_Mg_prolong(solver, cur_level, pase_u_H, cur_level-1, pase_u_h[i]);
    solver->eigenvalues[i] = eigenvalues[i];
    PASE_Vector_destroy(pase_u_H);
    HYPRE_ParVectorDestroy(u_H[i]);
  }
  end = clock();
  prolong_time += ((double)(end-start))/CLK_TCK;

  HYPRE_Solver ksp_solver = NULL;
  PASE_VECTOR  rhs = NULL;
  PASE_INT idx_level = 0;
  PASE_INT max_iter_smooth = solver->max_pre_iter+solver->max_post_iter;
  //PASE_INT max_iter_smooth = 1;
#if 1
    HYPRE_BoomerAMGCreate(&ksp_solver);
    HYPRE_BoomerAMGSetPrintLevel(ksp_solver, 0); /* print amg solution info */
    HYPRE_BoomerAMGSetInterpType(ksp_solver, 0);
    HYPRE_BoomerAMGSetPMaxElmts(ksp_solver, 0);
    HYPRE_BoomerAMGSetOldDefault(ksp_solver); /* Falgout coarsening with modified classical interpolaiton */
    HYPRE_BoomerAMGSetRelaxType(ksp_solver, 3);   /* G-S/Jacobi hybrid relaxation */
    HYPRE_BoomerAMGSetRelaxOrder(ksp_solver, 1);   /* uses C/F relaxation */
    HYPRE_BoomerAMGSetNumSweeps(ksp_solver, 1); /* 2 sweeps of smoothing */
    HYPRE_BoomerAMGSetTol(ksp_solver, 0.0); /* conv. tolerance zero */
    HYPRE_BoomerAMGSetCoarsenType(ksp_solver, 6);
    HYPRE_BoomerAMGSetMaxIter(ksp_solver, max_iter_smooth); /* do only one iteration! */
#else
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &ksp_solver);
    HYPRE_PCGSetMaxIter(ksp_solver, max_iter_smooth); /* max iterations */
    HYPRE_PCGSetTol(ksp_solver, 1.0e-50); 
    HYPRE_PCGSetTwoNorm(ksp_solver, 1);                    /* use the two norm as the stopping criteria */
    HYPRE_PCGSetPrintLevel(ksp_solver, 0); 
    HYPRE_PCGSetLogging(ksp_solver, 1);                    /* needed to get run info later */
    HYPRE_PCGSetPrecond(ksp_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
    hypre_PCGSetup(ksp_solver, (HYPRE_ParCSRMatrix)A[idx_level]->matrix_data, (HYPRE_ParVector)rhs->vector_data, (HYPRE_ParVector)pase_u_h[0]->vector_data);

#endif
  for(idx_level = cur_level-1; idx_level > 0; idx_level--) {
    start = clock();
    //解问题
    rhs = PASE_Vector_create_by_vector(pase_u_h[0]);
    HYPRE_BoomerAMGSetup(ksp_solver, (HYPRE_ParCSRMatrix)A[idx_level]->matrix_data, (HYPRE_ParVector)rhs->vector_data, (HYPRE_ParVector)pase_u_h[0]->vector_data);

    for(i = 0; i < block_size; i++) {
      PASE_Matrix_multiply_vector_general(solver->eigenvalues[i], B[idx_level], pase_u_h[i], 0.0, rhs);
      HYPRE_BoomerAMGSolve(ksp_solver, (HYPRE_ParCSRMatrix)A[idx_level]->matrix_data, (HYPRE_ParVector)rhs->vector_data, (HYPRE_ParVector)pase_u_h[i]->vector_data);
      //hypre_PCGSolve(ksp_solver, (HYPRE_ParCSRMatrix)A[idx_level]->matrix_data, (HYPRE_ParVector)rhs->vector_data, (HYPRE_ParVector)pase_u_h[i]->vector_data);
    }
    end = clock();
    smooth_time += ((double)(end-start))/CLK_TCK;
    start = clock();

    //构造辅助矩阵
    PASE_Mg_set_pase_aux_matrix_by_pase_matrix(solver, cur_level, idx_level, pase_u_h);
    PASE_Mg_set_pase_aux_vector(solver, cur_level);
    end = clock();
    set_aux_time += ((double)(end-start))/CLK_TCK;
    start = clock();

    //特征值问题
    PASE_Mg_direct_solve_by_gcg(solver);
    end = clock();
    aux_direct_time += ((double)(end-start))/CLK_TCK;
    start = clock();

    //投影到 idx_level层, 再投影到 idx_level-1 层
    PASE_Mg_prolong_from_pase_aux_vector_to_pase_vector(solver, cur_level, solver->aux_u[cur_level], idx_level, pase_u_h);
    for(i = 0; i < block_size; i++) {
      pase_u_H = PASE_Vector_create_by_matrix_and_vector_data_operator(A[idx_level-1], u_temp->ops);
      PASE_Mg_prolong(solver, idx_level, pase_u_h[i], idx_level-1, pase_u_H);
      PASE_Vector_destroy(pase_u_h[i]);
      pase_u_h[i] = pase_u_H;
    }
    end = clock();
    prolong_time += ((double)(end-start))/CLK_TCK;
  }

  for(i = 0; i < block_size; i++) {
    PASE_Vector_copy(pase_u_h[i], solver->u[i]);
    PASE_Vector_destroy(pase_u_h[i]);
  }
  for(i = block_size; i < max_block_size; i++) {
    PASE_Vector_destroy(solver->u[i]);
    HYPRE_ParVectorDestroy(u_H[i]);
  }

  start = clock();
  PASE_Mg_smoothing_by_amg_hypre(solver, "NeitherPreNorPost");
  end = clock();
  smooth_time += ((double)(end-start))/CLK_TCK;
  PASE_Printf(MPI_COMM_WORLD, "\n");
  PASE_Printf(MPI_COMM_WORLD, "max iter of direct solve = %d\n", maxIterations);
  PASE_Printf(MPI_COMM_WORLD, "max iter of smooth       = %d\n", max_iter_smooth);
  PASE_Printf(MPI_COMM_WORLD, "direct solve time        = %.6f\n", direct_time);
  PASE_Printf(MPI_COMM_WORLD, "aux direct solve time    = %.6f\n", aux_direct_time);
  PASE_Printf(MPI_COMM_WORLD, "smooth time              = %.6f\n", smooth_time);
  PASE_Printf(MPI_COMM_WORLD, "set aux time             = %.6f\n", set_aux_time);
  PASE_Printf(MPI_COMM_WORLD, "prolong time             = %.6f\n", prolong_time);
  PASE_Printf(MPI_COMM_WORLD, "\n");

  free((mv_TempMultiVector*)mv_MultiVectorGetData(eigenvectors_Hh));
  PASE_Free(eigenvectors_Hh);
  PASE_Free(interpreter_Hh);
  PASE_Vector_destroy(u_temp);
  PASE_Vector_destroy(x);
  PASE_Vector_destroy(rhs);

  PASE_Free(eigenvalues);
  PASE_Free(u_H);
  PASE_Free(pase_u_h);
  HYPRE_BoomerAMGDestroy(precond);
  HYPRE_BoomerAMGDestroy(ksp_solver);
  HYPRE_LOBPCGDestroy( lobpcg_solver);
  return 0;
}

PASE_INT
PASE_Mg_get_initial_vector_by_full_multigrid_hypre_for_guangji(void *mg_solver)
{

  PASE_MG_SOLVER solver        = (PASE_MG_SOLVER)mg_solver;
  solver->method_init          = "full multigrid for guangji";
  HYPRE_Solver   lobpcg_solver = NULL; 
  HYPRE_Solver   precond       = NULL; 
  PASE_INT       maxIterations = 500; 	        /* maximum number of iterations */
  PASE_INT       pcgMode       = 1;    	        /* use rhs as initial guess for inner pcg iterations */
  PASE_INT       verbosity     = 1;    	        /* print iterations info */
  PASE_REAL      atol 	       = solver->atol;	/* absolute tolerance (all eigenvalues) */
  PASE_REAL      rtol          = 1e-50;
  PASE_INT       lobpcgSeed    = 77;

  PASE_INT       myid          = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if(myid != 0) {
    verbosity = 0;
  }
  clock_t      start, end;
  PASE_REAL    direct_time    = 0.0;
  PASE_REAL    aux_direct_time    = 0.0;
  PASE_REAL    smooth_time    = 0.0;
  PASE_REAL    set_aux_time   = 0.0;
  PASE_REAL    prolong_time   = 0.0;

  start = clock();
  PASE_INT     i              = 0;
  PASE_INT     cur_level      = solver->idx_cycle_level[solver->max_cycle_level];        
  PASE_INT     block_size     = solver->block_size;       
  PASE_MATRIX *A              = solver->multigrid->A;
  PASE_MATRIX *B              = solver->multigrid->B;
  PASE_VECTOR  u_temp = PASE_Vector_create_by_matrix_and_vector_data_operator(A[cur_level], solver->u[0]->ops); 
  PASE_VECTOR  x              = PASE_Vector_create_by_vector(u_temp);
  PASE_MATRIX *Asub           = solver->multigrid_pre->A;

  PASE_INT     max_block_size = solver->max_block_size;
  PASE_SCALAR *eigenvalues    = (PASE_SCALAR*)PASE_Malloc(max_block_size*sizeof(PASE_SCALAR));

  mv_InterfaceInterpreter *interpreter_Hh  = hypre_CTAlloc(mv_InterfaceInterpreter, 1);
  HYPRE_MatvecFunctions    matvec_fn_Hh;
  HYPRE_ParCSRSetupInterpreter(interpreter_Hh);
  HYPRE_ParCSRSetupMatvec(&matvec_fn_Hh);

  mv_MultiVectorPtr        eigenvectors_Hh = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, max_block_size, (HYPRE_ParVector)u_temp->vector_data);
  mv_MultiVectorPtr        constraints_Hh  = NULL;
  mv_MultiVectorSetRandom(eigenvectors_Hh, lobpcgSeed);

  HYPRE_BoomerAMGCreate(&precond);
  HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
  HYPRE_BoomerAMGSetInterpType( precond,  0 );
  HYPRE_BoomerAMGSetPMaxElmts( precond,  0 );
  HYPRE_BoomerAMGSetNumSweeps(precond, 1); /* 2 sweeps of smoothing */
  HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
  HYPRE_BoomerAMGSetCoarsenType(precond, 6);
  HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

  HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
  HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
  HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
  HYPRE_LOBPCGSetTol(lobpcg_solver, atol);
  HYPRE_LOBPCGSetRTol(lobpcg_solver, rtol);
  HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);
  HYPRE_LOBPCGSetPrecond(lobpcg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

  hypre_LOBPCGSetup(lobpcg_solver, (HYPRE_ParCSRMatrix)A[cur_level]->matrix_data, (HYPRE_ParVector)u_temp->vector_data, (HYPRE_ParVector)x->vector_data);
  hypre_LOBPCGSetupB(lobpcg_solver, (HYPRE_ParCSRMatrix)B[cur_level]->matrix_data, (HYPRE_ParVector)u_temp->vector_data);
  HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);
  end = clock();
  direct_time += ((double)(end-start))/CLK_TCK;
  start = clock();

#if 0
  /* 根据初始的特征值分布，调整实际求解的特征值个数来提高收敛速度 */
  PASE_REAL *rate     = (PASE_REAL*)PASE_Malloc((max_block_size-block_size)*sizeof(PASE_REAL));
  PASE_INT   index    = 0;
  PASE_REAL  max_rate = 1.0;
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
  PASE_Free(rate);
#else
  solver->block_size      = max_block_size;
#endif
  PASE_Printf(MPI_COMM_WORLD, "modified block_size = %d\n\n", solver->block_size);

  block_size              = solver->block_size;
  mv_TempMultiVector *tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
  HYPRE_ParVector    *u_H = (HYPRE_ParVector*)(tmp->vector);
  PASE_VECTOR         pase_u_H = NULL;
  PASE_VECTOR        *pase_u_h = (PASE_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_VECTOR));
  for(i = 0; i < block_size; i++) {
    pase_u_H = PASE_Vector_assign(u_H[i], u_temp->ops);
    pase_u_h[i] = PASE_Vector_create_by_matrix_and_vector_data_operator(A[cur_level-1], u_temp->ops);
    PASE_Mg_prolong(solver, cur_level, pase_u_H, cur_level-1, pase_u_h[i]);
    solver->eigenvalues[i] = eigenvalues[i];
    PASE_Vector_destroy(pase_u_H);
    HYPRE_ParVectorDestroy(u_H[i]);
  }
  end = clock();
  prolong_time += ((double)(end-start))/CLK_TCK;

  HYPRE_Solver ksp_solver = NULL;
  PASE_VECTOR  rhs = NULL;
  PASE_INT idx_level = 0;
  PASE_INT max_iter_smooth = solver->max_pre_iter+solver->max_post_iter;
  //PASE_INT max_iter_smooth = 1;
#if 0
    HYPRE_BoomerAMGCreate(&ksp_solver);
    HYPRE_BoomerAMGSetPrintLevel(ksp_solver, 0); /* print amg solution info */
    HYPRE_BoomerAMGSetInterpType(ksp_solver, 0);
    HYPRE_BoomerAMGSetPMaxElmts(ksp_solver, 0);
    HYPRE_BoomerAMGSetOldDefault(ksp_solver); /* Falgout coarsening with modified classical interpolaiton */
    HYPRE_BoomerAMGSetRelaxType(ksp_solver, 3);   /* G-S/Jacobi hybrid relaxation */
    HYPRE_BoomerAMGSetRelaxOrder(ksp_solver, 1);   /* uses C/F relaxation */
    HYPRE_BoomerAMGSetNumSweeps(ksp_solver, 1); /* 2 sweeps of smoothing */
    HYPRE_BoomerAMGSetTol(ksp_solver, 0.0); /* conv. tolerance zero */
    HYPRE_BoomerAMGSetCoarsenType(ksp_solver, 6);
    HYPRE_BoomerAMGSetMaxIter(ksp_solver, max_iter_smooth); /* do only one iteration! */
#else
    HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &ksp_solver);
    HYPRE_PCGSetMaxIter(ksp_solver, max_iter_smooth); /* max iterations */
    HYPRE_PCGSetTol(ksp_solver, 1.0e-50); 
    HYPRE_PCGSetTwoNorm(ksp_solver, 1);                    /* use the two norm as the stopping criteria */
    HYPRE_PCGSetPrintLevel(ksp_solver, 2); 
    HYPRE_PCGSetLogging(ksp_solver, 1);                    /* needed to get run info later */
    HYPRE_PCGSetPrecond(ksp_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

#endif
  for(idx_level = cur_level-1; idx_level > 0; idx_level--) {
    start = clock();
    //解问题
    rhs = PASE_Vector_create_by_vector(pase_u_h[0]);
    hypre_PCGSetup(ksp_solver, (HYPRE_ParCSRMatrix)Asub[idx_level]->matrix_data, (HYPRE_ParVector)rhs->vector_data, (HYPRE_ParVector)pase_u_h[0]->vector_data);
    //HYPRE_BoomerAMGSetup(ksp_solver, (HYPRE_ParCSRMatrix)A[idx_level]->matrix_data, (HYPRE_ParVector)rhs->vector_data, (HYPRE_ParVector)pase_u_h[0]->vector_data);

    for(i = 0; i < block_size; i++) {
      PASE_Matrix_multiply_vector_general(solver->eigenvalues[i], B[idx_level], pase_u_h[i], 0.0, rhs);
      //HYPRE_BoomerAMGSolve(ksp_solver, (HYPRE_ParCSRMatrix)A[idx_level]->matrix_data, (HYPRE_ParVector)rhs->vector_data, (HYPRE_ParVector)pase_u_h[i]->vector_data);
      hypre_PCGSolve(ksp_solver, (HYPRE_ParCSRMatrix)A[idx_level]->matrix_data, (HYPRE_ParVector)rhs->vector_data, (HYPRE_ParVector)pase_u_h[i]->vector_data);
    }
    end = clock();
    smooth_time += ((double)(end-start))/CLK_TCK;
    start = clock();

    //构造辅助矩阵
    PASE_Mg_set_pase_aux_matrix_by_pase_matrix(solver, cur_level, idx_level, pase_u_h);
    PASE_Mg_set_pase_aux_vector(solver, cur_level);
    end = clock();
    set_aux_time += ((double)(end-start))/CLK_TCK;
    start = clock();

    //特征值问题
    PASE_Mg_direct_solve_by_lobpcg_aux_hypre(solver);
    //PASE_Mg_direct_solve_by_gcg(solver);
    end = clock();
    aux_direct_time += ((double)(end-start))/CLK_TCK;
    start = clock();

    //投影到 idx_level层, 再投影到 idx_level-1 层
    PASE_Mg_prolong_from_pase_aux_vector_to_pase_vector(solver, cur_level, solver->aux_u[cur_level], idx_level, pase_u_h);
    for(i = 0; i < block_size; i++) {
      pase_u_H = PASE_Vector_create_by_matrix_and_vector_data_operator(A[idx_level-1], u_temp->ops);
      PASE_Mg_prolong(solver, idx_level, pase_u_h[i], idx_level-1, pase_u_H);
      PASE_Vector_destroy(pase_u_h[i]);
      pase_u_h[i] = pase_u_H;
    }
    end = clock();
    prolong_time += ((double)(end-start))/CLK_TCK;
  }

  for(i = 0; i < block_size; i++) {
    PASE_Vector_copy(pase_u_h[i], solver->u[i]);
    PASE_Vector_destroy(pase_u_h[i]);
  }
  for(i = block_size; i < max_block_size; i++) {
    PASE_Vector_destroy(solver->u[i]);
    HYPRE_ParVectorDestroy(u_H[i]);
  }

  start = clock();
  PASE_Mg_smoothing_by_pcg_amg_hypre_for_guangji(solver);
  end = clock();
  smooth_time += ((double)(end-start))/CLK_TCK;
  PASE_Printf(MPI_COMM_WORLD, "\n");
  PASE_Printf(MPI_COMM_WORLD, "max iter of direct solve = %d\n", maxIterations);
  PASE_Printf(MPI_COMM_WORLD, "max iter of smooth       = %d\n", max_iter_smooth);
  PASE_Printf(MPI_COMM_WORLD, "direct solve time        = %.6f\n", direct_time);
  PASE_Printf(MPI_COMM_WORLD, "aux direct solve time    = %.6f\n", aux_direct_time);
  PASE_Printf(MPI_COMM_WORLD, "smooth time              = %.6f\n", smooth_time);
  PASE_Printf(MPI_COMM_WORLD, "set aux time             = %.6f\n", set_aux_time);
  PASE_Printf(MPI_COMM_WORLD, "prolong time             = %.6f\n", prolong_time);
  PASE_Printf(MPI_COMM_WORLD, "\n");

  free((mv_TempMultiVector*)mv_MultiVectorGetData(eigenvectors_Hh));
  PASE_Free(eigenvectors_Hh);
  PASE_Free(interpreter_Hh);
  PASE_Vector_destroy(u_temp);
  PASE_Vector_destroy(x);
  PASE_Vector_destroy(rhs);

  PASE_Free(eigenvalues);
  PASE_Free(u_H);
  PASE_Free(pase_u_h);
  HYPRE_BoomerAMGDestroy(precond);
  //HYPRE_BoomerAMGDestroy(ksp_solver);
  HYPRE_ParCSRPCGDestroy(ksp_solver);
  HYPRE_LOBPCGDestroy( lobpcg_solver);
  return 0;
}

PASE_INT 
PASE_Mg_direct_solve_by_lobpcg_aux_hypre(void *mg_solver)
{
  PASE_MG_SOLVER solver         = (PASE_MG_SOLVER)mg_solver;
  solver->method_dire = "lobpcg";
  HYPRE_Solver   lobpcg_solver 	= NULL; 
  PASE_INT       maxIterations 	= solver->max_direct_iter; 	              /* maximum number of iterations */
  PASE_INT       pcgMode        = 1;    	      /* use rhs as initial guess for inner pcg iterations */
  PASE_INT       verbosity 	= 1;    	      /* print iterations info */
  PASE_REAL      atol 		= solver->atol*1e-6;  /* absolute tolerance (all eigenvalues) */
  PASE_REAL      rtol           = 1e-50;

  PASE_INT myid = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  if(myid != 0) {
    verbosity = 0;
  }

  PASE_INT         i           = 0;
  PASE_INT         block_size  = solver->block_size;
  PASE_INT         cur_level   = solver->idx_cycle_level[solver->max_cycle_level];

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

  PASE_Free(aux_u);
  free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh));
  PASE_Free( eigenvectors_Hh);
  PASE_Free( interpreter_Hh);
  PASE_Aux_vector_destroy(x);
  HYPRE_LOBPCGDestroy( lobpcg_solver);
  return 0;
}

PASE_INT 
PASE_Mg_smoothing_by_pcg_hypre(void *mg_solver, char *PreOrPost)
{
  //PASE_SCALAR    inner_A, inner_B;
  PASE_MG_SOLVER solver      = (PASE_MG_SOLVER)mg_solver;
  PASE_INT       cur_level   = solver->idx_cycle_level[solver->cur_cycle_level];
  PASE_INT       block_size  = solver->block_size;
  PASE_INT       nconv       = solver->nconv; 
  PASE_INT       i	     = 0;
  PASE_INT       max_iter    = 1;
  if(0 == strcmp(PreOrPost, "pre")) {
    max_iter = solver->max_pre_iter;
    if(NULL == solver->method_pre) {
      solver->method_pre = "pcg without preconditioner";
    }
  } else if(0 == strcmp(PreOrPost, "post")) {
    max_iter = solver->max_post_iter;
    if(NULL == solver->method_post) {
      solver->method_post = "pcg without preconditioner";
    }
  }

  PASE_MATRIX    A           = solver->multigrid->A[cur_level];            
  PASE_MATRIX    B           = solver->multigrid->B[cur_level];            
  PASE_VECTOR   *u           = solver->u;
  PASE_SCALAR   *eigenvalues = solver->eigenvalues;
  PASE_VECTOR    rhs 	     = PASE_Vector_create_by_vector(u[0]);

  HYPRE_Solver   cg_solver   = NULL;
  PASE_Pcg_create(MPI_COMM_WORLD, &cg_solver);
  HYPRE_PCGSetMaxIter(cg_solver, max_iter); /* max iterations */
  HYPRE_PCGSetTol(cg_solver, 1.0e-50); 
  HYPRE_PCGSetTwoNorm(cg_solver, 1);                    /* use the two norm as the stopping criteria */
  HYPRE_PCGSetPrintLevel(cg_solver, 0); 
  HYPRE_PCGSetLogging(cg_solver, 1);                    /* needed to get run info later */
  hypre_PCGSetup(cg_solver, A, rhs, u[0]);

  for(i = nconv; i < block_size; i++) {
    PASE_Matrix_multiply_vector(B, u[i], rhs);
    PASE_Vector_scale(eigenvalues[i], rhs);
    hypre_PCGSolve(cg_solver, A, rhs, u[i]);

    //PASE_Vector_inner_product_general(u[i], u[i], A, &inner_A);
    //PASE_Vector_inner_product_general(u[i], u[i], B, &inner_B);
    //eigenvalues[i] = inner_A / inner_B;
  }

  PASE_Vector_destroy(rhs);
  HYPRE_ParCSRPCGDestroy(cg_solver);
  return 0;
}

PASE_INT 
PASE_Mg_smoothing_by_pcg_amg_hypre(void *mg_solver, char *PreOrPost)
{
  //PASE_SCALAR     inner_A, inner_B;
  PASE_MG_SOLVER  solver        = (PASE_MG_SOLVER)mg_solver;
  HYPRE_Solver    cg_solver     = NULL;
  PASE_INT        block_size	= solver->block_size;
  PASE_INT        nconv         = solver->nconv; 
  PASE_INT        i		= 0;
  PASE_INT        max_iter      = 1;
  if(0 == strcmp(PreOrPost, "pre")) {
    max_iter = solver->max_pre_iter;
    if(NULL == solver->method_pre) {
      solver->method_pre = "pcg with amg preconditioner";
    }
  } else if(0 == strcmp(PreOrPost, "post")) {
    max_iter = solver->max_post_iter;
    if(NULL == solver->method_post) {
      solver->method_post = "pcg with amg preconditioner";
    }
  }

  PASE_MATRIX     A             = solver->multigrid->A[0];            
  PASE_MATRIX     B             = solver->multigrid->B[0];            
  PASE_VECTOR    *u             = solver->u;
  PASE_SCALAR    *eigenvalues   = solver->eigenvalues;
  PASE_VECTOR     rhs 	        = PASE_Vector_create_by_vector(u[0]);

  HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &cg_solver);
  HYPRE_PCGSetTol(cg_solver, 1.0e-15); 
  HYPRE_PCGSetMaxIter(cg_solver, max_iter); /* max iterations */
  HYPRE_PCGSetTwoNorm(cg_solver, 1); /* use the two norm as the st    opping criteria */
  HYPRE_PCGSetPrintLevel(cg_solver, 0); 
  HYPRE_PCGSetLogging(cg_solver, 1); /* needed to get run info lat    er */

  HYPRE_ParCSRPCGSetup(cg_solver, (HYPRE_ParCSRMatrix)(A->matrix_data), (HYPRE_ParVector)(rhs->vector_data), (HYPRE_ParVector)(u[i]->vector_data));
  HYPRE_PCGSetPrecond(cg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, solver->multigrid->amg_data);

  for(i = nconv; i < block_size; i++) {
    PASE_Matrix_multiply_vector(B, u[i], rhs);
    PASE_Vector_scale(eigenvalues[i], rhs);
    HYPRE_ParCSRPCGSolve(cg_solver, (HYPRE_ParCSRMatrix)(A->matrix_data), (HYPRE_ParVector)(rhs->vector_data), (HYPRE_ParVector)(u[i]->vector_data));

    //PASE_Vector_inner_product_general(u[i], u[i], A, &inner_A);
    //PASE_Vector_inner_product_general(u[i], u[i], B, &inner_B);
    //eigenvalues[i] = inner_A / inner_B;
  }

  PASE_Vector_destroy(rhs);
  HYPRE_ParCSRPCGDestroy(cg_solver);
  return 0;
}

PASE_INT 
PASE_Mg_smoothing_by_pcg_amg_hypre_for_guangji(void *mg_solver)
{
  //PASE_SCALAR     inner_A, inner_B;
  PASE_MG_SOLVER  solver        = (PASE_MG_SOLVER)mg_solver;
  HYPRE_Solver    cg_solver     = NULL;
  PASE_INT        block_size	= solver->block_size;
  PASE_INT        nconv         = solver->nconv; 
  PASE_INT        i		= 0;
  PASE_INT        max_iter      = solver->max_post_iter;

  PASE_MATRIX     A             = solver->multigrid->A[0];            
  PASE_MATRIX     B             = solver->multigrid->B[0];            
  PASE_VECTOR    *u             = solver->u;
  PASE_SCALAR    *eigenvalues   = solver->eigenvalues;
  PASE_VECTOR     rhs 	        = PASE_Vector_create_by_vector(u[0]);

  //HYPRE_Solver    precond = NULL;
  //HYPRE_BoomerAMGCreate(&precond);
  //HYPRE_BoomerAMGSetPrintLevel(precond, 0); /* print amg solution info */
  //HYPRE_BoomerAMGSetNumSweeps(precond, 1); /* 2 sweeps of smoothing */
  //HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
  //HYPRE_BoomerAMGSetCoarsenType(precond, 6);
  //HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

  HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &cg_solver);
  HYPRE_PCGSetTol(cg_solver, 1.0e-10); 
  HYPRE_PCGSetMaxIter(cg_solver, max_iter); /* max iterations */
  HYPRE_PCGSetTwoNorm(cg_solver, 1); /* use the two norm as the st    opping criteria */
  HYPRE_PCGSetPrintLevel(cg_solver, 2); 
  HYPRE_PCGSetLogging(cg_solver, 1); /* needed to get run info lat    er */

  //HYPRE_PCGSetPrecond(cg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);
  HYPRE_ParCSRPCGSetup(cg_solver, (HYPRE_ParCSRMatrix)(solver->multigrid_pre->A[0]->matrix_data), (HYPRE_ParVector)(rhs->vector_data), (HYPRE_ParVector)(u[i]->vector_data));
  HYPRE_PCGSetPrecond(cg_solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, solver->multigrid_pre->amg_data);

  for(i = nconv; i < block_size; i++) {
    PASE_Matrix_multiply_vector(B, u[i], rhs);
    PASE_Vector_scale(eigenvalues[i], rhs);
    HYPRE_ParCSRPCGSolve(cg_solver, (HYPRE_ParCSRMatrix)(A->matrix_data), (HYPRE_ParVector)(rhs->vector_data), (HYPRE_ParVector)(u[i]->vector_data));

    //PASE_Vector_inner_product_general(u[i], u[i], A, &inner_A);
    //PASE_Vector_inner_product_general(u[i], u[i], B, &inner_B);
    //eigenvalues[i] = inner_A / inner_B;
    //PASE_Vector_scale(1.0/sqrt(inner_B), u[i]);
  }

  PASE_Vector_destroy(rhs);
  HYPRE_ParCSRPCGDestroy(cg_solver);
  //HYPRE_BoomerAMGDestroy(precond);
  return 0;
}

PASE_INT 
PASE_Mg_smoothing_by_amg_hypre(void *mg_solver, char *PreOrPost)
{
  //PASE_SCALAR     inner_A, inner_B;
  PASE_MG_SOLVER  solver        = (PASE_MG_SOLVER)mg_solver;
  PASE_INT        block_size	= solver->block_size;
  PASE_INT        nconv         = solver->nconv; 
  PASE_INT        i		= 0;
  PASE_INT        max_iter      = 1;
  if(0 == strcmp(PreOrPost, "pre")) {
    max_iter = solver->max_pre_iter;
    if(NULL == solver->method_pre) {
      solver->method_pre = "amg";
    }
  } else if(0 == strcmp(PreOrPost, "post")) {
    max_iter = solver->max_post_iter;
    if(NULL == solver->method_post) {
      solver->method_post = "amg";
    }
  }

  PASE_MATRIX     A             = solver->multigrid->A[0];            
  PASE_MATRIX     B             = solver->multigrid->B[0];            
  PASE_VECTOR    *u             = solver->u;
  PASE_SCALAR    *eigenvalues   = solver->eigenvalues;
  PASE_VECTOR     rhs 	        = PASE_Vector_create_by_vector(u[0]);

#if 0
  HYPRE_Solver    amg_solver     = NULL;
  HYPRE_BoomerAMGCreate(&amg_solver);
  HYPRE_BoomerAMGSetPrintLevel(amg_solver, 0); /* print amg solution info */
  HYPRE_BoomerAMGSetOldDefault(amg_solver); /* Falgout coarsening with modified classical interpolaiton */
  HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);   /* G-S/Jacobi hybrid relaxation */
  HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1);   /* uses C/F relaxation */
  HYPRE_BoomerAMGSetNumSweeps(amg_solver, 1); /* 2 sweeps of smoothing */
  HYPRE_BoomerAMGSetTol(amg_solver, solver->atol); /* conv. tolerance zero */
  HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
  HYPRE_BoomerAMGSetMaxIter(amg_solver, max_iter); 
  HYPRE_BoomerAMGSetup(amg_solver, (HYPRE_ParCSRMatrix)A->matrix_data, (HYPRE_ParVector)rhs->vector_data, (HYPRE_ParVector)u[0]->vector_data);
#endif

  HYPRE_BoomerAMGSetTol((HYPRE_Solver)solver->multigrid->amg_data, solver->atol); /* conv. tolerance zero */
  HYPRE_BoomerAMGSetMaxIter((HYPRE_Solver)solver->multigrid->amg_data, max_iter); 

  for(i = nconv; i < block_size; i++) {
    PASE_Matrix_multiply_vector(B, u[i], rhs);
    PASE_Vector_scale(eigenvalues[i], rhs);
    //HYPRE_BoomerAMGSolve(amg_solver, (HYPRE_ParCSRMatrix)(A->matrix_data), (HYPRE_ParVector)(rhs->vector_data), (HYPRE_ParVector)(u[i]->vector_data));
    HYPRE_BoomerAMGSolve((HYPRE_Solver)solver->multigrid->amg_data, (HYPRE_ParCSRMatrix)(A->matrix_data), (HYPRE_ParVector)(rhs->vector_data), (HYPRE_ParVector)(u[i]->vector_data));

    //PASE_Vector_inner_product_general(u[i], u[i], A, &inner_A);
    //PASE_Vector_inner_product_general(u[i], u[i], B, &inner_B);
    //PASE_Printf(MPI_COMM_WORLD, "inner_B = %f\n", 1.0/sqrt(inner_B));
    //PASE_Vector_scale(1.0/sqrt(inner_B), solver->u[i]);
    //eigenvalues[i] = inner_A / inner_B;
  }

  PASE_Vector_destroy(rhs);
  //HYPRE_BoomerAMGDestroy(amg_solver);
  return 0;
}

PASE_INT 
PASE_Mg_smoothing_by_pcg_aux_hypre(void *mg_solver, char *PreOrPost)
{
  PASE_SCALAR      inner_A, inner_B;
  PASE_MG_SOLVER   solver        = (PASE_MG_SOLVER)mg_solver;
  PASE_INT         cur_level 	 = solver->idx_cycle_level[solver->cur_cycle_level];
  PASE_INT         block_size	 = solver->block_size;
  PASE_INT         nconv         = solver->nconv; 
  PASE_INT         i		 = 0;
  PASE_INT         max_iter      = 1;
  if(0 == strcmp(PreOrPost, "pre")) {
    max_iter = solver->max_pre_iter;
    if(NULL == solver->method_pre_aux) {
      solver->method_pre_aux = "pcg without preconditioner";
    }
  } else if(0 == strcmp(PreOrPost, "post")) {
    max_iter = solver->max_post_iter;
    if(NULL == solver->method_post_aux) {
      solver->method_post_aux = "pcg without preconditioner";
    }
  }

  PASE_AUX_MATRIX  aux_A         = solver->multigrid->aux_A[cur_level];            
  PASE_AUX_MATRIX  aux_B         = solver->multigrid->aux_B[cur_level];            
  PASE_AUX_VECTOR *aux_u         = solver->aux_u[cur_level];
  PASE_SCALAR     *eigenvalues   = solver->eigenvalues;
  PASE_AUX_VECTOR  rhs           = PASE_Aux_vector_create_by_aux_vector(aux_u[0]);

  HYPRE_Solver     cg_solver     = NULL;
  PASE_Pcg_create_aux(MPI_COMM_WORLD, &cg_solver);
  HYPRE_PCGSetTol(cg_solver, 1.0e-50); 
  HYPRE_PCGSetMaxIter(cg_solver, max_iter); /* max iterations */
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

  PASE_Aux_vector_destroy(rhs);
  HYPRE_ParCSRPCGDestroy(cg_solver);
  return 0;
}

PASE_INT
PASE_Mg_presmoothing_by_pcg_hypre(void *mg_solver)
{
  PASE_Mg_smoothing_by_pcg_hypre(mg_solver, "pre");
  return 0;
}

PASE_INT
PASE_Mg_postsmoothing_by_pcg_hypre(void *mg_solver)
{
  PASE_Mg_smoothing_by_pcg_hypre(mg_solver, "post");
  return 0;
}

PASE_INT
PASE_Mg_presmoothing_by_amg_hypre(void *mg_solver)
{
  PASE_Mg_smoothing_by_amg_hypre(mg_solver, "pre");
  return 0;
}

PASE_INT
PASE_Mg_postsmoothing_by_amg_hypre(void *mg_solver)
{
  PASE_Mg_smoothing_by_amg_hypre(mg_solver, "post");
  return 0;
}

PASE_INT
PASE_Mg_presmoothing_by_pcg_amg_hypre(void *mg_solver)
{
  PASE_Mg_smoothing_by_pcg_amg_hypre(mg_solver, "pre");
  return 0;
}

PASE_INT
PASE_Mg_postsmoothing_by_pcg_amg_hypre(void *mg_solver)
{
  PASE_Mg_smoothing_by_pcg_amg_hypre(mg_solver, "post");
  return 0;
}

PASE_INT
PASE_Mg_presmoothing_by_pcg_aux_hypre(void *mg_solver)
{
  PASE_Mg_smoothing_by_pcg_aux_hypre(mg_solver, "pre");
  return 0;
}

PASE_INT
PASE_Mg_postsmoothing_by_pcg_aux_hypre(void *mg_solver)
{
  PASE_Mg_smoothing_by_pcg_aux_hypre(mg_solver, "post");
  return 0;
}

PASE_INT
PASE_Linear_solve_by_amg_hypre(PASE_MATRIX A, PASE_VECTOR *b, PASE_VECTOR *x, PASE_INT n, PASE_REAL tol, PASE_INT max_iter, void *amg_data)
{
  PASE_INT     i          = 0;
  HYPRE_Solver amg_solver = (HYPRE_Solver) amg_data;
  if(NULL == amg_data) {
    HYPRE_BoomerAMGCreate(&amg_solver);
    HYPRE_BoomerAMGSetPrintLevel(amg_solver, 0); /* print amg solution info */
    HYPRE_BoomerAMGSetOldDefault(amg_solver);    /* Falgout coarsening with modified classical interpolaiton */
    HYPRE_BoomerAMGSetRelaxType(amg_solver, 3);  /* G-S/Jacobi hybrid relaxation */
    HYPRE_BoomerAMGSetRelaxOrder(amg_solver, 1); /* uses C/F relaxation */
    HYPRE_BoomerAMGSetNumSweeps(amg_solver, 1);  /* 2 sweeps of smoothing */
    HYPRE_BoomerAMGSetCoarsenType(amg_solver, 6);
    HYPRE_BoomerAMGSetup(amg_solver, (HYPRE_ParCSRMatrix)A->matrix_data, (HYPRE_ParVector)b[0]->vector_data, (HYPRE_ParVector)x[0]->vector_data);
  }

  HYPRE_BoomerAMGSetTol(amg_solver, tol); 
  HYPRE_BoomerAMGSetMaxIter(amg_solver, max_iter); 
  for(i = 0; i < n; i++) {
    HYPRE_BoomerAMGSolve(amg_solver, (HYPRE_ParCSRMatrix)(A->matrix_data), (HYPRE_ParVector)(b[i]->vector_data), (HYPRE_ParVector)(x[i]->vector_data));
  }

  if(NULL == amg_data) {
    HYPRE_BoomerAMGDestroy(amg_solver);
  }

  return 0;
}
#endif
