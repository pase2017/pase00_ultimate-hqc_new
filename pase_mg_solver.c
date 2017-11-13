#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "pase_mg_solver.h"
#include "pase_pcg.h"
#include "HYPRE_utilities.h"
#include "HYPRE_lobpcg.h"
#include "lobpcg.h"

//PASE_MG_SOLVER
//PASE_Mg_solver_create(PASE_MATRIX A, PASE_MATRIX B, PASE_VECTOR x, PASE_PARAMETER param, PASE_MULTIGRID_OPERATOR ops)


PASE_MG_SOLVER
PASE_Mg_solver_create_by_multigrid(PASE_MULTIGRID multigrid)
{
    PASE_MG_SOLVER solver = (PASE_MG_SOLVER)PASE_Malloc(sizeof(PASE_MG_SOLVER_PRIVATE));

    solver->multigrid         = multigrid;
    solver->function          = NULL;

    solver->block_size        = 1;
    solver->pre_iter          = 1;
    solver->post_iter         = 1;
    solver->max_iter          = 1;
    solver->max_level         = multigrid->actual_level-1;
    solver->cur_level         = 0;
    solver->rtol              = 1e-8;
    solver->atol              = 1e-8;
    solver->r_norm            = 1e+10;
    solver->num_converged     = 0;
    solver->num_iter          = 0;
    solver->print_level       = 2;

    solver->eigenvalues       = NULL;
    solver->exact_eigenvalues = NULL;
    solver->u                 = NULL;
    solver->aux_u             = NULL;

    return solver;
}

PASE_INT 
PASE_Mg_solver_destroy(PASE_MG_SOLVER solver)
{
    PASE_INT i, j;
    if(solver) {
	//if(solver->multigrid) {
	//    PASE_Multigrid_destroy(solver->multigrid);
	//    solver->multigrid = NULL;
	//}
	if(solver->aux_u) {
	    for(i=0; i<=solver->max_level; i++) {
		if(solver->aux_u[i]){
		    for(j=0; j<solver->block_size; j++) {
                        if(solver->aux_u[i][j]) {
			    PASE_Aux_vector_destroy(solver->aux_u[i][j]);
			    solver->aux_u[i][j] = NULL;
			}
		    }
		    PASE_Free(solver->aux_u[i]);
		    solver->aux_u[i] = NULL;
		}
	    }
	    PASE_Free(solver->aux_u);
	    solver->aux_u = NULL;
	}
	if(solver->eigenvalues) {
	    PASE_Free(solver->eigenvalues);
	    solver->eigenvalues = NULL;
	}
	if(solver->u) {
	    for(j=0; j<solver->block_size; j++) {
		if(solver->u[j]) {
		    PASE_Vector_destroy(solver->u[j]);
		    solver->u[j] = NULL;
		}
	    }
	    PASE_Free(solver->u);
	    solver->u = NULL;
	}

	PASE_Free(solver);
	solver = NULL;
    }
    return 0;
}

PASE_INT
PASE_Mg_set_up(PASE_MG_SOLVER solver)
{
    PASE_INT level;
    solver->eigenvalues = (PASE_SCALAR*)PASE_Malloc(solver->block_size*sizeof(PASE_SCALAR));
    solver->aux_u       = (PASE_AUX_VECTOR**)PASE_Malloc((solver->max_level+1)*sizeof(PASE_AUX_VECTOR*));
    for(level=0; level<=solver->max_level; level++) {
	solver->aux_u[level] = NULL;
    }
    PASE_Get_initial_vector(solver);
    return 0;
}


/**
 * @brief MG求解 
 */
PASE_INT 
PASE_Mg_solve(PASE_MG_SOLVER solver)
{
   do {
       solver->num_iter ++;
       PASE_Mg_iteration(solver);
       PASE_Mg_error_estimate(solver);
   } while( solver->max_iter > solver->num_iter && solver->num_converged < solver->block_size);

   return 0;
}

PASE_INT 
PASE_Mg_iteration(PASE_MG_SOLVER solver)
{
   PASE_INT cur_level = solver->cur_level;
   PASE_INT max_level = solver->max_level;

   if( cur_level < max_level)
   {
       printf("cur_level = %d, max_level = %d\n", solver->cur_level, solver->max_level);
       /*前光滑*/
       printf("PreSmoothing..........\n");
       PASE_Mg_presmoothing(solver);
       printf("Creating AuxMatrix..........\n");
       PASE_Mg_pre_set_up(solver);
       
       /*粗空间校正*/
       printf("Correction on low-dim space\n");
       solver->cur_level++;
       PASE_Mg_iteration(solver);
       solver->cur_level--;

       /*后光滑*/
       printf("PostCorrecting..........\n");
       PASE_Mg_post_correction(solver);
       printf("PostSmoothing..........\n");
       PASE_Mg_postsmoothing(solver);
   }
   else if( cur_level == max_level)
   {
       //solver->function->solve_directly(solver);
       PASE_Mg_solve_directly_by_lobpcg_aux_hypre(solver);
   }
   return 0;
}

PASE_INT 
PASE_Mg_error_estimate(PASE_MG_SOLVER solver)
{
    PASE_INT         block_size	 = solver->block_size; 
    PASE_VECTOR     *u0	         = solver->u;
    PASE_SCALAR     *eigenvalues = solver->eigenvalues;
    PASE_MATRIX      A0	         = solver->multigrid->A[0];
    PASE_MATRIX      B0	         = solver->multigrid->B[0];

    /* 计算最细层的残差：r = Au - kMu */
    PASE_REAL        atol        = solver->atol;
    //PASE_REAL rtol = solver->rtol;
    PASE_INT         flag 	 = 0;
    PASE_INT         i		 = 0;
    PASE_REAL        r_norm 	 = 0;
    PASE_VECTOR      r           = PASE_Vector_create_by_vector(u0[0]);

    while( solver->num_converged < block_size && flag == 0) {
	i = solver->num_converged;
	PASE_Matrix_multiply_vector(A0, u0[i], r);
	PASE_Matrix_multiply_vector_general(-eigenvalues[i], B0, u0[i], 1.0, r); 
	PASE_Vector_inner_product(r, r, &r_norm);
	r_norm	= sqrt(r_norm);
	solver->r_norm = r_norm;
	if( r_norm < atol) {
	    solver->num_converged ++;
	} else {
	    flag = 1;
	}
    }
    PASE_Vector_destroy(r);

    if( solver->print_level > 0) {
	PASE_REAL error = 0.0; 
	//PASE_REAL error = fabs(solver->eigenvalues[0] - solver->exact_eigenvalues[0]);	
	printf("iter = %d, error of eigen[0] = %1.6e, the num of converged = %d, the norm of residul = %1.6e\n", solver->num_iter, error, solver->num_converged, solver->r_norm);
    }	
    return 0;
}

PASE_INT 
PASE_Mg_presmoothing(PASE_MG_SOLVER solver)
{
    if(solver->cur_level == 0)
    {
	//solver->function->presmoothing(solver);
        PASE_Mg_presmoothing_by_pcg_hypre(solver);
    }
    else
    {
	//solver->function->presmoothing_aux(solver);
        PASE_Mg_presmoothing_by_pcg_aux_hypre(solver);
    }
    return 0;
}

PASE_INT 
PASE_Mg_postsmoothing(PASE_MG_SOLVER solver)
{
    if(solver->cur_level == 0)
    {
	//solver->function->postsmoothing(solver);
        PASE_Mg_presmoothing_by_pcg_hypre(solver);
    }
    else
    {
	//solver->function->postsmoothing_aux(solver);
        PASE_Mg_presmoothing_by_pcg_aux_hypre(solver);
    }
    return 0;
}

PASE_INT 
PASE_Mg_pre_set_up(PASE_MG_SOLVER solver)
{
    PASE_INT         cur_level  = solver->cur_level;
    PASE_INT         block_size = solver->block_size;

    PASE_Mg_set_aux_matrix(solver);

    PASE_INT eigen_index = 0;
    if(NULL == solver->aux_u[cur_level+1]) {
	solver->aux_u[cur_level+1] = (PASE_AUX_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_AUX_VECTOR));
	solver->aux_u[cur_level+1][0] = PASE_Aux_vector_create_by_aux_matrix(solver->multigrid->aux_A[cur_level+1]);
	for(eigen_index=1; eigen_index<block_size; eigen_index++) {
	    solver->aux_u[cur_level+1][eigen_index] = PASE_Aux_vector_create_by_aux_vector(solver->aux_u[cur_level+1][0]);
	}
    }

    /*多次迭代需要多次初始化初值，但空间不需要重新申请*/
    for(eigen_index=0; eigen_index<block_size; eigen_index++) {
	PASE_Aux_vector_set_constant_value(solver->aux_u[cur_level+1][eigen_index], 0.0);
	solver->aux_u[cur_level+1][eigen_index]->block[eigen_index] = 1.0;
    }
    //HYPRE_ParVectorPrint(solver->aux_u[cur_level+1][0]->vec->vector_data, "b_H");

    return 0;
}

PASE_INT 
PASE_Mg_set_aux_matrix(PASE_MG_SOLVER solver)
{
    PASE_INT         cur_level  = solver->cur_level;
    PASE_INT         block_size = solver->block_size;

    PASE_MATRIX      A_H        = solver->multigrid->A[cur_level+1];
    PASE_MATRIX      A_h        = solver->multigrid->A[cur_level];
    PASE_MATRIX      B_H        = solver->multigrid->B[cur_level+1];
    PASE_MATRIX      B_h        = solver->multigrid->B[cur_level];
    PASE_MATRIX      R_hH       = solver->multigrid->R[cur_level];
    PASE_AUX_MATRIX  aux_A_h    = solver->multigrid->aux_A[cur_level];
    PASE_AUX_MATRIX  aux_B_h    = solver->multigrid->aux_B[cur_level];
    PASE_AUX_VECTOR *aux_u_h    = solver->aux_u[cur_level];

    if(cur_level == 0) {
	if(NULL == solver->multigrid->aux_A[cur_level+1]) {
            solver->multigrid->aux_A[cur_level+1] = PASE_Aux_matrix_create(A_H, R_hH, A_h, solver->u, block_size);  
            solver->multigrid->aux_B[cur_level+1] = PASE_Aux_matrix_create(B_H, R_hH, B_h, solver->u, block_size);  
	} else {
	    PASE_Aux_matrix_set_aux_space_some(solver->multigrid->aux_A[cur_level+1], solver->num_converged, block_size-1, R_hH, A_h, solver->u);
	    PASE_Aux_matrix_set_aux_space_some(solver->multigrid->aux_B[cur_level+1], solver->num_converged, block_size-1, R_hH, B_h, solver->u);
	}
    } else {
	if(NULL == solver->multigrid->aux_A[cur_level+1]) {
            solver->multigrid->aux_A[cur_level+1] = PASE_Aux_matrix_create_by_aux_matrix(A_H, R_hH, aux_A_h, aux_u_h, block_size);  
            solver->multigrid->aux_B[cur_level+1] = PASE_Aux_matrix_create_by_aux_matrix(B_H, R_hH, aux_B_h, aux_u_h, block_size);  
	} else {
	    PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(solver->multigrid->aux_A[cur_level+1], solver->num_converged, block_size-1, R_hH, aux_A_h, aux_u_h);
	    PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(solver->multigrid->aux_B[cur_level+1], solver->num_converged, block_size-1, R_hH, aux_B_h, aux_u_h);
	}
    }

    return 0;
}

PASE_INT 
PASE_Mg_post_correction(PASE_MG_SOLVER solver)
{
    PASE_INT i, j;
    PASE_INT cur_level  = solver->cur_level;
    PASE_INT block_size = solver->block_size;
    PASE_MATRIX P_Hh    = solver->multigrid->P[cur_level];

    /* u_new->b_H += P*u0->b_H */
    /* u_new += u1*u_0->aux_h */
    if(cur_level == 0) {
	PASE_VECTOR *u_new = (PASE_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_VECTOR));
        for(i=0; i<block_size; i++) {
	    u_new[i] = PASE_Vector_create_by_vector(solver->u[0]);
            PASE_Matrix_multiply_vector(P_Hh , solver->aux_u[1][i]->vec, u_new[i]);
            for(j=0; j<block_size; j++) {
                PASE_Vector_add_vector(solver->aux_u[1][i]->block[j], solver->u[j], u_new[i]);
            }
        }
        for( i=0; i<block_size; i++) {
	    PASE_Vector_copy(u_new[i], solver->u[i]);
            PASE_Vector_destroy(u_new[i]);
        }
	PASE_Free(u_new);
    } else {
	PASE_AUX_VECTOR *u_new = (PASE_AUX_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_AUX_VECTOR));
        for(i=0; i<block_size; i++) {
	    u_new[i] = PASE_Aux_vector_create_by_aux_vector(solver->aux_u[cur_level][0]);
            PASE_Matrix_multiply_vector(P_Hh , solver->aux_u[cur_level+1][i]->vec, u_new[i]->vec);
            for(j=0; j<block_size; j++) {
                PASE_Aux_vector_add(solver->aux_u[cur_level+1][i]->block[j], solver->aux_u[cur_level][j], u_new[i]);
            }
        }
        for(i=0; i<block_size; i++) {
	    PASE_Aux_vector_copy(u_new[i], solver->aux_u[cur_level][i]);
            PASE_Aux_vector_destroy(u_new[i]);
        }
	PASE_Free(u_new);
    }

    return 0;
}

PASE_INT
PASE_Get_initial_vector(PASE_MG_SOLVER solver)
{
    HYPRE_Solver lobpcg_solver 	= NULL; 
    PASE_INT  maxIterations 	= 200; 	/* maximum number of iterations */
    PASE_INT  pcgMode 		= 1;    	/* use rhs as initial guess for inner pcg iterations */
    PASE_INT  verbosity 	= 0;    	/* print iterations info */
    PASE_REAL tol 		= 1.e-8;	/* absolute tolerance (all eigenvalues) */
    PASE_INT  lobpcgSeed 	= 77;

    PASE_INT i;
    PASE_INT max_level          = solver->max_level;        
    PASE_INT block_size         = solver->block_size;       

    PASE_MATRIX A = solver->multigrid->A[max_level];
    PASE_MATRIX B = solver->multigrid->B[max_level];
    PASE_VECTOR u_temp = PASE_Vector_create_by_matrix(A, NULL); 
    PASE_VECTOR x = PASE_Vector_create_by_vector(u_temp);
    PASE_SCALAR *eigenvalues 	= solver->eigenvalues;

    mv_MultiVectorPtr eigenvectors_Hh = NULL;
    mv_MultiVectorPtr constraints_Hh  = NULL;
    mv_InterfaceInterpreter* interpreter_Hh;
    HYPRE_MatvecFunctions matvec_fn_Hh;
    interpreter_Hh = hypre_CTAlloc(mv_InterfaceInterpreter, 1);
    PASE_Lobpcg_setup_interpreter(interpreter_Hh);
    PASE_Lobpcg_setup_matvec(&matvec_fn_Hh);
    eigenvectors_Hh = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, block_size, u_temp);
    mv_MultiVectorSetRandom(eigenvectors_Hh, lobpcgSeed);
     
    HYPRE_LOBPCGCreate(interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
    HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
    HYPRE_LOBPCGSetPrecondUsageMode(lobpcg_solver, pcgMode);
    HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
    HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

    hypre_LOBPCGSetup(lobpcg_solver, A, u_temp, x);
    hypre_LOBPCGSetupB(lobpcg_solver, B, u_temp);
    HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);

    if( solver->print_level > 1)
    {
	printf("Get initial vector");
	for( i=0; i<block_size; i++)
	{
	    printf(", eigen[%d] = %.16f", i, eigenvalues[i]);
	}
	printf(".\n");
    }

    mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
    PASE_VECTOR *u_H = (PASE_VECTOR*)(tmp->vector);

    PASE_VECTOR u_h = NULL; 
    PASE_INT j;
    for(j=max_level-1; j>=0; j--) {
	u_h = PASE_Vector_create_by_matrix(solver->multigrid->A[j], NULL);
        for(i=0; i<block_size; i++) {
	    if(i > 0) {
	        u_h = PASE_Vector_create_by_vector(u_H[i-1]);	
	    } 
	    PASE_Matrix_multiply_vector(solver->multigrid->P[j], u_H[i], u_h); 
	    PASE_Vector_destroy(u_H[i]);
	    u_H[i] = u_h;
	    u_h    = NULL;
        }
    }

    solver->u = u_H;
#if 0
    char filename[20];
    char filepre[20] = "Init_vec";
    for( i=0; i<block_size; i++)
    {
	sprintf(filename, "%s_%d", filepre, i);
	HYPRE_ParVectorPrint((HYPRE_ParCSRMatrix)(u_H->matrix_data), filename); 
	printf("eigenvalues[%d] = %.12e\n", i, eigenvalues[i]);
    }
#endif

    PASE_Free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh));
    PASE_Free(eigenvectors_Hh);
    PASE_Free(interpreter_Hh);
    PASE_Vector_destroy(u_temp);
    PASE_Vector_destroy(x);

    HYPRE_LOBPCGDestroy( lobpcg_solver);
    return 0;
}

PASE_INT 
PASE_Mg_presmoothing_by_pcg_hypre(PASE_MG_SOLVER solver)
{
    HYPRE_Solver cg_solver = NULL;
    PASE_Pcg_create(MPI_COMM_WORLD, &cg_solver);
    HYPRE_PCGSetMaxIter(cg_solver, solver->pre_iter); /* max iterations */
    HYPRE_PCGSetTol(cg_solver, 1.0e-50); 
    HYPRE_PCGSetTwoNorm(cg_solver, 1); /* use the two norm as the st    opping criteria */
    HYPRE_PCGSetPrintLevel(cg_solver, 0); 
    HYPRE_PCGSetLogging(cg_solver, 1); /* needed to get run info lat    er */
           
    PASE_SCALAR  inner_A, inner_B;
    PASE_INT     cur_level 	= solver->cur_level;
    PASE_INT     block_size	= solver->block_size;
    PASE_INT     num_converged  = solver->num_converged; 
    PASE_INT     i		= 0;

    PASE_MATRIX  A              = solver->multigrid->A[cur_level];            
    PASE_MATRIX  B              = solver->multigrid->B[cur_level];            
    PASE_VECTOR *u              = solver->u;
    PASE_SCALAR *eigenvalues 	= solver->eigenvalues;
    PASE_VECTOR  rhs 		= PASE_Vector_create_by_vector(u[0]);
    for( i=num_converged; i<block_size; i++) {
	PASE_Matrix_multiply_vector(B, u[i], rhs);
	PASE_Vector_scale(eigenvalues[i], rhs);
	hypre_PCGSetup(cg_solver, A, rhs, u[i]);
	hypre_PCGSolve(cg_solver, A, rhs, u[i]);

	PASE_Matrix_multiply_vector(A, u[i], rhs);
	PASE_Vector_inner_product(rhs, u[i], &inner_A);
	PASE_Matrix_multiply_vector(B, u[i], rhs);
        PASE_Vector_inner_product(rhs, u[i], &inner_B);
	eigenvalues[i] = inner_A / inner_B;
    }

    if(solver->print_level > 1) {
	printf("Cur_level %d", cur_level);
	for(i=0; i<block_size; i++) {
	    printf(", eigen[%d] = %.16f", i, eigenvalues[i]);
	}
	printf(".\n");
    }
    PASE_Vector_destroy(rhs);
    HYPRE_ParCSRPCGDestroy(cg_solver);
    return 0;
}

PASE_INT 
PASE_Mg_presmoothing_by_pcg_aux_hypre(PASE_MG_SOLVER solver)
{
    HYPRE_Solver cg_solver = NULL;
    PASE_Pcg_create_aux(MPI_COMM_WORLD, &cg_solver);
    HYPRE_PCGSetMaxIter(cg_solver, solver->pre_iter); /* max iterations */
    HYPRE_PCGSetTol(cg_solver, 1.0e-50); 
    HYPRE_PCGSetTwoNorm(cg_solver, 1); /* use the two norm as the st    opping criteria */
    HYPRE_PCGSetPrintLevel(cg_solver, 0); 
    HYPRE_PCGSetLogging(cg_solver, 1); /* needed to get run info lat    er */
           
    PASE_SCALAR  inner_A, inner_B;
    PASE_INT     cur_level 	= solver->cur_level;
    PASE_INT     block_size	= solver->block_size;
    PASE_INT     num_converged  = solver->num_converged; 
    PASE_INT     i		= 0;

    PASE_AUX_MATRIX  aux_A       = solver->multigrid->aux_A[cur_level];            
    PASE_AUX_MATRIX  aux_B       = solver->multigrid->aux_B[cur_level];            
    PASE_AUX_VECTOR *aux_u       = solver->aux_u[cur_level];
    PASE_SCALAR     *eigenvalues = solver->eigenvalues;
    PASE_AUX_VECTOR  rhs         = PASE_Aux_vector_create_by_aux_vector(aux_u[0]);
    for( i=num_converged; i<block_size; i++) {
	PASE_Aux_matrix_multiply_aux_vector(aux_B, aux_u[i], rhs);
	PASE_Aux_vector_scale(eigenvalues[i], rhs);
	hypre_PCGSetup(cg_solver, aux_A, rhs, aux_u[i]);
	hypre_PCGSolve(cg_solver, aux_A, rhs, aux_u[i]);

	PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_u[i], rhs);
	PASE_Aux_vector_inner_product(rhs, aux_u[i], &inner_A);
	PASE_Aux_matrix_multiply_aux_vector(aux_B, aux_u[i], rhs);
        PASE_Aux_vector_inner_product(rhs, aux_u[i], &inner_B);
	eigenvalues[i] = inner_A / inner_B;
    }

    if(solver->print_level > 1) {
	printf("Cur_level %d", cur_level);
	for(i=0; i<block_size; i++) {
	    printf(", eigen[%d] = %.16f", i, eigenvalues[i]);
	}
	printf(".\n");
    }
    PASE_Aux_vector_destroy(rhs);
    HYPRE_ParCSRPCGDestroy(cg_solver);
    return 0;
}

PASE_INT 
PASE_Mg_solve_directly_by_lobpcg_aux_hypre(PASE_MG_SOLVER solver)
{
    HYPRE_Solver lobpcg_solver 	= NULL; 
    PASE_INT     maxIterations 	= 5000; 	/* maximum number of iterations */
    PASE_INT     pcgMode 	= 1;    	/* use rhs as initial guess for inner pcg iterations */
    PASE_INT     verbosity 	= 0;    	/* print iterations info */
    PASE_REAL    tol 		= 1.e-30;	/* absolute tolerance (all eigenvalues) */

    PASE_INT i;
    PASE_INT block_size 	= solver->block_size;
    PASE_INT cur_level 		= solver->cur_level;

    PASE_AUX_MATRIX  aux_A      = solver->multigrid->aux_A[cur_level];            
    PASE_AUX_MATRIX  aux_B      = solver->multigrid->aux_B[cur_level];            
    PASE_AUX_VECTOR *aux_u      = solver->aux_u[cur_level];
    PASE_SCALAR *eigenvalues    = solver->eigenvalues;
    //printf("block[0] = %.6e\n", aux_u[0]->block[0]);
    //HYPRE_ParVectorPrint(aux_u[0]->vec->vector_data, "aux_u");
    PASE_AUX_VECTOR  x        = PASE_Aux_vector_create_by_aux_vector(aux_u[0]);

    mv_MultiVectorPtr eigenvectors_Hh = NULL;
    mv_MultiVectorPtr constraints_Hh  = NULL;
    mv_InterfaceInterpreter* interpreter_Hh;
    HYPRE_MatvecFunctions matvec_fn_Hh;
    interpreter_Hh = hypre_CTAlloc( mv_InterfaceInterpreter, 1);
    PASE_Lobpcg_setup_interpreter_aux( interpreter_Hh);
    PASE_Lobpcg_setup_matvec_aux( &matvec_fn_Hh);

    eigenvectors_Hh = mv_MultiVectorCreateFromSampleVector( interpreter_Hh, block_size, aux_u[0]);
    //mv_MultiVectorSetRandom( eigenvectors_Hh, 77);
    mv_TempMultiVector* tmp = (mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh);
    solver->aux_u[cur_level] = (PASE_AUX_VECTOR*)(tmp -> vector);
    for(i=0; i<block_size; i++) {
        PASE_Aux_vector_destroy(solver->aux_u[cur_level][i]);
        solver->aux_u[cur_level][i] = aux_u[i];
    }
     
    HYPRE_LOBPCGCreate( interpreter_Hh, &matvec_fn_Hh, &lobpcg_solver);
    HYPRE_LOBPCGSetMaxIter(lobpcg_solver, maxIterations);
    HYPRE_LOBPCGSetPrecondUsageMode( lobpcg_solver, pcgMode);
    HYPRE_LOBPCGSetTol(lobpcg_solver, tol);
    HYPRE_LOBPCGSetPrintLevel(lobpcg_solver, verbosity);

    hypre_LOBPCGSetup(lobpcg_solver, aux_A, aux_u[0], x);
    hypre_LOBPCGSetupB(lobpcg_solver, aux_B, aux_u[0]);
    HYPRE_LOBPCGSolve(lobpcg_solver, constraints_Hh, eigenvectors_Hh, eigenvalues);

    
    if( solver->print_level > 1) {
	printf("Cur_level %d", cur_level);
	for( i=0; i<block_size; i++) {
	    printf(", eigen[%d] = %.16f", i, eigenvalues[i]);
	}
	printf(".\n");
    }

    PASE_Free(aux_u);
    PASE_Free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh));
    PASE_Free( eigenvectors_Hh);
    PASE_Free( interpreter_Hh);

    PASE_Aux_vector_destroy(x);
    HYPRE_LOBPCGDestroy( lobpcg_solver);
    return 0;
}

