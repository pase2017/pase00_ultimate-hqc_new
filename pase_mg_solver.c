#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "pase_mg_solver.h"
#include "pase_pcg.h"
#include "pase_arnoldi.h"
#include "HYPRE_utilities.h"
#include "HYPRE_lobpcg.h"
#include "lobpcg.h"

#define CLK_TCK 1000000

//PASE_MG_SOLVER
//PASE_Mg_solver_create(PASE_MATRIX A, PASE_MATRIX B, PASE_VECTOR x, PASE_PARAMETER param, PASE_MULTIGRID_OPERATOR ops)

/**
 * @brief PASE_MG_FUNCTION 的创建
 */
PASE_MG_FUNCTION
PASE_Mg_function_create(PASE_INT (*get_initial_vector) (void *solver),
	                PASE_INT (*direct_solve)       (void *solver),
                        PASE_INT (*presmoothing)       (void *solver), 
                        PASE_INT (*postsmoothing)      (void *solver), 
                        PASE_INT (*presmoothing_aux)   (void *solver), 
                        PASE_INT (*postsmoothing_aux)  (void *solver)) 
{
    PASE_MG_FUNCTION function    = (PASE_MG_FUNCTION)PASE_Malloc(sizeof(PASE_MG_FUNCTION_PRIVATE));
    function->get_initial_vector = get_initial_vector;
    function->direct_solve       = direct_solve;
    function->presmoothing       = presmoothing;
    function->postsmoothing      = postsmoothing;
    function->presmoothing_aux   = presmoothing_aux;
    function->postsmoothing_aux  = postsmoothing_aux;
    return function;
}

/**
 * @brief 依据给定的 PASE_MULTIGRID 创建 PASE_MG_SOLVER, 并对其中一些成员赋予默认的初始值.
 */
PASE_MG_SOLVER
PASE_Mg_solver_create_by_multigrid(PASE_MULTIGRID multigrid)
{
    PASE_MG_SOLVER solver      = (PASE_MG_SOLVER)PASE_Malloc(sizeof(PASE_MG_SOLVER_PRIVATE));

    solver->multigrid          = multigrid;
    solver->function           = PASE_Mg_function_create(PASE_Mg_get_initial_vector_by_coarse_grid,
	                                                 //PASE_Mg_solve_directly_by_IRA,
							 PASE_Mg_solve_directly_by_lobpcg_aux_hypre,
                                                         PASE_Mg_presmoothing_by_cg,
                                                         PASE_Mg_presmoothing_by_cg,
			                                 PASE_Mg_presmoothing_by_cg_aux,
			                                 PASE_Mg_presmoothing_by_cg_aux);
    solver->block_size         = 1;
    solver->actual_block_size  = 1;
    solver->pre_iter           = 1;
    solver->post_iter          = 1;
    solver->max_iter           = 30;
    solver->max_level          = multigrid->actual_level-1;
    solver->cur_level          = 0;
    solver->rtol               = 1e-8;
    solver->atol               = 1e-8;
    solver->r_norm             = NULL;
    solver->num_converged      = 0;
    solver->last_num_converged = 0;
    solver->num_iter           = 0;
    solver->print_level        = 1;

    solver->set_up_time        = 0.0;
    solver->smooth_time        = 0.0;
    solver->set_aux_time       = 0.0;
    solver->prolong_time       = 0.0;
    solver->direct_solve_time  = 0.0;
    solver->eigenvalues        = NULL;
    solver->exact_eigenvalues  = NULL;
    solver->u                  = NULL;
    solver->aux_u              = NULL;
    return solver;
}

/**
 * @brief 销毁 PASE_MG_SOLVER 并释放内存空间.
 */
PASE_INT 
PASE_Mg_solver_destroy(PASE_MG_SOLVER solver)
{
    PASE_INT i, j;
    if(solver) {
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
	if(solver->r_norm) {
	    PASE_Free(solver->r_norm);
	    solver->r_norm= NULL;
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
	if(solver->function) {
	    PASE_Free(solver->function);
	    solver->function= NULL;
	}

	PASE_Free(solver);
	solver = NULL;
    }
    return 0;
}

/**
 * @brief PASE_MG_SOLVER 的初始化，包括：
 *            1. solver->aux_u 数组的创建, 数组里的具体元素将在 PASE_Mg_pre_set_up 里面进行空间的申请和初始化;
 *            2. solver->u 和 solver->eigenvalues 将在初始化函数 solver->function->get_initial_vector 里申请空间, 并得到初始值.
 */
PASE_INT
PASE_Mg_set_up(PASE_MG_SOLVER solver)
{
    clock_t start, end;
    start = clock();
    PASE_INT level;
    //solver->eigenvalues = (PASE_SCALAR*)PASE_Malloc(solver->block_size*sizeof(PASE_SCALAR));
    solver->aux_u       = (PASE_AUX_VECTOR**)PASE_Malloc((solver->max_level+1)*sizeof(PASE_AUX_VECTOR*));
    for(level=0; level<=solver->max_level; level++) {
	solver->aux_u[level] = NULL;
    }

    solver->function->get_initial_vector(solver);
    end = clock();
    solver->set_up_time += ((double)(end-start))/CLK_TCK;
    return 0;
}


/**
 * @brief MG求解, 主要通过迭代 PASE_Mg_iteration 函数实现. 
 */
PASE_INT 
PASE_Mg_solve(PASE_MG_SOLVER solver)
{
   do {
       solver->num_iter ++;
       PASE_Mg_iteration(solver);
       PASE_Mg_error_estimate(solver);
   } while( solver->max_iter > solver->num_iter && solver->num_converged < solver->actual_block_size);
   PASE_Mg_print(solver);

   return 0;
}

/**
 * @brief MG 方法的主体，采用递归定义，每层上主要包含分成三步：前光滑，粗空间校正，后光滑. 
 */
PASE_INT 
PASE_Mg_iteration(PASE_MG_SOLVER solver)
{
   PASE_INT cur_level = solver->cur_level;
   PASE_INT max_level = solver->max_level;

   if( cur_level < max_level)
   {
       //printf("cur_level = %d, max_level = %d\n", solver->cur_level, solver->max_level);
       /*前光滑*/
       //printf("PreSmoothing..........\n");
       PASE_Mg_presmoothing(solver);
       //PASE_Mg_orth(solver);
       //printf("Creating AuxMatrix..........\n");
       PASE_Mg_pre_set_up(solver);
       
       /*粗空间校正*/
       //printf("Correction on low-dim space\n");
       solver->cur_level++;
       PASE_Mg_iteration(solver);
       solver->cur_level--;

       /*后光滑*/
       //printf("PostCorrecting..........\n");
       PASE_Mg_prolong(solver);
       //printf("PostSmoothing..........\n");
       PASE_Mg_postsmoothing(solver);
       //PASE_Mg_orth(solver);
   }
   else if( cur_level == max_level)
   {
       clock_t start, end;
       start = clock();
       solver->function->direct_solve(solver);
       end = clock();
       solver->direct_solve_time += ((double)(end-start))/CLK_TCK;
   }
   return 0;
}

/**
 * @brief 完成一次 PASE_Mg_iteration 后, 需计算残差及已收敛特征对个数. 已收敛特征对在之后的迭代中，不再计算和更改. 
 */
PASE_INT 
PASE_Mg_error_estimate(PASE_MG_SOLVER solver)
{
    PASE_INT         block_size	   = solver->block_size; 
    PASE_INT         num_converged = solver->num_converged;
    PASE_REAL        atol          = solver->atol;
    PASE_VECTOR     *u0	           = solver->u;
    PASE_SCALAR     *eigenvalues   = solver->eigenvalues;
    PASE_MATRIX      A0	           = solver->multigrid->A[0];
    PASE_MATRIX      B0	           = solver->multigrid->B[0];

    /* 计算最细层的残差：r = Au - kMu */
    PASE_INT         flag          = 0;
    PASE_REAL       *check_multi   = (PASE_REAL*)PASE_Malloc((block_size-1)*sizeof(PASE_REAL));
    //PASE_REAL rtol = solver->rtol;
    PASE_INT         i		   = 0;
    PASE_REAL        r_norm        = 1e+5;
    PASE_VECTOR      r             = PASE_Vector_create_by_vector(u0[0]);
    solver->last_num_converged     = num_converged;

    if(NULL == solver->r_norm) {
	solver->r_norm = (PASE_REAL*)PASE_Malloc(block_size*sizeof(PASE_REAL));
    }

    for(i=num_converged; i<block_size; i++) {
	PASE_Matrix_multiply_vector(A0, u0[i], r);
	PASE_Matrix_multiply_vector_general(-eigenvalues[i], B0, u0[i], 1.0, r); 
	PASE_Vector_inner_product(r, r, &r_norm);
	r_norm	= sqrt(r_norm);
	solver->r_norm[i] = r_norm;
	if(i+1<block_size) {
	    check_multi[i] = fabs((eigenvalues[i]-eigenvalues[i+1])/eigenvalues[i]);
	    //printf("check_multi[%d] = %e\n", i, check_multi[i-num_converged]);
	}
	if( r_norm < atol && flag == 0) {
	    solver->num_converged ++;
	} else {
	    flag = 1;
	}
    }
    //检查第一个为收敛的特征值与最后一个刚收敛的特征值是否有可能是重特征值，未保证之后的排序问题，需让重特征值同时在收敛的集合或未收敛的集合.
    if(solver->num_converged > num_converged && solver->num_converged < block_size) {
        while(check_multi[solver->num_converged-1] < 1e-8 && solver->num_converged > num_converged) {
            solver->num_converged --;
        }
    }
    PASE_Vector_destroy(r);

    PASE_INT myid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if(solver->print_level > 0 && myid == 0) {
	//PASE_REAL error = fabs(solver->eigenvalues[0] - solver->exact_eigenvalues[0]);	
	printf("iter = %d, nconv = %d, ", solver->num_iter, solver->num_converged);
	if(solver->num_converged < solver->block_size) {
	    printf("residual of the first unconverged = %1.6e\n", solver->r_norm[solver->num_converged]);
	} else {
	    printf("all the wanted eigenpairs have converged.\n");
	}
	//printf("smooth time = %.4e\n", solver->smooth_time);
    }	

    return 0;
}

/**
 * @brief 打印计算所得特征值，及其对应的残差.
 */
PASE_INT
PASE_Mg_print(PASE_MG_SOLVER solver)
{
    PASE_INT myid, idx_eigen;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if(solver->print_level > 0 && myid == 0) {
	printf("\n");
	printf("=============================================================\n");
	for(idx_eigen=0; idx_eigen<solver->block_size; idx_eigen++) {
	    printf("%d-th eig=%.8e, aresi = %.8e\n", idx_eigen, solver->eigenvalues[idx_eigen], solver->r_norm[idx_eigen]);
	}
	printf("=============================================================\n");
        printf("set up time       = %f seconds\n", solver->set_up_time);
        printf("smooth time       = %f seconds\n", solver->smooth_time);
        printf("set aux time      = %f seconds\n", solver->set_aux_time);
        printf("prolong time      = %f seconds\n", solver->prolong_time);
        printf("direct solve time = %f seconds\n", solver->direct_solve_time);
    }	
    return 0;
}

/**
 * @brief 前光滑函数
 */
PASE_INT 
PASE_Mg_presmoothing(PASE_MG_SOLVER solver)
{
    clock_t start, end;
    start = clock();
    if(solver->cur_level == 0) {
	solver->function->presmoothing(solver);
    } else {
	solver->function->presmoothing_aux(solver);
    }
    end = clock();
    solver->smooth_time += ((double)(end-start))/CLK_TCK;

    return 0;
}

/**
 * @brief 后光滑函数
 */
PASE_INT 
PASE_Mg_postsmoothing(PASE_MG_SOLVER solver)
{
    clock_t start, end;
    start = clock();
    if(solver->cur_level == 0) {
	solver->function->postsmoothing(solver);
        PASE_Mg_orth(solver);
    } else {
	solver->function->postsmoothing_aux(solver);
    }
    end = clock();
    solver->smooth_time += ((double)(end-start))/CLK_TCK;
    return 0;
}

/**
 * @brief 设置更粗层辅助空间的矩阵, 及前光滑所需的初始向量.
 */
PASE_INT 
PASE_Mg_pre_set_up(PASE_MG_SOLVER solver)
{
    PASE_INT         cur_level  = solver->cur_level;
    PASE_INT         block_size = solver->block_size;

    clock_t start, end;
    start = clock();
    PASE_Mg_set_aux_matrix(solver);
    end = clock();
    solver->set_aux_time += ((double)(end-start))/CLK_TCK;

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
	//PASE_Aux_vector_set_constant_value(solver->aux_u[cur_level+1][eigen_index], 0.0);
	/* 对double型的数组用menset置零，提高效率 */
	PASE_Vector_set_constant_value(solver->aux_u[cur_level+1][eigen_index]->vec, 0.0);
	memset(solver->aux_u[cur_level+1][eigen_index]->block, 0, block_size*sizeof(PASE_SCALAR));
	solver->aux_u[cur_level+1][eigen_index]->block[eigen_index] = 1.0;
    }

    return 0;
}

/**
 * @brief 设置更粗层辅助空间的矩阵.
 */
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
    //printf("last_num_converged = %d\n", solver->last_num_converged);

#if 1
    if(cur_level == 0) {
	if(NULL == solver->multigrid->aux_A[cur_level+1]) {
            solver->multigrid->aux_A[cur_level+1] = PASE_Aux_matrix_create(A_H, R_hH, A_h, solver->u, block_size);  
            solver->multigrid->aux_B[cur_level+1] = PASE_Aux_matrix_create(B_H, R_hH, B_h, solver->u, block_size);  
	} else {
	    PASE_Aux_matrix_set_aux_space_some(solver->multigrid->aux_A[cur_level+1], solver->last_num_converged, block_size-1, R_hH, A_h, solver->u);
	    PASE_Aux_matrix_set_aux_space_some(solver->multigrid->aux_B[cur_level+1], solver->last_num_converged, block_size-1, R_hH, B_h, solver->u);
	}
    } else {
	if(NULL == solver->multigrid->aux_A[cur_level+1]) {
            solver->multigrid->aux_A[cur_level+1] = PASE_Aux_matrix_create_by_aux_matrix(A_H, R_hH, aux_A_h, aux_u_h, block_size);  
            solver->multigrid->aux_B[cur_level+1] = PASE_Aux_matrix_create_by_aux_matrix(B_H, R_hH, aux_B_h, aux_u_h, block_size);  
	} else {
	    PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(solver->multigrid->aux_A[cur_level+1], solver->last_num_converged, block_size-1, R_hH, aux_A_h, aux_u_h);
	    PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(solver->multigrid->aux_B[cur_level+1], solver->last_num_converged, block_size-1, R_hH, aux_B_h, aux_u_h);
	}
    }
#else
    if(cur_level == 0) {
	if(NULL == solver->multigrid->aux_A[cur_level+1]) {
            solver->multigrid->aux_A[cur_level+1] = PASE_Aux_matrix_create(A_H, R_hH, A_h, solver->u, block_size);  
            solver->multigrid->aux_B[cur_level+1] = PASE_Aux_matrix_create(B_H, R_hH, B_h, solver->u, block_size);  
	} else {
	    PASE_Aux_matrix_set_aux_space_some(solver->multigrid->aux_A[cur_level+1], 0, block_size-1, R_hH, A_h, solver->u);
	    PASE_Aux_matrix_set_aux_space_some(solver->multigrid->aux_B[cur_level+1], 0, block_size-1, R_hH, B_h, solver->u);
	}
    } else {
	if(NULL == solver->multigrid->aux_A[cur_level+1]) {
            solver->multigrid->aux_A[cur_level+1] = PASE_Aux_matrix_create_by_aux_matrix(A_H, R_hH, aux_A_h, aux_u_h, block_size);  
            solver->multigrid->aux_B[cur_level+1] = PASE_Aux_matrix_create_by_aux_matrix(B_H, R_hH, aux_B_h, aux_u_h, block_size);  
	} else {
	    PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(solver->multigrid->aux_A[cur_level+1], 0, block_size-1, R_hH, aux_A_h, aux_u_h);
	    PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(solver->multigrid->aux_B[cur_level+1], 0, block_size-1, R_hH, aux_B_h, aux_u_h);
	}
    }
#endif

    return 0;
}
 
/**
 * @brief 将粗层的辅助空间向量插值到细层空间.
 */
PASE_INT 
PASE_Mg_prolong(PASE_MG_SOLVER solver)
{
    clock_t start, end;
    start = clock();
    PASE_INT i;
    PASE_INT cur_level  = solver->cur_level;
    PASE_INT block_size = solver->block_size;
    PASE_MATRIX P_Hh    = solver->multigrid->P[cur_level];
#if 1
    PASE_INT num_converged = solver->num_converged;

    /* u_new += u1*u_0->aux_h */
    /* u_new->b_H += P*u0->b_H */
    if(cur_level == 0) {
	PASE_VECTOR *u_new = (PASE_VECTOR*)PASE_Malloc((block_size-num_converged)*sizeof(PASE_VECTOR));
        for(i=num_converged; i<block_size; i++) {
	    u_new[i-num_converged] = PASE_Vector_create_by_vector(solver->u[0]);
	    PASE_Multi_vector_combination(solver->u, block_size, solver->aux_u[1][i]->block, u_new[i-num_converged]);
            PASE_Matrix_multiply_vector_general(1.0, P_Hh , solver->aux_u[1][i]->vec, 1.0, u_new[i-num_converged]);
        }
        for( i=num_converged; i<block_size; i++) {
	    PASE_Vector_copy(u_new[i-num_converged], solver->u[i]);
            PASE_Vector_destroy(u_new[i-num_converged]);
        }
	PASE_Free(u_new);
    } else {
	PASE_AUX_VECTOR *u_new = (PASE_AUX_VECTOR*)PASE_Malloc((block_size-num_converged)*sizeof(PASE_AUX_VECTOR));
        for(i=num_converged; i<block_size; i++) {
	    u_new[i-num_converged] = PASE_Aux_vector_create_by_aux_vector(solver->aux_u[cur_level][0]);
	    PASE_Multi_aux_vector_combination(solver->aux_u[cur_level], block_size, solver->aux_u[cur_level+1][i]->block, u_new[i-num_converged]);
            PASE_Matrix_multiply_vector_general(1.0, P_Hh , solver->aux_u[cur_level+1][i]->vec, 1.0, u_new[i-num_converged]->vec);
        }
        for(i=num_converged; i<block_size; i++) {
	    PASE_Aux_vector_copy(u_new[i-num_converged], solver->aux_u[cur_level][i]);
            PASE_Aux_vector_destroy(u_new[i-num_converged]);
        }
	PASE_Free(u_new);
    }
#else 
#endif
    end = clock();
    solver->prolong_time += ((double)(end-start))/CLK_TCK;

    return 0;
}

/**
 * @brief 正交化函数.
 */
PASE_INT
PASE_Mg_orth(PASE_MG_SOLVER solver)
{
    PASE_INT cur_level 	= solver->cur_level;
    PASE_INT block_size	= solver->block_size;
    PASE_INT num_converged = solver->num_converged;

    PASE_INT idx_eigen;

    if(cur_level == 0) {
	PASE_Vector_orth_general(solver->u, num_converged, block_size, solver->multigrid->B[0]);
	for(idx_eigen=0; idx_eigen<block_size; idx_eigen++) {
	    PASE_Vector_inner_product_general(solver->u[idx_eigen], solver->u[idx_eigen], solver->multigrid->A[0], &(solver->eigenvalues[idx_eigen]));
	}
    } else {
	PASE_Aux_vector_orth_general(solver->aux_u[cur_level], num_converged, block_size, solver->multigrid->aux_B[cur_level]);
	for(idx_eigen=0; idx_eigen<block_size; idx_eigen++) {
	   PASE_Aux_vector_inner_product_general(solver->aux_u[cur_level][idx_eigen], solver->aux_u[cur_level][idx_eigen], solver->multigrid->aux_A[cur_level], &(solver->eigenvalues[idx_eigen]));
	}
    }

    return 0;
}

PASE_INT 
PASE_Mg_set_block_size(PASE_MG_SOLVER solver, PASE_INT block_size)
{
    solver->block_size        = block_size;
    solver->actual_block_size = block_size;
    return 0;
}

PASE_INT 
PASE_Mg_set_max_iteration(PASE_MG_SOLVER solver, PASE_INT max_iter)
{
    solver->max_iter = max_iter;
    return 0;
}

PASE_INT 
PASE_Mg_set_max_pre_iteration(PASE_MG_SOLVER solver, PASE_INT pre_iter)
{
    solver->pre_iter = pre_iter;
    return 0;
}

PASE_INT 
PASE_Mg_set_max_post_iteration(PASE_MG_SOLVER solver, PASE_INT post_iter)
{
    solver->post_iter = post_iter;
    return 0;
}

PASE_INT 
PASE_Mg_set_atol(PASE_MG_SOLVER solver, PASE_REAL atol)
{
    solver->atol = atol;
    return 0;
}

PASE_INT 
PASE_Mg_set_rtol(PASE_MG_SOLVER solver, PASE_REAL rtol)
{
    solver->rtol = rtol;
    return 0;
}

PASE_INT 
PASE_Mg_set_print_level(PASE_MG_SOLVER solver, PASE_INT print_level)
{
    solver->print_level = print_level;
    return 0;
}

PASE_INT
PASE_Mg_set_exact_eigenvalues(PASE_MG_SOLVER solver, PASE_SCALAR *exact_eigenvalues)
{
    solver->exact_eigenvalues = exact_eigenvalues;
    return 0;
}

PASE_INT
PASE_Mg_get_initial_vector_by_coarse_grid(void *mg_solver)
{
    PASE_MG_SOLVER solver = (PASE_MG_SOLVER)mg_solver;
    HYPRE_Solver lobpcg_solver 	= NULL; 
    PASE_INT  maxIterations 	= 100; 	/* maximum number of iterations */
    PASE_INT  pcgMode 		= 1;    	/* use rhs as initial guess for inner pcg iterations */
    PASE_INT  verbosity 	= 0;    	/* print iterations info */
    PASE_REAL atol 		= solver->atol;	/* absolute tolerance (all eigenvalues) */
    PASE_REAL rtol              = 1e-50;
    PASE_INT  lobpcgSeed 	= 77;

    PASE_INT myid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if(myid != 0) {
        verbosity = 0;
    }

    PASE_INT i;
    PASE_INT max_level          = solver->max_level;        
    PASE_INT block_size         = solver->block_size;       

    PASE_MATRIX A = solver->multigrid->A[max_level];
    PASE_MATRIX B = solver->multigrid->B[max_level];
    PASE_VECTOR u_temp = PASE_Vector_create_by_matrix(A, NULL); 
    PASE_VECTOR x = PASE_Vector_create_by_vector(u_temp);


    PASE_INT max_block_size = ((2*block_size)<(block_size+5))?(2*block_size):(block_size+5);
    PASE_SCALAR *eigenvalues 	= (PASE_SCALAR*)PASE_Malloc(max_block_size*sizeof(PASE_SCALAR));

    mv_MultiVectorPtr eigenvectors_Hh = NULL;
    mv_MultiVectorPtr constraints_Hh  = NULL;
    mv_InterfaceInterpreter* interpreter_Hh;
    HYPRE_MatvecFunctions matvec_fn_Hh;
    interpreter_Hh = hypre_CTAlloc(mv_InterfaceInterpreter, 1);
    PASE_Lobpcg_setup_interpreter(interpreter_Hh);
    PASE_Lobpcg_setup_matvec(&matvec_fn_Hh);
    eigenvectors_Hh = mv_MultiVectorCreateFromSampleVector(interpreter_Hh, max_block_size, u_temp);
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
    PASE_REAL *rate = (PASE_REAL*)PASE_Malloc((max_block_size-block_size)*sizeof(PASE_REAL));
    PASE_INT index = 0;
    PASE_REAL max_rate = 1.0;
    i = 0;
    while(i<(max_block_size-block_size) && index > -1) {
	rate[i] = fabs(eigenvalues[i+block_size-1]/eigenvalues[i+block_size]);
	//printf("rata[%d] = %e\n", i, rate[i]);
	if(rate[i] < 0.9) {
	    solver->block_size = block_size+i;
	    index = -1;
	} else if(max_rate>rate[i]) {
	    max_rate = rate[i];
	    index = i;
	}
	i++;
    }

    if(index > -1) {
	solver->block_size = block_size + index;
    }
    //solver->block_size = 21;
    if(myid == 0){
        printf("modified block_size = %d\n\n", solver->block_size);
    }

    block_size = solver->block_size;

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

    solver->u = (PASE_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_VECTOR));
    solver->eigenvalues = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
    for(i=0; i<block_size; i++) {
	solver->eigenvalues[i] = eigenvalues[i];
	solver->u[i]           = u_H[i];
    }
    for(i=block_size; i<max_block_size; i++) {
	PASE_Vector_destroy(u_H[i]);
    }

    if( solver->print_level > 1)
    {
	printf("Get initial vector");
	for( i=0; i<block_size; i++)
	{
	    printf(", eigen[%d] = %.16f", i, solver->eigenvalues[i]);
	}
	printf(".\n");
    }
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

    PASE_Free(rate);
    PASE_Free((mv_TempMultiVector*) mv_MultiVectorGetData(eigenvectors_Hh));
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
    PASE_MG_SOLVER solver = (PASE_MG_SOLVER)mg_solver;
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
PASE_Mg_presmoothing_by_pcg_aux_hypre(void *mg_solver)
{
    PASE_MG_SOLVER solver = (PASE_MG_SOLVER)mg_solver;
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
PASE_Mg_solve_directly_by_lobpcg_aux_hypre(void *mg_solver)
{
    PASE_MG_SOLVER solver = (PASE_MG_SOLVER)mg_solver;
    HYPRE_Solver lobpcg_solver 	= NULL; 
    PASE_INT     maxIterations 	= 10; 	/* maximum number of iterations */
    PASE_INT     pcgMode 	= 1;    	/* use rhs as initial guess for inner pcg iterations */
    PASE_INT     verbosity 	= 0;    	/* print iterations info */
    PASE_REAL    atol 		= solver->atol;	/* absolute tolerance (all eigenvalues) */
    PASE_REAL    rtol           = 1e-50;

    PASE_INT myid = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    if(myid != 0) {
        verbosity = 0;
    }

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
    HYPRE_LOBPCGSetTol(lobpcg_solver, atol);
    HYPRE_LOBPCGSetRTol(lobpcg_solver, rtol);
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

PASE_INT
PASE_Mg_presmoothing_by_cg(void *mg_solver)
{
    PASE_MG_SOLVER solver = (PASE_MG_SOLVER)mg_solver;
    PASE_INT     cur_level  = solver->cur_level;
    PASE_INT     block_size = solver->block_size;
    PASE_INT     max_iter   = solver->pre_iter;
    PASE_SCALAR *eigenvalues = solver->eigenvalues;

    PASE_INT idx_eigen;
    PASE_REAL tol = 1e-10;
    PASE_REAL inner_A, inner_B;
    PASE_VECTOR b = PASE_Vector_create_by_vector(solver->u[0]);
    //clock_t start, end;
	//printf("\n");
	//printf("cur_level = %d\n", cur_level);
    for(idx_eigen=solver->num_converged; idx_eigen<block_size; idx_eigen++) {
	//start = clock();
        PASE_Matrix_multiply_vector_general(eigenvalues[idx_eigen], solver->multigrid->B[0], solver->u[idx_eigen], 0.0, b);
        PASE_Linear_solve_by_cg(solver->multigrid->A[0], b, solver->u[idx_eigen], tol, max_iter);

        PASE_Vector_inner_product_general(solver->u[idx_eigen], solver->u[idx_eigen], solver->multigrid->A[0], &inner_A);
        PASE_Vector_inner_product_general(solver->u[idx_eigen], solver->u[idx_eigen], solver->multigrid->B[0], &inner_B);
        eigenvalues[idx_eigen] = inner_A / inner_B;
        //printf("after Cg, eigenvalues[%d] = %.12e\n", j, eigenvalues[j]);
	//end = clock();
	//printf("the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
    }
    PASE_Vector_destroy(b);

#if 1
    if( solver->print_level > 1) {
        printf("Cur_level %d", cur_level);
        for(idx_eigen=0; idx_eigen<block_size; idx_eigen++)
        {
            printf(", eigen[%d] = %.16f", idx_eigen, eigenvalues[idx_eigen]);
        }
        printf(".\n");
    }
#endif
    
    return 0;
}

PASE_INT
PASE_Mg_presmoothing_by_cg_aux(void *mg_solver)
{
    PASE_MG_SOLVER solver = (PASE_MG_SOLVER)mg_solver;
    PASE_INT     cur_level  = solver->cur_level;
    PASE_INT     block_size = solver->block_size;
    PASE_INT     max_iter   = solver->pre_iter + cur_level * 0.5;
    PASE_SCALAR *eigenvalues = solver->eigenvalues;

    PASE_INT idx_eigen;
    PASE_REAL tol = 1e-10;
    PASE_REAL inner_A, inner_B;
    PASE_AUX_VECTOR *aux_u = solver->aux_u[cur_level];
    PASE_AUX_VECTOR  aux_b = PASE_Aux_vector_create_by_aux_vector(aux_u[0]);
    //clock_t start, end;
    //printf("\n");
    //printf("cur_level = %d\n", cur_level);
    for(idx_eigen=solver->num_converged; idx_eigen<block_size; idx_eigen++) {
	//start = clock();
        PASE_Aux_matrix_multiply_aux_vector_general(eigenvalues[idx_eigen], solver->multigrid->aux_B[cur_level], aux_u[idx_eigen], 0.0, aux_b);
        PASE_Linear_solve_by_cg_aux(solver->multigrid->aux_A[cur_level], aux_b, aux_u[idx_eigen], tol, max_iter);

        PASE_Aux_vector_inner_product_general(aux_u[idx_eigen], aux_u[idx_eigen], solver->multigrid->aux_A[cur_level], &inner_A);
        PASE_Aux_vector_inner_product_general(aux_u[idx_eigen], aux_u[idx_eigen], solver->multigrid->aux_B[cur_level], &inner_B);
        eigenvalues[idx_eigen] = inner_A / inner_B;
        //printf("after Cg, eigenvalues[%d] = %.12e\n", j, eigenvalues[j]);
	//end = clock();
	//printf("the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
    }
    PASE_Aux_vector_destroy(aux_b);

#if 1
    if( solver->print_level > 1) {
        printf("Cur_level %d", cur_level);
        for( idx_eigen=0; idx_eigen<block_size; idx_eigen++) {
            printf(", eigen[%d] = %.16f", idx_eigen, eigenvalues[idx_eigen]);
        }
        printf(".\n");
    }
#endif
    
    return 0;
}

PASE_INT
PASE_Linear_solve_by_cg(PASE_MATRIX A, PASE_VECTOR b, PASE_VECTOR x, PASE_REAL tol, PASE_INT max_iter)
{
    PASE_INT iter = 0;
    PASE_REAL bnorm = 1.0, rnorm = 1.0, rho = 1.0, rho_1 = 1.0, alpha = 1.0, beta = 1.0, tmp = 1.0;
    PASE_VECTOR r = PASE_Vector_create_by_vector(x);
    PASE_VECTOR p = PASE_Vector_create_by_vector(x);
    PASE_VECTOR q = PASE_Vector_create_by_vector(x);

    //start = clock();
    PASE_Vector_inner_product(b, b, &bnorm);
    bnorm = sqrt(bnorm);
    PASE_Vector_copy(b, r);
    PASE_Matrix_multiply_vector_general(-1.0, A, x, 1.0, r);
    PASE_Vector_inner_product(r, r, &rho);
    rnorm = sqrt(rho);
    //printf("before Cg, rnorm = %.12e, bnorm = %.12e, rnorm/bnorm = %.12e\n", rnorm, bnorm, rnorm/bnorm);
    if(rnorm/bnorm < tol) {
        return 0;
    }
    for(iter=0; iter<max_iter; iter++) {
        if(iter>0) {
            beta = rho / rho_1; 
            PASE_Vector_scale(beta, p);
            PASE_Vector_add_vector(1.0, r, p);
        } else {
            PASE_Vector_copy(r, p);
        }
        PASE_Matrix_multiply_vector(A, p, q);
        PASE_Vector_inner_product(p, q, &tmp);
        alpha = rho / tmp;
        PASE_Vector_add_vector(alpha, p, x);
        PASE_Vector_add_vector(-1.0*alpha, q, r);
    
        rho_1 = rho;
        PASE_Vector_inner_product(r, r, &rho);
        rnorm = sqrt(rho);
    
        //printf("after Cg, rnorm = %.12e, bnorm = %.12e, rnorm/bnorm = %.12e\n", rnorm, bnorm, rnorm/bnorm);
        if(rnorm/bnorm < tol) {
            break;
        }
    }
    //end = clock();
    //printf("the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
    PASE_Vector_destroy(r);
    PASE_Vector_destroy(p);
    PASE_Vector_destroy(q);
    return 0;
}

PASE_INT
PASE_Linear_solve_by_cg_aux(PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_b, PASE_AUX_VECTOR aux_x, PASE_REAL tol, PASE_INT max_iter)
{
    PASE_INT iter = 0;
    PASE_REAL bnorm = 1.0, rnorm = 1.0, rho = 1.0, rho_1 = 1.0, alpha = 1.0, beta = 1.0, tmp = 1.0;
    PASE_AUX_VECTOR aux_r = PASE_Aux_vector_create_by_aux_vector(aux_x);
    PASE_AUX_VECTOR aux_p = PASE_Aux_vector_create_by_aux_vector(aux_x);
    PASE_AUX_VECTOR aux_q = PASE_Aux_vector_create_by_aux_vector(aux_x);

    //start = clock();
    PASE_Aux_vector_inner_product(aux_b, aux_b, &bnorm);
    bnorm = sqrt(bnorm);
    PASE_Aux_vector_copy(aux_b, aux_r);
    PASE_Aux_matrix_multiply_aux_vector_general(-1.0, aux_A, aux_x, 1.0, aux_r);
    PASE_Aux_vector_inner_product(aux_r, aux_r, &rho);
    rnorm = sqrt(rho);
    //printf("before Cg, rnorm = %.12e, bnorm = %.12e, rnorm/bnorm = %.12e\n", rnorm, bnorm, rnorm/bnorm);
    if(rnorm/bnorm < tol) {
        return 0;
    }
    for(iter=0; iter<max_iter; iter++) {
        if(iter>0) {
            beta = rho / rho_1; 
            PASE_Aux_vector_scale(beta, aux_p);
            PASE_Aux_vector_add(1.0, aux_r, aux_p);
        } else {
            PASE_Aux_vector_copy(aux_r, aux_p);
        }
        PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_p, aux_q);
        PASE_Aux_vector_inner_product(aux_p, aux_q, &tmp);
        alpha = rho / tmp;
        PASE_Aux_vector_add(alpha, aux_p, aux_x);
        PASE_Aux_vector_add(-1.0*alpha, aux_q, aux_r);
    
        rho_1 = rho;
        PASE_Aux_vector_inner_product(aux_r, aux_r, &rho);
        rnorm = sqrt(rho);
    
        //printf("after Cg, rnorm = %.12e, bnorm = %.12e, rnorm/bnorm = %.12e\n", rnorm, bnorm, rnorm/bnorm);
        if(rnorm/bnorm < tol) {
            break;
        }
    }
    //end = clock();
    //printf("the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
    PASE_Aux_vector_destroy(aux_r);
    PASE_Aux_vector_destroy(aux_p);
    PASE_Aux_vector_destroy(aux_q);
    return 0;
}

PASE_INT
PASE_Mg_solve_directly_by_IRA(void *mg_solver)
{
    PASE_MG_SOLVER   solver      = (PASE_MG_SOLVER)mg_solver;
    PASE_INT         cur_level   = solver->cur_level;
    PASE_INT         block_size  = solver->block_size;

    PASE_AUX_MATRIX  aux_A       = solver->multigrid->aux_A[cur_level];            
    PASE_AUX_MATRIX  aux_B       = solver->multigrid->aux_B[cur_level];            
    PASE_AUX_VECTOR *aux_u       = solver->aux_u[cur_level];
    PASE_SCALAR     *eigenvalues = solver->eigenvalues;

    PASE_INT         ncv         = ((2*block_size) > 10)? (2*block_size) : 10;
    PASE_INT         max_iter    = 1000;
    PASE_INT         i           = 0;
    PASE_REAL        tol         = solver->atol * 1e-40;

    PASE_AUX_VECTOR *krylovspace = (PASE_AUX_VECTOR*)PASE_Malloc((ncv+1)*sizeof(PASE_AUX_VECTOR));
    for(i=0; i<ncv+1; i++) {
        krylovspace[i] = PASE_Aux_vector_create_by_aux_vector(aux_u[0]);
    }
    printf("solve directly by arpack\n");
    IRA_RestartArnoldi(eigenvalues, aux_u, aux_A, aux_B, krylovspace, ncv, block_size, max_iter, tol); 
#if 1
    PASE_REAL r_norm = 0;
    for(i=0; i<block_size; i++) {
	PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_u[i], krylovspace[0]);
	PASE_Aux_matrix_multiply_aux_vector_general(-eigenvalues[i], aux_B, aux_u[i], 1.0, krylovspace[0]); 
	PASE_Aux_vector_inner_product(krylovspace[0], krylovspace[0], &r_norm);
	r_norm	= sqrt(r_norm);
        printf("eigenvalues[%d] = %.8e, residual[%d] = %.6e\n", i, eigenvalues[i], i, r_norm);
    }
    printf("\n");
#endif

    for(i=0; i<ncv+1; i++) {
        PASE_Aux_vector_destroy(krylovspace[i]);
    }
    PASE_Free(krylovspace);
    return 0;
}

