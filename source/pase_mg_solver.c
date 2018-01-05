#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "pase_mg_solver_hypre.h"
#include "pase_mg_solver.h"
#include "pase_lobpcg_hypre.h"
#include "pase_gcg.h"

#define CLK_TCK 1000000

PASE_MG_SOLVER
PASE_Mg_solver_create(PASE_MATRIX A, PASE_MATRIX B, PASE_PARAMETER param, PASE_MULTIGRID_OPERATOR ops)
{
  PASE_MULTIGRID multigrid = PASE_Multigrid_create(A, B, param, ops);
  PASE_MG_SOLVER solver = PASE_Mg_solver_create_by_multigrid(multigrid);
  return solver;
}

/**
 * @brief PASE_MG_FUNCTION 的创建
 *
 * @param solver  输入/输出参数
 *
 * @return PASE_MG_FUNCTION
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
 *
 * @param solver  输入/输出参数
 *
 * @return PASE_MG_SOLVER
 */
PASE_MG_SOLVER
PASE_Mg_solver_create_by_multigrid(PASE_MULTIGRID multigrid)
{
  PASE_MG_SOLVER solver      = (PASE_MG_SOLVER)PASE_Malloc(sizeof(PASE_MG_SOLVER_PRIVATE));

  solver->multigrid          = multigrid;
  solver->function           = PASE_Mg_function_create(//PASE_Mg_get_initial_vector_by_coarse_grid_hypre,
                                                       //PASE_Mg_get_initial_vector_by_coarse_grid_lobpcg_amg_hypre, 
						       PASE_Mg_get_initial_vector_by_full_multigrid_hypre, 
                                                       //PASE_Mg_direct_solve_by_lobpcg_aux_hypre,
						       PASE_Mg_direct_solve_by_gcg, 
                                                       //PASE_Mg_presmoothing_by_cg,
                                                       //PASE_Mg_presmoothing_by_cg,
						       //PASE_Mg_presmoothing_by_pcg_amg_hypre, 
						       //PASE_Mg_presmoothing_by_pcg_amg_hypre, 
						       PASE_Mg_presmoothing_by_amg_hypre, 
						       PASE_Mg_presmoothing_by_amg_hypre, 
                                                       PASE_Mg_presmoothing_by_cg_aux,
                                                       PASE_Mg_presmoothing_by_cg_aux);
  solver->solver_type         = 0;
  solver->block_size          = 1;
  solver->max_block_size      = 1;
  solver->actual_block_size   = 1;
  solver->max_pre_iter        = 1;
  solver->max_post_iter       = 1;
  solver->max_cycle           = 200;
  solver->max_level           = multigrid->actual_level-1;
  solver->cur_level           = 0;
  solver->rtol                = 1e-8;
  solver->atol                = 1e-8;
  solver->r_norm              = NULL;
  solver->nconv               = 0;
  solver->nlock               = 0;
  solver->ncycl               = 0;
  solver->print_level         = 1;

  solver->set_up_time         = 0.0;
  solver->smooth_time         = 0.0;
  solver->set_aux_time        = 0.0;
  solver->prolong_time        = 0.0;
  solver->direct_solve_time   = 0.0;
  solver->total_solve_time    = 0.0;
  solver->total_time          = 0.0;
  solver->eigenvalues         = NULL;
  solver->exact_eigenvalues   = NULL;
  solver->u                   = NULL;
  solver->is_u_owner          = PASE_YES;
  solver->aux_u               = NULL;
  return solver;
}

/**
 * @brief 销毁 PASE_MG_SOLVER 并释放内存空间.
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_solver_destroy(PASE_MG_SOLVER solver)
{
  PASE_INT i, j;
  if(NULL != solver) {
    if(NULL != solver->aux_u) {
      for(i = 0; i <= solver->max_level; i++) {
	if(NULL != solver->aux_u[i]){
	  for(j = 0; j < solver->block_size; j++) {
	    if(solver->aux_u[i][j]) {
	      PASE_Aux_vector_destroy(solver->aux_u[i][j]);
	    }
	  }
	  PASE_Free(solver->aux_u[i]);
	}
      }
      PASE_Free(solver->aux_u);
    }
    if(NULL != solver->eigenvalues) {
      PASE_Free(solver->eigenvalues);
    }
    if(NULL != solver->r_norm) {
      PASE_Free(solver->r_norm);
    }
    if(NULL != solver->u) {
      if(PASE_YES == solver->is_u_owner) {
        for(j = 0; j < solver->block_size; j++) {
          if(NULL != solver->u[j]) {
            PASE_Vector_destroy(solver->u[j]);
          }
        }
      } else {
        for(j = solver->actual_block_size; j < solver->block_size; j++) {
          if(NULL != solver->u[j]) {
            PASE_Vector_destroy(solver->u[j]);
          }
        }
      }
      PASE_Free(solver->u);
    }
    if(NULL != solver->function) {
      PASE_Free(solver->function);
    }
    if(NULL != solver->multigrid) {
      PASE_Multigrid_destroy(solver->multigrid);
    }
    PASE_Free(solver);
  }
  return 0;
}

/**
 * @brief PASE_MG_SOLVER 的初始化
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_up(PASE_MG_SOLVER solver, PASE_VECTOR x)
{
  clock_t start, end;
  start = clock();
  PASE_INT i = 0;
  if(NULL == solver->eigenvalues) {
    solver->eigenvalues = (PASE_SCALAR*)PASE_Malloc(solver->max_block_size*sizeof(PASE_SCALAR));
  }
  if(NULL == solver->u) {
    solver->u = (PASE_VECTOR*)PASE_Malloc(solver->max_block_size*sizeof(PASE_VECTOR));
    if(NULL != x) {
      for(i = 0; i < solver->max_block_size; i++) {
        solver->u[i] = PASE_Vector_create_by_vector(x);
      }
    } else {
      //缺省情况, 依据 A0 创建 u[0], 再依据 u[0] 创建 u[i] (i>0).
      //对于基于 hypre 的向量结构, 特征值求解个数多的时候, 所有向量公用一个partitioning, 节省内存消耗
      solver->u[0] = PASE_Vector_create_by_matrix_and_vector_data_operator(solver->multigrid->A[0], NULL);
      for(i = 1; i < solver->max_block_size; i++) {
        solver->u[i] = PASE_Vector_create_by_vector(solver->u[0]);
      }
    }
  }
  solver->aux_u       = (PASE_AUX_VECTOR**)PASE_Malloc((solver->max_level+1)*sizeof(PASE_AUX_VECTOR*));
  for(i = 0; i <= solver->max_level; i++) {
    solver->aux_u[i] = NULL;
  }

  solver->function->get_initial_vector(solver);
  end = clock();
  solver->set_up_time += ((double)(end-start))/CLK_TCK;
  return 0;
}


/**
 * @brief MG求解, 主要通过迭代 PASE_Mg_cycle 函数实现. 
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_solve(PASE_MG_SOLVER solver)
{
  PASE_Mg_error_estimate(solver);
  PASE_Mg_print(solver);
  clock_t start, end;
  start = clock();
  do {
    solver->ncycl++;
    PASE_Mg_cycle(solver);
    PASE_Mg_error_estimate(solver);
  } while( solver->max_cycle > solver->ncycl && solver->nconv < solver->actual_block_size);
  end = clock();
  solver->total_solve_time += ((double)(end-start))/CLK_TCK;
  solver->total_time        = solver->total_solve_time + solver->set_up_time;
  PASE_Mg_print(solver);

  return 0;
}

PASE_INT
PASE_Mg_cycle(PASE_MG_SOLVER solver)
{
  if(0 == solver->solver_type) {
    PASE_Mg_cycle_two_gird(solver);
  } else if(1 == solver->solver_type) {
    PASE_Mg_cycle_multigrid(solver);
  } 
  return 0;
}

/**
 * @brief MG 方法的主体，采用递归定义，每层上主要包含分成三步：前光滑，粗空间校正，后光滑. 
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_cycle_multigrid(PASE_MG_SOLVER solver)
{
  PASE_INT cur_level = solver->cur_level;
  PASE_INT max_level = solver->max_level;

  if( cur_level < max_level)
  {
    //PASE_Printf(MPI_COMM_WORLD, "cur_level = %d, max_level = %d\n", solver->cur_level, solver->max_level);
    /*前光滑*/
    //PASE_Printf(MPI_COMM_WORLD, "PreSmoothing..........\n");
    PASE_Mg_presmoothing(solver);
    //PASE_Mg_orthogonalize(solver);
    //PASE_Printf(MPI_COMM_WORLD, "Creating AuxMatrix..........\n");
    PASE_Mg_set_aux_space_multigrid(solver);

    /*粗空间校正*/
    //PASE_Printf(MPI_COMM_WORLD, "Correction on low-dim space\n");
    solver->cur_level++;
    PASE_Mg_cycle(solver);
    solver->cur_level--;

    /*后光滑*/
    //PASE_Printf(MPI_COMM_WORLD, "PostCorrecting..........\n");
    PASE_Mg_prolong_multigrid(solver);
    //PASE_Printf(MPI_COMM_WORLD, "PostSmoothing..........\n");
    PASE_Mg_postsmoothing(solver);
    //PASE_Mg_orthogonalize(solver);
  }
  else if( cur_level == max_level)
  {
    PASE_Mg_direct_solve(solver);
  }
  return 0;
}

/**
 * @brief 只在最细最粗两层网格间迭代的 MG 方法的单步迭代步
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_cycle_two_gird(PASE_MG_SOLVER solver)
{
  //限制到粗空间
  PASE_Mg_set_aux_space_two_grid(solver);

  //在粗空间解特征值问题
  solver->cur_level = solver->max_level;
  PASE_Mg_direct_solve(solver);
  solver->cur_level = 0;

  //投影回细空间
  PASE_Mg_prolong_two_grid(solver);
  //在细空间解解问题
  PASE_Mg_postsmoothing(solver);
  return 0;
}

/**
 * @brief 设置最粗层的辅助矩阵, 以及初始辅助向量
 *
 * @param solver
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_aux_space_two_grid(PASE_MG_SOLVER solver)
{
  PASE_INT    idx_eigen  = 0;
  PASE_INT    block_size = solver->block_size;
  PASE_INT    max_level  = solver->max_level;
  clock_t     start, end;

  start = clock();
  PASE_Mg_coarsest_aux_matrix_set(solver);

  if(NULL == solver->aux_u[max_level]) {
    solver->aux_u[max_level] = (PASE_AUX_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_AUX_VECTOR));
    solver->aux_u[max_level][0] = PASE_Aux_vector_create_by_aux_matrix(solver->multigrid->aux_A[max_level]);
    for(idx_eigen=1; idx_eigen<block_size; idx_eigen++) {
      solver->aux_u[max_level][idx_eigen] = PASE_Aux_vector_create_by_aux_vector(solver->aux_u[max_level][0]);
    }
  }

  /*多次迭代需要多次初始化初值，但空间不需要重新申请*/
  for(idx_eigen = 0; idx_eigen < block_size; idx_eigen++) {
    PASE_Vector_set_constant_value(solver->aux_u[max_level][idx_eigen]->vec, 0.0);
    memset(solver->aux_u[max_level][idx_eigen]->block, 0, block_size*sizeof(PASE_SCALAR));
    solver->aux_u[max_level][idx_eigen]->block[idx_eigen] = 1.0;
  }
  end = clock();
  solver->set_aux_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

/**
 * @brief 将最粗层的辅助空间的辅助向量投影到最细层空间中
 *
 * @param solver
 *
 * @return 
 */
PASE_INT
PASE_Mg_prolong_two_grid(PASE_MG_SOLVER solver)
{
  clock_t  start, end;
  PASE_INT idx_block     = 0;
  PASE_INT j             = 0;
  PASE_INT block_size    = solver->block_size;
  PASE_INT max_level     = solver->max_level;
  PASE_INT nconv         = solver->nconv;
  start = clock();
  PASE_VECTOR *u_h        = (PASE_VECTOR*)PASE_Malloc((block_size-nconv)*sizeof(PASE_VECTOR));
  for(idx_block = nconv; idx_block < block_size; idx_block++) {
    u_h[idx_block-nconv] = PASE_Vector_create_by_vector(solver->u[0]);
    PASE_Mg_prolong(solver, max_level, solver->aux_u[max_level][idx_block]->vec, 0, u_h[idx_block-nconv]);
    for(j = 0; j < block_size; j++) {
      PASE_Vector_axpy(solver->aux_u[max_level][idx_block]->block[j], solver->u[j],  u_h[idx_block-nconv]);
    }
  }
  for(idx_block = nconv; idx_block < block_size; idx_block++) {
    PASE_Vector_copy(u_h[idx_block-nconv], solver->u[idx_block]);
    PASE_Vector_destroy(u_h[idx_block-nconv]);
  }
  PASE_Free(u_h);
  end = clock();
  solver->prolong_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

/**
 * @brief 将第 i 层网格上向量 u_i 投影到第 j 层的网格上, 并存储在向量 u_j 中
 *
 * @param solver
 * @param i
 * @param u_i
 * @param j
 * @param u_j
 *
 * @return 
 */
PASE_INT 
PASE_Mg_prolong(PASE_MG_SOLVER solver,  PASE_INT i, PASE_VECTOR u_i, PASE_INT j, PASE_VECTOR u_j)
{
  PASE_INT    idx_level = 0;
  PASE_VECTOR u_h       = NULL; 
  PASE_VECTOR u_H       = u_i;
  for(idx_level = i-1; idx_level >= j+1; idx_level--) {
    u_h = PASE_Vector_create_by_matrix_and_vector_data_operator(solver->multigrid->A[idx_level], u_i->ops);
    PASE_Matrix_multiply_vector(solver->multigrid->P[idx_level], u_H, u_h); 
    if(idx_level < i-1) {
      PASE_Vector_destroy(u_H);
    }
    u_H = u_h;
    u_h = NULL;
  }
  PASE_Matrix_multiply_vector(solver->multigrid->P[j], u_H, u_j); 
  if(i > j+1) {
    PASE_Vector_destroy(u_H);
  }
  return 0;
}

/**
 * @brief 完成一次 PASE_Mg_cycle 后, 需计算残差及已收敛特征对个数. 已收敛特征对在之后的迭代中，不再计算和更改. 
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_error_estimate(PASE_MG_SOLVER solver)
{
  PASE_INT         block_size	 = solver->block_size; 
  PASE_INT         nconv         = solver->nconv;
  PASE_REAL        atol          = solver->atol;
  PASE_VECTOR     *u0	         = solver->u;
  PASE_SCALAR     *eigenvalues   = solver->eigenvalues;
  PASE_MATRIX      A0	         = solver->multigrid->A[0];
  PASE_MATRIX      B0	         = solver->multigrid->B[0];

  /* 计算最细层的残差：r = Au - kMu */
  PASE_INT         flag          = 0;
  PASE_REAL       *check_multi   = (PASE_REAL*)PASE_Malloc((block_size-1)*sizeof(PASE_REAL));
  PASE_INT         i		 = 0;
  PASE_REAL        r_norm        = 1e+5;
  PASE_VECTOR      r             = PASE_Vector_create_by_vector(u0[0]);
  solver->nlock                  = nconv;

  if(NULL == solver->r_norm) {
    solver->r_norm = (PASE_REAL*)PASE_Malloc(block_size*sizeof(PASE_REAL));
  }

  for(i = nconv; i < block_size; i++) {
    PASE_Matrix_multiply_vector(A0, u0[i], r);
    PASE_Matrix_multiply_vector_general(-eigenvalues[i], B0, u0[i], 1.0, r); 
    PASE_Vector_inner_product(r, r, &r_norm);
    r_norm	      = sqrt(r_norm);
    solver->r_norm[i] = r_norm;
    if(i+1 < block_size) {
      check_multi[i] = fabs((eigenvalues[i]-eigenvalues[i+1])/eigenvalues[i]);
    }
    if( r_norm < atol && flag == 0) {
      solver->nconv++;
    } else {
      flag = 1;
    }
  }
  //检查第一个为收敛的特征值与最后一个刚收敛的特征值是否有可能是重特征值，为保证之后的排序问题，需让重特征值同时在收敛的集合或未收敛的集合.
  while(solver->nconv > nconv && solver->nconv < block_size && check_multi[solver->nconv-1] < 1e-8) {
    solver->nconv--;
  }
  PASE_Free(check_multi);
  PASE_Vector_destroy(r);

  if(solver->print_level > 0) {
    //PASE_REAL error = fabs(solver->eigenvalues[0] - solver->exact_eigenvalues[0]);	
    PASE_Printf(MPI_COMM_WORLD, "cycle = %d, nconv = %d, ", solver->ncycl, solver->nconv);
    if(solver->nconv < solver->block_size) {
      PASE_Printf(MPI_COMM_WORLD, "residual of the first unconverged = %1.6e\n", solver->r_norm[solver->nconv]);
    } else {
      PASE_Printf(MPI_COMM_WORLD, "all the wanted eigenpairs have converged.\n");
    }
  }	

  return 0;
}

/**
 * @brief 打印计算所得特征值，及其对应的残差.
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_print(PASE_MG_SOLVER solver)
{
  PASE_INT idx_eigen = 0;
  if(solver->print_level > 0) {
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n");
    for(idx_eigen=0; idx_eigen<solver->block_size; idx_eigen++) {
      PASE_Printf(MPI_COMM_WORLD, "%d-th eig=%.8e, aresi = %.8e\n", idx_eigen, solver->eigenvalues[idx_eigen], solver->r_norm[idx_eigen]);
    }
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n");
    PASE_Printf(MPI_COMM_WORLD, "set up time       = %f seconds\n", solver->set_up_time);
    PASE_Printf(MPI_COMM_WORLD, "smooth time       = %f seconds\n", solver->smooth_time);
    PASE_Printf(MPI_COMM_WORLD, "set aux time      = %f seconds\n", solver->set_aux_time);
    PASE_Printf(MPI_COMM_WORLD, "prolong time      = %f seconds\n", solver->prolong_time);
    PASE_Printf(MPI_COMM_WORLD, "direct solve time = %f seconds\n", solver->direct_solve_time);
    PASE_Printf(MPI_COMM_WORLD, "total solve time  = %f seconds\n", solver->total_solve_time);
    PASE_Printf(MPI_COMM_WORLD, "total time        = %f seconds\n", solver->total_time);
  }	
  return 0;
}

/**
 * @brief 前光滑函数
 *
 * @param solver  输入/输出参数
 *
 * @return 
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
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_postsmoothing(PASE_MG_SOLVER solver)
{
  clock_t start, end;
  start = clock();
  if(solver->cur_level == 0) {
    solver->function->postsmoothing(solver);
    //PASE_Mg_orthogonalize(solver);
  } else {
    solver->function->postsmoothing_aux(solver);
  }
  end = clock();
  solver->smooth_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

/**
 * @brief 最粗层直接求解辅助空间的特征值问题
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_direct_solve(PASE_MG_SOLVER solver)
{
  clock_t start, end;
  start = clock();
  solver->function->direct_solve(solver);
  end   = clock();
  solver->direct_solve_time += ((double)(end-start))/CLK_TCK;
  return 0;
}

/**
 * @brief 设置更粗层辅助空间的矩阵, 及前光滑所需的初始向量
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_aux_space_multigrid(PASE_MG_SOLVER solver)
{
  PASE_INT         cur_level  = solver->cur_level;
  PASE_INT         block_size = solver->block_size;

  clock_t start, end;
  start = clock();
  PASE_Mg_set_aux_matrix_multigrid(solver);
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
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_aux_matrix_multigrid(PASE_MG_SOLVER solver)
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

#if 1
  if(cur_level == 0) {
    if(NULL == solver->multigrid->aux_A[cur_level+1]) {
      solver->multigrid->aux_A[cur_level+1] = PASE_Aux_matrix_create(A_H, R_hH, A_h, solver->u, block_size);  
      solver->multigrid->aux_B[cur_level+1] = PASE_Aux_matrix_create(B_H, R_hH, B_h, solver->u, block_size);  
    } else {
      PASE_Aux_matrix_set_aux_space_some(solver->multigrid->aux_A[cur_level+1], solver->nlock, block_size-1, R_hH, A_h, solver->u);
      PASE_Aux_matrix_set_aux_space_some(solver->multigrid->aux_B[cur_level+1], solver->nlock, block_size-1, R_hH, B_h, solver->u);
    }
  } else {
    if(NULL == solver->multigrid->aux_A[cur_level+1]) {
      solver->multigrid->aux_A[cur_level+1] = PASE_Aux_matrix_create_by_aux_matrix(A_H, R_hH, aux_A_h, aux_u_h, block_size);  
      solver->multigrid->aux_B[cur_level+1] = PASE_Aux_matrix_create_by_aux_matrix(B_H, R_hH, aux_B_h, aux_u_h, block_size);  
    } else {
      PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(solver->multigrid->aux_A[cur_level+1], solver->nlock, block_size-1, R_hH, aux_A_h, aux_u_h);
      PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(solver->multigrid->aux_B[cur_level+1], solver->nlock, block_size-1, R_hH, aux_B_h, aux_u_h);
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
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_prolong_multigrid(PASE_MG_SOLVER solver)
{
  clock_t start, end;
  start = clock();
  PASE_INT i;
  PASE_INT cur_level  = solver->cur_level;
  PASE_INT block_size = solver->block_size;
  PASE_MATRIX P_Hh    = solver->multigrid->P[cur_level];
#if 1
  PASE_INT nconv = solver->nconv;

  /* u_new += u1*u_0->aux_h */
  /* u_new->b_H += P*u0->b_H */
  if(cur_level == 0) {
    PASE_VECTOR *u_new = (PASE_VECTOR*)PASE_Malloc((block_size-nconv)*sizeof(PASE_VECTOR));
    for(i = nconv; i<block_size; i++) {
      u_new[i-nconv] = PASE_Vector_create_by_vector(solver->u[0]);
      PASE_Multi_vector_combination(solver->u, block_size, solver->aux_u[1][i]->block, u_new[i-nconv]);
      PASE_Matrix_multiply_vector_general(1.0, P_Hh , solver->aux_u[1][i]->vec, 1.0, u_new[i-nconv]);
    }
    for(i = nconv; i < block_size; i++) {
      PASE_Vector_copy(u_new[i-nconv], solver->u[i]);
      PASE_Vector_destroy(u_new[i-nconv]);
    }
    PASE_Free(u_new);
  } else {
    PASE_AUX_VECTOR *u_new = (PASE_AUX_VECTOR*)PASE_Malloc((block_size-nconv)*sizeof(PASE_AUX_VECTOR));
    for(i = nconv; i<block_size; i++) {
      u_new[i-nconv] = PASE_Aux_vector_create_by_aux_vector(solver->aux_u[cur_level][0]);
      PASE_Multi_aux_vector_combination(solver->aux_u[cur_level], block_size, solver->aux_u[cur_level+1][i]->block, u_new[i-nconv]);
      PASE_Matrix_multiply_vector_general(1.0, P_Hh , solver->aux_u[cur_level+1][i]->vec, 1.0, u_new[i-nconv]->vec);
    }
    for(i = nconv; i < block_size; i++) {
      PASE_Aux_vector_copy(u_new[i-nconv], solver->aux_u[cur_level][i]);
      PASE_Aux_vector_destroy(u_new[i-nconv]);
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
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_orthogonalize(PASE_MG_SOLVER solver)
{
  PASE_INT cur_level 	 = solver->cur_level;
  PASE_INT block_size	 = solver->block_size;
  PASE_INT nconv         = solver->nconv;
  PASE_INT idx_eigen     = 0;

  if(cur_level == 0) {
    for(idx_eigen = nconv; idx_eigen < block_size; idx_eigen++) {
      PASE_Vector_orthogonalize_general(solver->u, idx_eigen, 0, idx_eigen-1, solver->multigrid->B[0]);
      PASE_Vector_inner_product_general(solver->u[idx_eigen], solver->u[idx_eigen], solver->multigrid->A[0], &(solver->eigenvalues[idx_eigen]));
    }
  } else {
    for(idx_eigen = 0; idx_eigen < block_size; idx_eigen++) {
      PASE_Aux_vector_orthogonalize_general(solver->aux_u[cur_level], idx_eigen, 0, idx_eigen-1, solver->multigrid->aux_B[cur_level]);
      PASE_Aux_vector_inner_product_general(solver->aux_u[cur_level][idx_eigen], solver->aux_u[cur_level][idx_eigen], solver->multigrid->aux_A[cur_level], &(solver->eigenvalues[idx_eigen]));
    }
  }

  return 0;
}

/**
 * @brief 设置特征值的求解个数
 *
 * @param solver      输入/输出向量
 * @param block_size  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_block_size(PASE_MG_SOLVER solver, PASE_INT block_size)
{
  solver->block_size        = block_size;
  solver->actual_block_size = block_size;
  return 0;
}

/**
 * @brief 设置特征值的最大求解个数, 主要用于申请足够大的内存空间存放求解的特征向量组.
 *
 * @param solver          输入/输出向量
 * @param max_block_size  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_max_block_size(PASE_MG_SOLVER solver, PASE_INT max_block_size)
{
  solver->max_block_size = max_block_size;
  return 0;
}

/**
 * @brief 设置 MG 方法的最大迭代步数
 *
 * @param solver     输入/输出向量
 * @param max_cycle  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_max_cycle(PASE_MG_SOLVER solver, PASE_INT max_cycle)
{
  solver->max_cycle = max_cycle;
  return 0;
}

/**
 * @brief 设置 MG 方法每个迭代步中, 前光滑的最大光滑步数
 *
 * @param solver        输入/输出向量
 * @param max_pre_iter  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_max_pre_iteration(PASE_MG_SOLVER solver, PASE_INT max_pre_iter)
{
  solver->max_pre_iter = max_pre_iter;
  return 0;
}

/**
 * @brief 设置 MG 方法每个迭代步中, 后光滑的最大光滑步数
 *
 * @param solver         输入/输出向量
 * @param max_post_iter  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_max_post_iteration(PASE_MG_SOLVER solver, PASE_INT max_post_iter)
{
  solver->max_post_iter = max_post_iter;
  return 0;
}

/**
 * @brief 设置 MG 方法停止迭代需满足的绝对残差精度要求
 *
 * @param solver  输入/输出向量
 * @param atol    输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_atol(PASE_MG_SOLVER solver, PASE_REAL atol)
{
  solver->atol = atol;
  return 0;
}

/**
 * @brief 设置 MG 方法停止迭代需满足的相对残差精度要求
 *
 * @param solver  输入/输出向量
 * @param rtol    输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_rtol(PASE_MG_SOLVER solver, PASE_REAL rtol)
{
  solver->rtol = rtol;
  return 0;
}

/**
 * @brief 设置 MG 方法的信息打印级别
 *
 * @param solver       输入/输出向量
 * @param print_level  输入参数
 *
 * @return 
 */
PASE_INT 
PASE_Mg_set_print_level(PASE_MG_SOLVER solver, PASE_INT print_level)
{
  solver->print_level = print_level;
  return 0;
}

/**
 * @brief 设置问题的精确特征值, 可用于算法测试
 *
 * @param solver             输入/输出向量
 * @param exact_eigenvalues  输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_set_exact_eigenvalues(PASE_MG_SOLVER solver, PASE_SCALAR *exact_eigenvalues)
{
  solver->exact_eigenvalues = exact_eigenvalues;
  return 0;
}

/**
 * @brief 用 cg 方法做光滑
 *
 * @param mg_solver  输入/输出向量
 *
 * @return 
 */
PASE_INT
PASE_Mg_presmoothing_by_cg(void *mg_solver)
{
  PASE_MG_SOLVER solver = (PASE_MG_SOLVER)mg_solver;
  PASE_INT     cur_level  = solver->cur_level;
  PASE_INT     block_size = solver->block_size;
  PASE_INT     max_iter   = solver->max_pre_iter;
  PASE_SCALAR *eigenvalues = solver->eigenvalues;

  PASE_INT idx_eigen;
  PASE_REAL tol = 1e-10;
  PASE_REAL inner_A, inner_B;
  PASE_VECTOR b = PASE_Vector_create_by_vector(solver->u[0]);
  //clock_t start, end;
  for(idx_eigen = solver->nconv; idx_eigen < block_size; idx_eigen++) {
    //start = clock();
    PASE_Matrix_multiply_vector_general(eigenvalues[idx_eigen], solver->multigrid->B[0], solver->u[idx_eigen], 0.0, b);
    PASE_Linear_solve_by_cg(solver->multigrid->A[0], b, solver->u[idx_eigen], tol, max_iter);

    PASE_Vector_inner_product_general(solver->u[idx_eigen], solver->u[idx_eigen], solver->multigrid->A[0], &inner_A);
    PASE_Vector_inner_product_general(solver->u[idx_eigen], solver->u[idx_eigen], solver->multigrid->B[0], &inner_B);
    eigenvalues[idx_eigen] = inner_A / inner_B;
    //PASE_Printf(MPI_COMM_WORLD, "after Cg, eigenvalues[%d] = %.12e\n", j, eigenvalues[j]);
    //end = clock();
    //PASE_Printf(MPI_COMM_WORLD, "the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
  }
  PASE_Vector_destroy(b);

#if 1
  if( solver->print_level > 1) {
    PASE_Printf(MPI_COMM_WORLD, "Cur_level %d", cur_level);
    for(idx_eigen=0; idx_eigen<block_size; idx_eigen++)
    {
      PASE_Printf(MPI_COMM_WORLD, ", eigen[%d] = %.16f", idx_eigen, eigenvalues[idx_eigen]);
    }
    PASE_Printf(MPI_COMM_WORLD, ".\n");
  }
#endif

  return 0;
}

/**
 * @brief 用 cg 方法做辅助空间内的光滑
 *
 * @param mg_solver  输入/输出向量
 *
 * @return
 */
PASE_INT
PASE_Mg_presmoothing_by_cg_aux(void *mg_solver)
{
  PASE_MG_SOLVER solver     = (PASE_MG_SOLVER)mg_solver;
  PASE_INT       cur_level  = solver->cur_level;
  PASE_INT       block_size = solver->block_size;
  PASE_INT       max_iter   = solver->max_pre_iter + cur_level * 0.5;
  PASE_SCALAR   *eigenvalues = solver->eigenvalues;

  PASE_INT idx_eigen;
  PASE_REAL tol = 1e-10;
  PASE_REAL inner_A, inner_B;
  PASE_AUX_VECTOR *aux_u = solver->aux_u[cur_level];
  PASE_AUX_VECTOR  aux_b = PASE_Aux_vector_create_by_aux_vector(aux_u[0]);
  //clock_t start, end;
  for(idx_eigen = solver->nconv; idx_eigen < block_size; idx_eigen++) {
    //start = clock();
    PASE_Aux_matrix_multiply_aux_vector_general(eigenvalues[idx_eigen], solver->multigrid->aux_B[cur_level], aux_u[idx_eigen], 0.0, aux_b);
    PASE_Linear_solve_by_cg_aux(solver->multigrid->aux_A[cur_level], aux_b, aux_u[idx_eigen], tol, max_iter);

    PASE_Aux_vector_inner_product_general(aux_u[idx_eigen], aux_u[idx_eigen], solver->multigrid->aux_A[cur_level], &inner_A);
    PASE_Aux_vector_inner_product_general(aux_u[idx_eigen], aux_u[idx_eigen], solver->multigrid->aux_B[cur_level], &inner_B);
    eigenvalues[idx_eigen] = inner_A / inner_B;
    //PASE_Printf(MPI_COMM_WORLD, "after Cg, eigenvalues[%d] = %.12e\n", j, eigenvalues[j]);
    //end = clock();
    //PASE_Printf(MPI_COMM_WORLD, "the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
  }
  PASE_Aux_vector_destroy(aux_b);

#if 1
  if( solver->print_level > 1) {
    PASE_Printf(MPI_COMM_WORLD, "Cur_level %d", cur_level);
    for( idx_eigen=0; idx_eigen<block_size; idx_eigen++) {
      PASE_Printf(MPI_COMM_WORLD, ", eigen[%d] = %.16f", idx_eigen, eigenvalues[idx_eigen]);
    }
    PASE_Printf(MPI_COMM_WORLD, ".\n");
  }
#endif

  return 0;
}

/**
 * @brief 用 cg 方法解线性方程组
 *
 * @param A         输入参数
 * @param b         输入参数
 * @param x         输入/输出参数
 * @param tol       输入参数
 * @param max_iter  输入参数
 *
 * @return 
 */
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
  if(rnorm/bnorm < tol) {
    return 0;
  }
  for(iter=0; iter<max_iter; iter++) {
    if(iter>0) {
      beta = rho / rho_1; 
      PASE_Vector_scale(beta, p);
      PASE_Vector_axpy(1.0, r, p);
    } else {
      PASE_Vector_copy(r, p);
    }
    PASE_Matrix_multiply_vector(A, p, q);
    PASE_Vector_inner_product(p, q, &tmp);
    alpha = rho / tmp;
    PASE_Vector_axpy(alpha, p, x);
    PASE_Vector_axpy(-1.0*alpha, q, r);

    rho_1 = rho;
    PASE_Vector_inner_product(r, r, &rho);
    rnorm = sqrt(rho);

    if(rnorm/bnorm < tol) {
      break;
    }
  }
  //end = clock();
  //PASE_Printf(MPI_COMM_WORLD, "the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
  //PASE_Printf(MPI_COMM_WORLD, "iter = %d\n", iter);
  PASE_Vector_destroy(r);
  PASE_Vector_destroy(p);
  PASE_Vector_destroy(q);
  return 0;
}

/**
 * @brief 用 cg 方法解辅助空间内的线性方程组
 *
 * @param aux_A     输入参数
 * @param aux_b     输入参数
 * @param aux_x     输入/输出参数
 * @param tol       输入参数
 * @param max_iter  输入参数
 *
 * @return 
 */
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
  if(rnorm/bnorm < tol) {
    return 0;
  }
  for(iter = 0; iter < max_iter; iter++) {
    if(iter>0) {
      beta = rho / rho_1; 
      PASE_Aux_vector_scale(beta, aux_p);
      PASE_Aux_vector_axpy(1.0, aux_r, aux_p);
    } else {
      PASE_Aux_vector_copy(aux_r, aux_p);
    }
    PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_p, aux_q);
    PASE_Aux_vector_inner_product(aux_p, aux_q, &tmp);
    alpha = rho / tmp;
    PASE_Aux_vector_axpy(alpha, aux_p, aux_x);
    PASE_Aux_vector_axpy(-1.0*alpha, aux_q, aux_r);

    rho_1 = rho;
    PASE_Aux_vector_inner_product(aux_r, aux_r, &rho);
    rnorm = sqrt(rho);

    if(rnorm/bnorm < tol) {
      break;
    }
  }
  //end = clock();
  //PASE_Printf(MPI_COMM_WORLD, "the %dth eigenvalue, cg time = %.4e, iter = %d\n", idx_eigen, ((double)(end-start))/CLK_TCK, iter);
  PASE_Aux_vector_destroy(aux_r);
  PASE_Aux_vector_destroy(aux_p);
  PASE_Aux_vector_destroy(aux_q);
  return 0;
}

/**
 * @brief 设置最粗层的辅助空间
 *
 * @param solver  输入/输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_coarsest_aux_matrix_set(PASE_MG_SOLVER solver)
{
  PASE_INT         max_level          = solver->max_level;
  PASE_INT         block_size         = solver->block_size;
  PASE_INT         nlock = solver->nlock;
  PASE_INT         idx_block          = 0;
  PASE_MATRIX     *A                  = solver->multigrid->A;
  PASE_MATRIX     *B                  = solver->multigrid->B;
  PASE_VECTOR     *u_h                = solver->u;
  PASE_AUX_MATRIX  aux_A              = solver->multigrid->aux_A[max_level];
  PASE_AUX_MATRIX  aux_B              = solver->multigrid->aux_B[max_level];

  if(NULL == aux_A && NULL == aux_B) {
    PASE_Mg_coarsest_aux_matrix_create(solver, &(solver->multigrid->aux_A[max_level]), A[max_level]);
    PASE_Mg_coarsest_aux_matrix_create(solver, &(solver->multigrid->aux_B[max_level]), B[max_level]);
    aux_A = solver->multigrid->aux_A[max_level];
    aux_B = solver->multigrid->aux_B[max_level];
  }

  PASE_VECTOR Au        = NULL;
  for(idx_block = nlock; idx_block < block_size; idx_block++) {
    Au = PASE_Vector_create_by_vector(u_h[0]);
    PASE_Matrix_multiply_vector(A[0], u_h[idx_block], Au);
    PASE_Mg_restrict(solver, 0, Au, max_level, aux_A->vec[idx_block]);
    PASE_Matrix_multiply_vector(B[0], u_h[idx_block], Au);
    PASE_Mg_restrict(solver, 0, Au, max_level, aux_B->vec[idx_block]);
    PASE_Vector_destroy(Au);
  }
  PASE_Aux_matrix_set_block_some(aux_A, nlock, block_size-1, A[0], u_h);
  PASE_Aux_matrix_set_block_some(aux_B, nlock, block_size-1, B[0], u_h);
  return 0;
}

/**
 * @brief 最粗层辅助空间的内存分配
 *
 * @param solver  输入参数
 * @param aux_A   输入/输出参数
 * @param A_H     输入参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_coarsest_aux_matrix_create(PASE_MG_SOLVER solver, PASE_AUX_MATRIX *aux_A, PASE_MATRIX A_H)
{
  PASE_INT idx_block   = 0;
  PASE_INT block_size  = solver->block_size;
  *aux_A               = (PASE_AUX_MATRIX)PASE_Malloc(sizeof(PASE_AUX_MATRIX_PRIVATE));
  (*aux_A)->mat          = A_H;
  (*aux_A)->is_mat_owner = PASE_NO;
  (*aux_A)->block_size   = block_size;

  (*aux_A)->vec          = (PASE_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_VECTOR));
  for(idx_block = 0; idx_block < block_size; idx_block++) {
    (*aux_A)->vec[idx_block] = PASE_Vector_create_by_matrix_and_vector_data_operator(A_H, solver->u[0]->ops);
  }

  (*aux_A)->block = (PASE_SCALAR**)PASE_Malloc(block_size*sizeof(PASE_SCALAR*));
  for(idx_block = 0; idx_block < block_size; idx_block++) {
    (*aux_A)->block[idx_block] = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
  }

  return 0;
}

/**
 * @brief 将第 i 层网格的向量 u_i 限制到第 j 层网格上, 并储存在向量 u_j 中
 *
 * @param solver  输入参数
 * @param i       输入参数
 * @param u_i     输入参数
 * @param j       输入参数
 * @param u_j     输出参数
 *
 * @return 
 */
PASE_INT
PASE_Mg_restrict(PASE_MG_SOLVER solver, PASE_INT i, PASE_VECTOR u_i, PASE_INT j, PASE_VECTOR u_j)
{
  PASE_INT    idx_level = 0;
  PASE_VECTOR tmp_u_H   = NULL;
  PASE_VECTOR tmp_u_h   = u_i;
  PASE_MATRIX *A = solver->multigrid->A;
  PASE_MATRIX *R = solver->multigrid->R;
  for(idx_level = i; idx_level <= j-2; idx_level++) {
    tmp_u_H = PASE_Vector_create_by_matrix_and_vector_data_operator(A[idx_level+1], u_i->ops);
    PASE_Matrix_multiply_vector(R[idx_level], tmp_u_h, tmp_u_H);
    if(idx_level > i) {
      PASE_Vector_destroy(tmp_u_h);
    }
    tmp_u_h = tmp_u_H;
  }
  PASE_Matrix_multiply_vector(R[j-1], tmp_u_h, u_j);
  if(j > i+1) {
    PASE_Vector_destroy(tmp_u_h);
  }
  return 0;
}

/**
 * @brief 
 *
 * @param A
 * @param B
 * @param eval
 * @param evec
 * @param block_size
 * @param param
 */
PASE_INT
PASE_EigenSolver(PASE_MATRIX A, PASE_MATRIX B, PASE_SCALAR *eval, PASE_VECTOR *evec, PASE_INT block_size, PASE_PARAMETER param)
{
  PASE_INT i              = 0;
  PASE_INT max_block_size = ((2*block_size)<(block_size+5))?(2*block_size):(block_size+5);
  PASE_MG_SOLVER solver   = PASE_Mg_solver_create(A, B, param, NULL);

  PASE_Mg_set_block_size(solver, block_size);
  PASE_Mg_set_max_block_size(solver, max_block_size);
  PASE_Mg_set_max_cycle(solver, param->max_cycle);
  PASE_Mg_set_max_pre_iteration(solver, param->max_pre_iter);
  PASE_Mg_set_max_post_iteration(solver, param->max_post_iter);
  PASE_Mg_set_atol(solver, param->atol);
  PASE_Mg_set_rtol(solver, param->rtol);
  PASE_Mg_set_print_level(solver, param->print_level);

  solver->u = (PASE_VECTOR*)PASE_Malloc(max_block_size*sizeof(PASE_VECTOR));
  if(NULL != evec) {
    for(i = 0; i < block_size; i++) {
      solver->u[i] = evec[i];
    }
    for(i = block_size; i < max_block_size; i++) {
      solver->u[i] = PASE_Vector_create_by_vector(solver->u[0]);
    }
    solver->is_u_owner = PASE_NO;
  }
  PASE_Mg_set_up(solver, NULL);
  PASE_Mg_solve(solver);
  PASE_Mg_solver_destroy(solver);
  return 0;
}

/**
 * @brief 打印当前的近似特征向量和当前所在的层数
 *
 * @param solver
 *
 * @return 
 */
PASE_INT
PASE_Mg_print_eigenvalue_of_current_level(PASE_MG_SOLVER solver)
{
  PASE_INT i = 0;
  if(solver->print_level > 1) {
    PASE_Printf(MPI_COMM_WORLD, "Cur_level %d", solver->cur_level);
    for(i = 0; i < solver->block_size; i++) {
      PASE_Printf(MPI_COMM_WORLD, ", eigen[%d] = %.16f", i, solver->eigenvalues[i]);
    }
    PASE_Printf(MPI_COMM_WORLD, ".\n");
  }
  return 0;
}

PASE_INT
PASE_Mg_direct_solve_by_gcg(void *mg_solver)
{
  PASE_MG_SOLVER   solver      = (PASE_MG_SOLVER)mg_solver;
  PASE_INT         max_level   = solver->max_level;
  PASE_INT         block_size  = solver->block_size;

  PASE_AUX_MATRIX  aux_A       = solver->multigrid->aux_A[max_level];            
  PASE_AUX_MATRIX  aux_B       = solver->multigrid->aux_B[max_level];            
  PASE_AUX_VECTOR *aux_u       = solver->aux_u[max_level];
  PASE_SCALAR     *eigenvalues = solver->eigenvalues;

  PASE_INT         max_iter    = 10;
  PASE_REAL        tol         = solver->atol;

#if 0
  PASE_INT        i      = 0;
  PASE_REAL       r_norm = 0;
  PASE_AUX_VECTOR tmp    = PASE_Aux_vector_create_by_aux_vector(aux_u[0]);
  for(i = 0; i<block_size; i++) {
    PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_u[i], tmp);
    PASE_Aux_matrix_multiply_aux_vector_general(-eigenvalues[i], aux_B, aux_u[i], 1.0, tmp);
    PASE_Aux_vector_inner_product(tmp, tmp, &r_norm);
    r_norm	= sqrt(r_norm);
    PASE_Printf(MPI_COMM_WORLD, "eigenvalues[%d] = %.8e, residual[%d] = %.6e\n", i, eigenvalues[i], i, r_norm);
  }
  PASE_Printf(MPI_COMM_WORLD, "\n");
#endif

#if 1
#endif

  PASE_Printf(MPI_COMM_WORLD, "Begin: solve directly by gcg\n");
  //PASE_Printf(MPI_COMM_WORLD, "\n");
  GCG_Eigen(aux_A, aux_B, eigenvalues, aux_u, block_size, tol*1e-2, tol, max_iter, 10, solver->nconv);
  //PASE_Printf(MPI_COMM_WORLD, "\n");
  PASE_Printf(MPI_COMM_WORLD, "Done: solve directly by gcg\n");

#if 0
  for(i = 0; i < block_size; i++) {
    PASE_Aux_matrix_multiply_aux_vector(aux_A, aux_u[i], tmp);
    PASE_Aux_matrix_multiply_aux_vector_general(-eigenvalues[i], aux_B, aux_u[i], 1.0, tmp);
    PASE_Aux_vector_inner_product(tmp, tmp, &r_norm);
    r_norm	= sqrt(r_norm);
    PASE_Printf(MPI_COMM_WORLD, "eigenvalues[%d] = %.8e, residual[%d] = %.6e\n", i, eigenvalues[i], i, r_norm);
  }
  PASE_Printf(MPI_COMM_WORLD, "\n");
  PASE_Aux_vector_destroy(tmp);
#endif
  return 0;
}

PASE_INT
PASE_Mg_set_pase_aux_matrix_by_pase_matrix(PASE_MG_SOLVER solver, PASE_INT i, PASE_INT j, PASE_VECTOR *u_j)
{
  PASE_INT block_size = solver->block_size;
  PASE_INT idx_block  = 0;
  PASE_MATRIX *A = solver->multigrid->A;
  PASE_MATRIX *B = solver->multigrid->B;
  PASE_Mg_pase_aux_matrix_create(solver, i);
  PASE_AUX_MATRIX aux_A = solver->multigrid->aux_A[i];
  PASE_AUX_MATRIX aux_B = solver->multigrid->aux_B[i];
  
  PASE_VECTOR Au = NULL;
  for(idx_block = 0; idx_block < block_size; idx_block++) {
    Au = PASE_Vector_create_by_vector(u_j[0]);
    PASE_Matrix_multiply_vector(A[j], u_j[idx_block], Au);
    PASE_Mg_restrict(solver, j, Au, i, aux_A->vec[idx_block]);
    PASE_Matrix_multiply_vector(B[j], u_j[idx_block], Au);
    PASE_Mg_restrict(solver, j, Au, i, aux_B->vec[idx_block]);
    PASE_Vector_destroy(Au);
  }
  PASE_Aux_matrix_set_block_some(aux_A, 0, block_size-1, A[j], u_j);
  PASE_Aux_matrix_set_block_some(aux_B, 0, block_size-1, B[j], u_j);
  return 0;
}

PASE_INT
PASE_Mg_pase_aux_matrix_create(PASE_MG_SOLVER solver, PASE_INT i)
{
  PASE_INT         block_size = solver->block_size;
  PASE_INT         idx_block  = 0;
  PASE_MATRIX     *A     = solver->multigrid->A;
  PASE_MATRIX     *B     = solver->multigrid->B;
  PASE_AUX_MATRIX *aux_A = solver->multigrid->aux_A;
  PASE_AUX_MATRIX *aux_B = solver->multigrid->aux_B;
  if(NULL == aux_A[i]) {
    aux_A[i] = (PASE_AUX_MATRIX)PASE_Malloc(sizeof(PASE_AUX_MATRIX_PRIVATE));
    aux_A[i]->mat = A[i];
    aux_A[i]->is_mat_owner = PASE_NO;
    aux_A[i]->block_size   = block_size;

    aux_A[i]->vec = (PASE_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_VECTOR));
    aux_A[i]->block = (PASE_SCALAR**)PASE_Malloc(block_size*sizeof(PASE_SCALAR*));
    for(idx_block = 0; idx_block < block_size; idx_block++) {
      aux_A[i]->vec[idx_block] = PASE_Vector_create_by_matrix_and_vector_data_operator(A[i], solver->u[0]->ops);
      aux_A[i]->block[idx_block] = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
    }
  }
  if(NULL == aux_B[i]) {
    aux_B[i] = (PASE_AUX_MATRIX)PASE_Malloc(sizeof(PASE_AUX_MATRIX_PRIVATE));
    aux_B[i]->mat = B[i];
    aux_B[i]->is_mat_owner = PASE_NO;
    aux_B[i]->block_size   = block_size;

    aux_B[i]->vec = (PASE_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_VECTOR));
    aux_B[i]->block = (PASE_SCALAR**)PASE_Malloc(block_size*sizeof(PASE_SCALAR*));
    for(idx_block = 0; idx_block < block_size; idx_block++) {
      aux_B[i]->vec[idx_block] = PASE_Vector_create_by_matrix_and_vector_data_operator(B[i], solver->u[0]->ops);
      aux_B[i]->block[idx_block] = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
    }
  }
  return 0;
}

PASE_INT
PASE_Mg_prolong_from_pase_aux_vector_to_pase_vector(PASE_MG_SOLVER solver, PASE_INT i, PASE_AUX_VECTOR *aux_u_i, PASE_INT j, PASE_VECTOR *u_j)
{
  PASE_INT idx_block = 0;
  PASE_INT jdx_block = 0;
  PASE_INT block_size = solver->block_size;
  PASE_VECTOR *u_h = (PASE_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_VECTOR));
  for(idx_block = 0; idx_block < block_size; idx_block++) {
    u_h[idx_block] = PASE_Vector_create_by_vector(u_j[0]);
    PASE_Mg_prolong(solver, i, aux_u_i[idx_block]->vec, j, u_h[idx_block]);
    for(jdx_block = 0; jdx_block < block_size; jdx_block++) {
      PASE_Vector_axpy(aux_u_i[idx_block]->block[jdx_block], u_j[jdx_block], u_h[idx_block]);
    }
  }
  for(idx_block = 0; idx_block < block_size; idx_block++) {
    PASE_Vector_copy(u_h[idx_block], u_j[idx_block]);
    PASE_Vector_destroy(u_h[idx_block]);
  }
  PASE_Free(u_h);

  return 0;
}

