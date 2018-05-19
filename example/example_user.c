#include "pase_mg_solver.h"
#include "pase_multigrid.h"
#include "pase_config.h"
#include "pase_matrix_hypre.h"
#include "pase_vector_hypre.h"
#include "pase_multigrid_hypre.h"

#include "HYPRE_seq_mv.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "temp_multivector.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "HYPRE_utilities.h"
#include "HYPRE_lobpcg.h"
#include "lobpcg.h"

void GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n, PASE_INT *block_size, PASE_REAL *atol, PASE_INT *max_pre_iter, PASE_INT *max_post_iter, PASE_INT *max_direct_iter, PASE_INT *max_level);
void GetEigenProblem(HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, PASE_INT n);
void CreatePaseMatrix();
void CreatePaseVector();
void CreatePaseMultigrid();
void PrintParameter(PASE_PARAMETER param);

PASE_INT main(PASE_INT argc, char *argv[])
{
  PASE_INT myid = 0;
#if PASE_USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#endif

  /* set parameters */
  PASE_INT       n       = 200;
  PASE_PARAMETER param   = (PASE_PARAMETER) PASE_Malloc(sizeof(PASE_PARAMETER_PRIVATE));
  param->data_form       = -1;
  param->cycle_type      = 0;
  param->block_size      = 5;
  param->max_cycle       = 100;
  param->max_pre_iter    = 0;
  param->max_post_iter   = 2;
  param->max_direct_iter = 3;
  param->atol            = 1e-8;
  param->rtol            = 1e-6;
  param->print_level     = 1;
  param->max_level       = 20;
  GetCommandLineInfo(argc, argv, &n, &(param->block_size), &(param->atol), &(param->max_pre_iter), &(param->max_post_iter), &(param->max_direct_iter), &(param->max_level));
  //PASE_INT  min_coarse_size = block_size * 30;
  param->min_coarse_size = 4000;
  param->max_block_size  = ((2*param->block_size)<(param->block_size+5))?(2*param->block_size):(param->block_size+5);
  PASE_Printf(MPI_COMM_WORLD, "The dimension of the eigenvalue problem = %d\n", n*n);
  PrintParameter(param);

  /* Create matrix */
  HYPRE_IJMatrix A, B;
  HYPRE_ParCSRMatrix parcsr_A, parcsr_B;
  GetEigenProblem(&A, &B, n);
  HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
  HYPRE_IJMatrixGetObject(B, (void**) &parcsr_B);

  PASE_MATRIX_DATA_OPERATOR 
    ops_matrix = PASE_Matrix_data_operator_assign(PASE_Matrix_create_by_matrix_hypre,
                                                  PASE_Matrix_destroy_hypre,
                                                  PASE_Matrix_transpose_hypre,
                                                  PASE_Matrix_copy_hypre,
                                                  PASE_Matrix_multiply_matrix_hypre,
                                                  PASE_MatrixT_multiply_matrix_hypre,
                                                  PASE_Matrix_multiply_vector_hypre,
                                                  PASE_MatrixT_multiply_vector_hypre,
                                                  PASE_Matrix_multiply_vector_general_hypre,
                                                  PASE_MatrixT_multiply_vector_general_hypre,
                                                  PASE_Matrix_get_global_nrow_hypre,
                                                  PASE_Matrix_get_global_ncol_hypre,
                                                  PASE_Matrix_get_mpi_comm_hypre);
  PASE_MATRIX pase_A = PASE_Matrix_assign((void*)parcsr_A, ops_matrix);
  PASE_MATRIX pase_B = PASE_Matrix_assign((void*)parcsr_B, ops_matrix);
  PASE_Matrix_data_operator_destroy(ops_matrix);


  /* Create vector */
  HYPRE_ParVector *parcsr_evec  = (HYPRE_ParVector*) PASE_Malloc(param->block_size*sizeof(HYPRE_ParVector));
  PASE_INT         i            = 0;
  MPI_Comm         comm         = hypre_ParCSRMatrixComm(parcsr_A);
  PASE_INT         global_size  = hypre_ParCSRMatrixGlobalNumRows(parcsr_A);
  PASE_INT        *partitioning = NULL;
  HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_A,  &partitioning);
  for(i = 0; i < param->block_size; ++i) {
    parcsr_evec[i]= hypre_ParVectorCreate(comm,  global_size,  partitioning);
    HYPRE_ParVectorInitialize(parcsr_evec[i]);
    hypre_ParVectorSetPartitioningOwner(parcsr_evec[i], 0); 
  }
  hypre_ParVectorSetPartitioningOwner(parcsr_evec[0], 1); 

  PASE_VECTOR_DATA_OPERATOR
    ops_vector = PASE_Vector_data_operator_assign(PASE_Vector_data_create_by_vector_hypre,
                                                  PASE_Vector_data_create_by_matrix_hypre,
                                                  PASE_Vector_data_copy_hypre,
                                                  PASE_Vector_data_destroy_hypre,
                                                  PASE_Vector_data_set_constant_value_hypre,
                                                  PASE_Vector_data_set_random_value_hypre,
                                                  PASE_Vector_data_inner_product_hypre,
                                                  PASE_Vector_data_axpy_hypre,
                                                  PASE_Vector_data_scale_hypre,
                                                  PASE_Vector_data_get_global_nrow_hypre);
  PASE_VECTOR *evec = (PASE_VECTOR*) PASE_Malloc(param->block_size*sizeof(PASE_VECTOR));
  for(i = 0; i < param->block_size; ++i) {
    evec[i] = PASE_Vector_assign(parcsr_evec[i], ops_vector);
  }
  PASE_Vector_data_operator_destroy(ops_vector);

  PASE_REAL *eigenvalues = (PASE_REAL*)PASE_Malloc(param->block_size*sizeof(PASE_REAL));

  /* Create multigrid */
  PASE_MULTIGRID_OPERATOR 
    ops_multigrid = PASE_Multigrid_operator_assign(PASE_Multigrid_get_amg_array_hypre,
                                                   PASE_Multigrid_destroy_amg_data_hypre);
  PASE_MULTIGRID multigrid = PASE_Multigrid_create(pase_A, pase_B, param, ops_multigrid);
  PASE_Multigrid_operator_destroy(ops_multigrid);

  /* solve */
  PASE_Eigensolver_user(multigrid, eigenvalues, evec, param->block_size, param);

  /* Destroy */
  PASE_Matrix_destroy(pase_A);
  PASE_Matrix_destroy(pase_B);
  for(i = 0; i < param->block_size; ++i) {
    PASE_Vector_destroy(evec[i]);
    HYPRE_ParVectorDestroy(parcsr_evec[i]);
  }
  PASE_Free(parcsr_evec);
  PASE_Free(eigenvalues);
  PASE_Free(evec);
  //PASE_Free(exact_eigenvalues);
  HYPRE_IJMatrixDestroy(A);
  HYPRE_IJMatrixDestroy(B);
  PASE_Free(param);

  MPI_Finalize();
  return(0);
}

void GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n, PASE_INT *block_size, PASE_REAL *atol, PASE_INT *max_pre_iter, PASE_INT *max_post_iter, PASE_INT *max_direct_iter, PASE_INT *max_level)
{
  PASE_INT arg_index = 0;
  PASE_INT print_usage = 0;
  PASE_INT myid;

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  while(arg_index < argc) {
    if(strcmp(argv[arg_index], "-n") == 0) {
      arg_index++;
      *n = atoi(argv[arg_index++]);
    } else if(strcmp(argv[arg_index], "-block_size") == 0 ) {
      arg_index++;
      *block_size = atoi(argv[arg_index++]);
    } else if(strcmp(argv[arg_index], "-atol") == 0) {
      arg_index++;
      *atol= pow(10, atoi(argv[arg_index++]));
    } else if(strcmp(argv[arg_index], "-max_pre_iter") == 0) {
      arg_index++;
      *max_pre_iter = atoi(argv[arg_index++]);
    } else if(strcmp(argv[arg_index], "-max_post_iter") == 0) {
      arg_index++;
      *max_post_iter = atoi(argv[arg_index++]);
    } else if(strcmp(argv[arg_index], "-max_direct_iter") == 0) {
      arg_index++;
      *max_direct_iter = atoi(argv[arg_index++]);
    } else if(strcmp(argv[arg_index], "-max_levels") == 0) {
      arg_index++;
      *max_level = atoi(argv[arg_index++]);
    } else if(strcmp(argv[arg_index], "-help") == 0) {
      print_usage = 1;
      break;
    } else {
      arg_index++;
    }
  }

  if(print_usage) {
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "Usage: %s [<options>]\n", argv[0]);
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "  -n <n>               : problem size in each direction (default: 200)\n");
    PASE_Printf(MPI_COMM_WORLD, "  -block_size <n>      : eigenproblem block size (default: 5)\n");
    PASE_Printf(MPI_COMM_WORLD, "  -max_pre_iter <n>    : (default: 0)\n");
    PASE_Printf(MPI_COMM_WORLD, "  -max_post_iter <n>   : (default: 1)\n");
    PASE_Printf(MPI_COMM_WORLD, "  -max_levels <n>      : max levels of AMG (default: 10)\n");
    PASE_Printf(MPI_COMM_WORLD, "\n");
    exit(-1);
  }
}

void GetEigenProblem(HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, PASE_INT n)
{
  PASE_INT i;
  PASE_INT ilower, iupper;
  PASE_INT local_size, extra;
  PASE_INT myid, num_procs;

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  /* Each processor knows only of its own rows - the range is denoted by ilower
     and iupper.  Here we partition the rows. We account for the fact that
     N may not divide evenly by the number of processors. */
  PASE_INT dim = n*n;
  local_size = dim/num_procs;
  extra = dim - local_size*num_procs;

  ilower = local_size*myid;
  ilower += hypre_min(myid, extra);

  iupper = local_size*(myid+1);
  iupper += hypre_min(myid+1, extra);
  iupper = iupper - 1;

  /* How many rows do I have? */
  local_size = iupper - ilower + 1;

  /* Create the matrix.
     Note that this is a square matrix, so we indicate the row partition
     size twice (since number of rows = number of cols) */
  HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, A);
  HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, B);

  /* Choose a parallel csr format storage (see the User's Manual) */
  HYPRE_IJMatrixSetObjectType(*A, HYPRE_PARCSR);
  HYPRE_IJMatrixSetObjectType(*B, HYPRE_PARCSR);

  /* Initialize before setting coefficients */
  HYPRE_IJMatrixInitialize(*A);
  HYPRE_IJMatrixInitialize(*B);

  /* Now go through my local rows and set the matrix entries.
     Each row has at most 5 entries. For example, if n=3:

     A = [M -I 0; -I M -I; 0 -I M]
     M = [4 -1 0; -1 4 -1; 0 -1 4]

     Note that here we are setting one row at a time, though
     one could set all the rows together (see the User's Manual).
     */
  {
    PASE_INT nnz;
    PASE_REAL values[5];
    PASE_INT cols[5];

    for (i = ilower; i <= iupper; i++)
    {
      nnz = 0;

      /* The left identity block:position i-n */
      if ((i-n)>=0)
      {
	cols[nnz] = i-n;
	values[nnz] = -1.0;
	nnz++;
      }

      /* The left -1: position i-1 */
      if (i%n)
      {
	cols[nnz] = i-1;
	values[nnz] = -1.0;
	nnz++;
      }

      /* Set the diagonal: position i */
      cols[nnz] = i;
      values[nnz] = 4.0;
      nnz++;

      /* The right -1: position i+1 */
      if ((i+1)%n)
      {
	cols[nnz] = i+1;
	values[nnz] = -1.0;
	nnz++;
      }

      /* The right identity block:position i+n */
      if ((i+n)< dim)
      {
	cols[nnz] = i+n;
	values[nnz] = -1.0;
	nnz++;
      }

      /* Set the values for row i */
      HYPRE_IJMatrixSetValues(*A, 1, &nnz, &i, cols, values);
    }
  }
  {
    PASE_INT nnz;
    PASE_REAL values[5];
    PASE_INT cols[5];
    for (i = ilower; i <= iupper; i++)
    {
      nnz = 1;
      cols[0] = i;
      values[0] = 1.0;
      /* Set the values for row i */
      HYPRE_IJMatrixSetValues(*B, 1, &nnz, &i, cols, values);
    }
  }

  /* Assemble after setting the coefficients */
  HYPRE_IJMatrixAssemble(*A);
  HYPRE_IJMatrixAssemble(*B);
}

void PrintParameter(PASE_PARAMETER param)
{
    PASE_Printf(MPI_COMM_WORLD, "PASE (Parallel Auxiliary Space Eigen-solver), parallel version\n"); 
    PASE_Printf(MPI_COMM_WORLD, "Please contact liyu@lsec.cc.ac.cn, if there is any bugs.\n"); 
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n" );
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "Set parameters:\n");
    PASE_Printf(MPI_COMM_WORLD, "cycle type      = %d\n", param->cycle_type);
    PASE_Printf(MPI_COMM_WORLD, "block size      = %d\n", param->block_size);
    PASE_Printf(MPI_COMM_WORLD, "max block size  = %d\n", param->max_block_size);
    PASE_Printf(MPI_COMM_WORLD, "max pre iter    = %d\n", param->max_pre_iter);
    PASE_Printf(MPI_COMM_WORLD, "max post iter   = %d\n", param->max_post_iter);
    PASE_Printf(MPI_COMM_WORLD, "max direct iter = %d\n", param->max_direct_iter);
    PASE_Printf(MPI_COMM_WORLD, "atol            = %e\n", param->atol);
    PASE_Printf(MPI_COMM_WORLD, "max cycle       = %d\n", param->max_cycle);
    PASE_Printf(MPI_COMM_WORLD, "max level       = %d\n", param->max_level);
    PASE_Printf(MPI_COMM_WORLD, "min coarse size = %d\n", param->min_coarse_size);
    PASE_Printf(MPI_COMM_WORLD, "\n");
}
