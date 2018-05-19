#include "pase_mg_solver.h"
#include "pase_multigrid.h"

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

void GetEigenProblem(HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, PASE_INT n);
void GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n, PASE_INT *block_size, PASE_REAL *atol, PASE_INT *nsmooth);
void PrintParameter(PASE_PARAMETER param);

PASE_INT main(PASE_INT argc, char *argv[])
{
  PASE_INT  myid;
#if PASE_USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
#endif

  //set parameters
  PASE_PARAMETER param   = (PASE_PARAMETER) PASE_Malloc(sizeof(PASE_PARAMETER_PRIVATE));
  param->data_form       = DATA_FORM_HYPRE;
  param->cycle_type      = 0;
  param->block_size      = 5;
  param->max_cycle       = 50;
  param->max_pre_iter    = 1;
  param->max_post_iter   = 1;
  param->atol            = 1e-8;
  param->rtol            = 1e-6;
  param->print_level     = 1;
  param->max_level       = 6;
  PASE_INT  n            = 200;
  GetCommandLineInfo(argc, argv, &n, &(param->block_size), &(param->atol), &(param->max_pre_iter));
  param->min_coarse_size = param->block_size * 30;
  //param->min_coarse_size = 500;
  PASE_Printf(MPI_COMM_WORLD, "The dimension of the eigenvalue problem = %d\n", n*n);
  PrintParameter(param);

  //Create
  HYPRE_IJMatrix A, B;
  HYPRE_ParCSRMatrix parcsr_A, parcsr_B;
  GetEigenProblem(&A, &B, n);
  HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
  HYPRE_IJMatrixGetObject(B, (void**) &parcsr_B);

  PASE_SCALAR     *eval         = PASE_Malloc(param->block_size*sizeof(PASE_SCALAR));
  HYPRE_ParVector *evec         = (HYPRE_ParVector*) PASE_Malloc(param->block_size*sizeof(HYPRE_ParVector));
  PASE_INT         i            = 0;
  MPI_Comm         comm         = hypre_ParCSRMatrixComm(parcsr_A);
  PASE_INT         global_size  = hypre_ParCSRMatrixGlobalNumRows(parcsr_A);
  PASE_INT        *partitioning = NULL;
  HYPRE_ParCSRMatrixGetRowPartitioning(parcsr_A,  &partitioning);
  for(i = 0; i < block_size; ++i) {
    evec[i]= hypre_ParVectorCreate(comm,  global_size,  partitioning);
    HYPRE_ParVectorInitialize(evec[i]);
    hypre_ParVectorSetPartitioningOwner(evec[i], 0); 
  }
  hypre_ParVectorSetPartitioningOwner(evec[0], 1); 

  //Solve
  PASE_EigenSolver_default((void*)parcsr_A, (void*)parcsr_B, eval, evec, param->block_size, param);

  //Destroy
  for(i = 0; i < param->block_size; ++i) {
    HYPRE_ParVectorDestroy(evec[i]);
  }
  PASE_Free(evec);
  PASE_Free(eval);
  PASE_Free(param);
  PASE_Matrix_destroy(pase_A);
  PASE_Matrix_destroy(pase_B);
  HYPRE_IJMatrixDestroy(A);
  HYPRE_IJMatrixDestroy(B);

  MPI_Finalize();
  return(0);
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

void GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n, PASE_INT *block_size, PASE_REAL *atol, PASE_INT *nsmooth)
{
  PASE_INT arg_index = 0;
  PASE_INT print_usage = 0;
  PASE_INT myid;

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  while (arg_index < argc)
  {
    if ( strcmp(argv[arg_index], "-n") == 0 )
    {
      arg_index++;
      *n = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-block_size") == 0 )
    {
      arg_index++;
      *block_size = atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-atol") == 0 )
    {
      arg_index++;
      *atol= pow(10, atoi(argv[arg_index++]));
    }
    else if ( strcmp(argv[arg_index], "-nsmooth") == 0 )
    {
      arg_index++;
      *nsmooth= atoi(argv[arg_index++]);
    }
    else if ( strcmp(argv[arg_index], "-help") == 0 )
    {
      print_usage = 1;
      break;
    }
    else
    {
      arg_index++;
    }
  }

  if(print_usage)
  {
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "Usage: %s [<options>]\n", argv[0]);
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "  -n <n>              : problem size in each direction (default: 33)\n");
    PASE_Printf(MPI_COMM_WORLD, "  -block_size <n>      : eigenproblem block size (default: 3)\n");
    PASE_Printf(MPI_COMM_WORLD, "  -max_levels <n>      : max levels of AMG (default: 5)\n");
    PASE_Printf(MPI_COMM_WORLD, "\n");
    exit(-1);
  }
}

void PrintParameter(PASE_PARAMETER param)
{
    PASE_Printf(MPI_COMM_WORLD, "PASE (Parallel Auxiliary Space Eigen-solver), parallel version\n"); 
    PASE_Printf(MPI_COMM_WORLD, "Please contact liyu@lsec.cc.ac.cn, if there is any bugs.\n"); 
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n" );
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "Set parameters:\n");
    PASE_Printf(MPI_COMM_WORLD, "block size      = %d\n", param->block_size);
    PASE_Printf(MPI_COMM_WORLD, "max pre iter    = %d\n", param->max_pre_iter);
    PASE_Printf(MPI_COMM_WORLD, "atol            = %e\n", param->atol);
    PASE_Printf(MPI_COMM_WORLD, "max cycle       = %d\n", param->max_cycle);
    PASE_Printf(MPI_COMM_WORLD, "min coarse size = %d\n", param->min_coarse_size);
    PASE_Printf(MPI_COMM_WORLD, "\n");
}

