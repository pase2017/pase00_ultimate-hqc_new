/*
   parPASE_ver01

Interface:    Linear-Algebraic (IJ)

Compile with: make parPASE_ver01

Sample run:   parPASE_ver01 -block_size 20 -n 100 -max_levels 6 

Description:  This example solves the 2-D Laplacian eigenvalue
problem with zero boundary conditions on an nxn grid.
The number of unknowns is N=n^2. The standard 5-point
stencil is used, and we solve for the interior nodes
only.

We use the same matrix as in Examples 3 and 5.
The eigensolver is PASE (Parallels Auxiliary Space Eigen-solver)
with LOBPCG and AMG preconditioner.

Created:      2017.09.16

Author:       Li Yu (liyu@lsec.cc.ac.cn).
*/

#include "pase_mg_solver.h"
#include "pase_multigrid.h"
#include "pase_config.h"

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

static PASE_INT cmp( const void *a ,  const void *b );
void GetEigenProblem(HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, PASE_INT n);
void GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n, PASE_INT *block_size, PASE_REAL *atol, PASE_INT *max_pre_iter, PASE_INT *max_post_iter, PASE_INT *max_direct_iter, PASE_INT *max_level);
void GetExactEigenvalues(PASE_REAL **exact_eigenvalues, PASE_INT n, PASE_INT block_size);

PASE_INT main(PASE_INT argc, char *argv[])
{
  PASE_INT  myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  PASE_INT  n             = 200;
  PASE_INT  cycle_type    = 0;
  PASE_INT  block_size    = 5;
  PASE_INT  max_cycle     = 100;
  PASE_INT  max_pre_iter  = 0;
  PASE_INT  max_post_iter = 2;
  PASE_INT  max_direct_iter = 3;
  PASE_REAL atol          = 1e-8;
  PASE_REAL rtol          = 1e-6;
  PASE_INT  print_level   = 1;
  PASE_INT  max_level     = 20;
  GetCommandLineInfo(argc, argv, &n, &block_size, &atol, &max_pre_iter, &max_post_iter, &max_direct_iter, &max_level);
  //PASE_INT  min_coarse_size = block_size * 30;
  PASE_INT  min_coarse_size = 4000;
  PASE_INT  max_block_size= ((2*block_size)<(block_size+5))?(2*block_size):(block_size+5);

  HYPRE_IJMatrix A, B;
  HYPRE_ParCSRMatrix parcsr_A, parcsr_B;
  GetEigenProblem(&A, &B, n);
  HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
  HYPRE_IJMatrixGetObject(B, (void**) &parcsr_B);

  PASE_REAL *eigenvalues;
  eigenvalues = PASE_Malloc(block_size*sizeof(PASE_REAL));
  //PASE_REAL *exact_eigenvalues;
  //GetExactEigenvalues(&exact_eigenvalues, n, block_size);

  PASE_Printf(MPI_COMM_WORLD, "PASE (Parallel Auxiliary Space Eigen-solver), parallel version\n"); 
  PASE_Printf(MPI_COMM_WORLD, "Please contact liyu@lsec.cc.ac.cn, if there is any bugs.\n"); 
  PASE_Printf(MPI_COMM_WORLD, "=============================================================\n" );
  PASE_Printf(MPI_COMM_WORLD, "\n");
  PASE_Printf(MPI_COMM_WORLD, "Set parameters:\n");
  PASE_Printf(MPI_COMM_WORLD, "dimension       = %d\n", n*n);
  PASE_Printf(MPI_COMM_WORLD, "cycle type      = %d\n", cycle_type);
  PASE_Printf(MPI_COMM_WORLD, "block size      = %d\n", block_size);
  PASE_Printf(MPI_COMM_WORLD, "max block size  = %d\n", max_block_size);
  PASE_Printf(MPI_COMM_WORLD, "max pre iter    = %d\n", max_pre_iter);
  PASE_Printf(MPI_COMM_WORLD, "max post iter   = %d\n", max_post_iter);
  PASE_Printf(MPI_COMM_WORLD, "max direct iter = %d\n", max_direct_iter);
  PASE_Printf(MPI_COMM_WORLD, "atol            = %e\n", atol);
  PASE_Printf(MPI_COMM_WORLD, "max cycle       = %d\n", max_cycle);
  PASE_Printf(MPI_COMM_WORLD, "max level       = %d\n", max_level);
  PASE_Printf(MPI_COMM_WORLD, "min coarse size = %d\n", min_coarse_size);
  PASE_Printf(MPI_COMM_WORLD, "\n");

  //Create
  PASE_MATRIX pase_A       = PASE_Matrix_create((void*)parcsr_A, 1);
  PASE_MATRIX pase_B       = PASE_Matrix_create((void*)parcsr_B, 1);
  PASE_VECTOR pase_x       = PASE_Vector_create_by_matrix_and_vector_data_operator(pase_A, NULL);

  PASE_PARAMETER param     = (PASE_PARAMETER) PASE_Malloc(sizeof(PASE_PARAMETER_PRIVATE));
  param->max_level         = max_level;
  param->min_coarse_size   = min_coarse_size;;
  PASE_MG_SOLVER solver    = PASE_Mg_solver_create(pase_A, pase_B, param);

  //Set up
  PASE_Mg_set_cycle_type(solver, cycle_type);
  PASE_Mg_set_block_size(solver, block_size);
  PASE_Mg_set_max_block_size(solver, max_block_size);
  PASE_Mg_set_max_cycle(solver, max_cycle);
  PASE_Mg_set_max_pre_iteration(solver, max_pre_iter);
  PASE_Mg_set_max_post_iteration(solver, max_post_iter);
  PASE_Mg_set_max_direct_iteration(solver, max_direct_iter);
  PASE_Mg_set_atol(solver, atol);
  PASE_Mg_set_rtol(solver, rtol);
  PASE_Mg_set_print_level(solver, print_level);
  //PASE_Mg_set_exact_eigenvalues(solver, exact_eigenvalues);
  PASE_Mg_set_up(solver, pase_A, pase_B, pase_x, param);

  //Solve
  PASE_Mg_solve(solver);

  //Destroy
  PASE_Mg_solver_destroy(solver);
  PASE_Free(param);
  PASE_Matrix_destroy(pase_A);
  PASE_Matrix_destroy(pase_B);
  PASE_Vector_destroy(pase_x);
  PASE_Free(eigenvalues);
  //PASE_Free(exact_eigenvalues);
  HYPRE_IJMatrixDestroy(A);
  HYPRE_IJMatrixDestroy(B);

  MPI_Finalize();
  return(0);
}

static PASE_INT cmp( const void *a ,  const void *b )
{   
  return *(PASE_REAL *)a > *(PASE_REAL *)b ? 1 : -1; 
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

void GetExactEigenvalues(PASE_REAL **exact_eigenvalues, PASE_INT n, PASE_INT block_size)
{
  PASE_INT i, k;
  PASE_REAL h        = 1.0/(n+1); /* mesh size*/
  PASE_REAL h2       = h*h;
  PASE_INT tmp_nn    = (PASE_INT) sqrt(block_size) + 2;
  *exact_eigenvalues = PASE_Malloc(sizeof(PASE_REAL)*tmp_nn*tmp_nn);
  for(i = 0; i < tmp_nn; ++i) {
    for(k = 0; k < tmp_nn; ++k) {
      (*exact_eigenvalues)[i*tmp_nn+k] = h2*M_PI*M_PI*(pow(i+1, 2)+pow(k+1, 2));
    }
  }
  qsort(*exact_eigenvalues, tmp_nn*tmp_nn, sizeof(PASE_REAL), cmp);
}

