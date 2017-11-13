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



static int cmp( const void *a ,  const void *b )
{   return *(double *)a > *(double *)b ? 1 : -1; }

int main (int argc, char *argv[])
{
   int i, k;
   int myid, num_procs;
   int N, n;
   int block_size, max_levels;

   int ilower, iupper;
   int local_size, extra;

   double h, h2;
//   int time_index; 
   int global_time_index;

   /* -------------------------矩阵向量声明---------------------- */ 
   /* 最细矩阵 */
   HYPRE_IJMatrix A;
   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_IJMatrix B;
   HYPRE_ParCSRMatrix parcsr_B;
   HYPRE_IJVector b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector x;
   HYPRE_ParVector par_x;

   HYPRE_Real *eigenvalues;
   HYPRE_Real *exact_eigenvalues;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   printf("=============================================================\n" );
   printf("PASE (Parallel Auxiliary Space Eigen-solver), parallel version\n"); 
   printf("Please contact liyu@lsec.cc.ac.cn, if there is any bugs.\n"); 
   printf("=============================================================\n" );

   global_time_index = hypre_InitializeTiming("PASE Solve");
   hypre_BeginTiming(global_time_index);


   /* Default problem parameters */
   n = 200;
   max_levels = 5;
   block_size = (int) n*n/pow(4, max_levels);
   printf ( "block_size = n*n/pow(4, max_levels)\n" );

   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

      while (arg_index < argc)
      {
	 if ( strcmp(argv[arg_index], "-n") == 0 )
	 {
	    arg_index++;
	    n = atoi(argv[arg_index++]);
	 }
	 else if ( strcmp(argv[arg_index], "-block_size") == 0 )
	 {
	    arg_index++;
	    block_size = atoi(argv[arg_index++]);
	 }
	 else if ( strcmp(argv[arg_index], "-max_levels") == 0 )
	 {
	    arg_index++;
	    max_levels = atoi(argv[arg_index++]);
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

      if ((print_usage) && (myid == 0))
      {
	 printf("\n");
	 printf("Usage: %s [<options>]\n", argv[0]);
	 printf("\n");
	 printf("  -n <n>              : problem size in each direction (default: 33)\n");
	 printf("  -block_size <n>      : eigenproblem block size (default: 3)\n");
	 printf("  -max_levels <n>      : max levels of AMG (default: 5)\n");
	 printf("\n");
      }

      if (print_usage)
      {
	 MPI_Finalize();
	 return (0);
      }
   }
   /* Preliminaries: want at least one processor per row */
   if (n*n < num_procs) n = sqrt(num_procs) + 1;
   N = n*n; /* global number of rows */
   h = 1.0/(n+1); /* mesh size*/
   h2 = h*h;

   /*----------------------- Laplace精确特征值 ---------------------*/
   /* eigenvalues - allocate space */
   eigenvalues = hypre_CTAlloc (HYPRE_Real, block_size);
   {
      int tmp_nn = (int) sqrt(block_size) + 2;
      exact_eigenvalues = hypre_CTAlloc (HYPRE_Real, tmp_nn*tmp_nn);
      for (i = 0; i < tmp_nn; ++i) 
      {
	 for (k = 0; k < tmp_nn; ++k) 
	 {
	    exact_eigenvalues[i*tmp_nn+k] = h2*M_PI*M_PI*(pow(i+1, 2)+pow(k+1, 2));
	 }
      }
      qsort(exact_eigenvalues, tmp_nn*tmp_nn, sizeof(double), cmp);
   }




   /* Each processor knows only of its own rows - the range is denoted by ilower
      and iupper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   local_size = N/num_procs;
   extra = N - local_size*num_procs;

   ilower = local_size*myid;
   ilower += hypre_min(myid, extra);

   iupper = local_size*(myid+1);
   iupper += hypre_min(myid+1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;


   /* -------------------最细矩阵赋值------------------------ */

   /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &B);

   /* Choose a parallel csr format storage (see the User's Manual) */
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_IJMatrixSetObjectType(B, HYPRE_PARCSR);

   /* Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize(A);
   HYPRE_IJMatrixInitialize(B);

   /* Now go through my local rows and set the matrix entries.
      Each row has at most 5 entries. For example, if n=3:

      A = [M -I 0; -I M -I; 0 -I M]
      M = [4 -1 0; -1 4 -1; 0 -1 4]

      Note that here we are setting one row at a time, though
      one could set all the rows together (see the User's Manual).
      */
   {
      int nnz;
      double values[5];
      int cols[5];

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
	 if ((i+n)< N)
	 {
	    cols[nnz] = i+n;
	    values[nnz] = -1.0;
	    nnz++;
	 }

	 /* Set the values for row i */
	 HYPRE_IJMatrixSetValues(A, 1, &nnz, &i, cols, values);
      }
   }
   {
      int nnz;
      double values[5];
      int cols[5];
      for (i = ilower; i <= iupper; i++)
      {
	 nnz = 1;
	 cols[0] = i;
	 values[0] = 1.0;
	 /* Set the values for row i */
	 HYPRE_IJMatrixSetValues(B, 1, &nnz, &i, cols, values);
      }
   }
   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(A);
   HYPRE_IJMatrixAssemble(B);
   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
   HYPRE_IJMatrixGetObject(B, (void**) &parcsr_B);

   /* Create sample rhs and solution vectors */
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorGetObject(b, (void **) &par_b);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);
   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject(x, (void **) &par_x);

   /*求解特征值问题的MG方法*/
   {
       PASE_MATRIX pase_A = PASE_Matrix_create_default((void*)parcsr_A, 1);
       PASE_MATRIX pase_B = PASE_Matrix_create_default((void*)parcsr_B, 1);
       PASE_VECTOR pase_x = PASE_Vector_create_default((void*)par_x, 1);
       PASE_PARAMETER param = (PASE_PARAMETER) PASE_Malloc(sizeof(PASE_PARAMETER_PRIVATE));
       param->max_level = 3;
       PASE_MULTIGRID multigrid = PASE_Multigrid_create(pase_A, pase_B, param, NULL);
       PASE_MG_SOLVER solver = PASE_Mg_solver_create_by_multigrid(multigrid);
       PASE_Mg_set_up(solver);
       PASE_Mg_solve(solver);
       PASE_Mg_solver_destroy(solver);
       PASE_Free(param);
       PASE_Multigrid_destroy(multigrid);
       PASE_Matrix_destroy(pase_A);
       PASE_Matrix_destroy(pase_B);
       PASE_Vector_destroy(pase_x);
   }

   /* Clean up */
   hypre_TFree(eigenvalues);
   hypre_TFree(exact_eigenvalues);


   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJMatrixDestroy(B);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   hypre_EndTiming(global_time_index);
   hypre_PrintTiming("Solve phase times", MPI_COMM_WORLD);
   hypre_FinalizeTiming(global_time_index);
   hypre_ClearTiming();

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
