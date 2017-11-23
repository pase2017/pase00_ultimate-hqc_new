/*
   mv

   Compile with: make mv

   Sample run:   ./mv -n 10 

   Description:  This example solves the 2-D Laplacian eigenvalue
                 problem with zero boundary conditions on an nxn grid.
                 The number of unknowns is N=n^2. The standard 5-point
                 stencil is used, and we solve for the interior nodes
                 only.

                 We use the same matrix as in Examples 3 and 5.
                 The eigensolver is PASE (Parallels Auxiliary Space Eigen-solver)
                 with LOBPCG and AMG preconditioner.
   
   Created:      2017.08.26

   Author:       Li Yu (liyu@lsec.cc.ac.cn).
*/

#include "pase_matrix.h"
#include "pase_vector.h"
#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_seq_mv.h"
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"
#include "temp_multivector.h"
#include "HYPRE_utilities.h"


int main (int argc, char *argv[])
{
   int i;
   int myid, num_procs;
   int N, n;

   int ilower, iupper;
   int local_size, extra;

   int global_time_index;

   /* -------------------------矩阵向量声明---------------------- */ 
   /* 最细矩阵 */
   HYPRE_IJMatrix A, B;
   HYPRE_ParCSRMatrix parcsr_A, parcsr_B;
   HYPRE_IJVector b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector x;
   HYPRE_ParVector par_x;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   global_time_index = hypre_InitializeTiming("PASE Solve");
   hypre_BeginTiming(global_time_index);

   n = 2;
   N = 2; /* global number of rows */

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
   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,&x);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);
   HYPRE_IJVectorInitialize(x);
   {
      double *x_values;
      double *b_values;
      int    *rows;

      x_values =  (double*) calloc(local_size, sizeof(double));
      b_values =  (double*) calloc(local_size, sizeof(double));
      rows = (int*) calloc(local_size, sizeof(int));

      for (i=0; i<local_size; i++)
      {
         x_values[i] = 1.0;
         b_values[i] = 1.0;
         rows[i] = ilower + i;
      }

      HYPRE_IJVectorSetValues(b, local_size, rows, b_values);
      HYPRE_IJVectorSetValues(x, local_size, rows, x_values);

      free(b_values);
      free(x_values);
      free(rows);
   }
   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorAssemble(b);
   HYPRE_IJVectorGetObject(x, (void **) &par_x);
   HYPRE_IJVectorGetObject(b, (void **) &par_b);

   //PASE MATRIX 测试
   {
       //Create
       PASE_MATRIX pase_A = PASE_Matrix_create_default((void*)parcsr_A, 1); 
       PASE_MATRIX pase_B = PASE_Matrix_create_default((void*)parcsr_B, 1); 
       PASE_VECTOR pase_x = PASE_Vector_create_default((void*)par_x, 1);
       PASE_VECTOR pase_b = PASE_Vector_create_default((void*)par_b, 1);
       //HYPRE_ParCSRMatrixPrint(parcsr_A, "A");
       printf("=============================================================\n" );
       printf("This is a test program for PASE_MATRIX\n");
       printf("=============================================================\n" );
       printf("\n");
       printf("Initial\n");
       printf("A = [%f, %f], B = [%f, %f]\n", parcsr_A->diag->data[0], parcsr_A->diag->data[1], parcsr_B->diag->data[0], parcsr_B->diag->data[1]);
       printf("    [%f, %f]      [%f, %f]\n", parcsr_A->diag->data[3], parcsr_A->diag->data[2], parcsr_B->diag->data[3], parcsr_B->diag->data[2]);
       printf("x = [%f, %f], b = [%f, %f].\n", par_x->local_vector->data[0], par_x->local_vector->data[1], par_b->local_vector->data[0], par_b->local_vector->data[1]);
       printf("\n");

       //Multiply two matrices
       printf("Multiply C = A * B\n");
       PASE_MATRIX pase_C = PASE_Matrix_multiply_matrix(pase_A, pase_B);
       HYPRE_ParCSRMatrix parcsr_C = (HYPRE_ParCSRMatrix)pase_C->matrix_data;
       printf("C = [%f, %f]\n", parcsr_C->diag->data[0], parcsr_C->diag->data[1]);
       printf("    [%f, %f]\n", parcsr_C->diag->data[3], parcsr_C->diag->data[2]);
       printf("\n");

       //Copy matrix
       printf("Copy A = C\n");
       PASE_Matrix_copy(pase_C, pase_A);
       printf("A = [%f, %f]\n", parcsr_A->diag->data[0], parcsr_A->diag->data[1]);
       printf("    [%f, %f]\n", parcsr_A->diag->data[3], parcsr_A->diag->data[2]);
       printf("\n");

       //Multiply matrix and vector
       printf("Multiply b = A * x\n");
       PASE_Matrix_multiply_vector(pase_A, pase_x, pase_b);
       printf("b = [%f, %f].\n", par_b->local_vector->data[0], par_b->local_vector->data[1]);
       printf("\n");

       //Destroy
       PASE_Matrix_destroy(pase_A);
       PASE_Matrix_destroy(pase_B);
       PASE_Matrix_destroy(pase_C);
       PASE_Vector_destroy(pase_x);
       PASE_Vector_destroy(pase_b);
   }

   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJMatrixDestroy(B);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);
   
   hypre_EndTiming(global_time_index);
   hypre_FinalizeTiming(global_time_index);
   hypre_ClearTiming();


   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
