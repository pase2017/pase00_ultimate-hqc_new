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
   int i, N;
   int myid, num_procs;

   int ilower, iupper;
   int local_size, extra;

   int global_time_index;

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

   /* Each processor knows only of its own rows - the range is denoted by ilower
      and iupper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   N = 2;
   local_size = N/num_procs;
   extra = N - local_size*num_procs;

   ilower = local_size*myid;
   ilower += hypre_min(myid, extra);

   iupper = local_size*(myid+1);
   iupper += hypre_min(myid+1, extra);
   iupper = iupper - 1;

   /* How many rows do I have? */
   local_size = iupper - ilower + 1;

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
         b_values[i] = 2.0;
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
    
   //PASE VECTOR 测试
   {
       //Create
       PASE_VECTOR pase_x = PASE_Vector_create_default((void*)par_x, 1);
       PASE_VECTOR pase_b = PASE_Vector_create_default((void*)par_b, 1);
   printf("=============================================================\n" );
       printf("This is a test program for PASE_VECTOR\n");
   printf("=============================================================\n" );
       printf("\n");
       printf("Initial\n");
       printf("x = [%f, %f].\n", par_x->local_vector->data[0], par_x->local_vector->data[1]);
       printf("b = [%f, %f].\n", par_b->local_vector->data[0], par_b->local_vector->data[1]);
       printf("\n");

       //Copy b = x
       printf("Copy x to b\n");
       PASE_Vector_copy(pase_x, pase_b);
       printf("b = [%f, %f].\n", par_b->local_vector->data[0], par_b->local_vector->data[1]);
       printf("\n");
       //Set constant value
       PASE_REAL constant = 3.5;
       printf("Set constant value %f to x\n", constant);
       PASE_Vector_set_constant_value( pase_x, constant);
       printf("x = [%f, %f].\n", par_x->local_vector->data[0], par_x->local_vector->data[1]);
       printf("\n");
       //Set random value
       PASE_INT seed = 1;
       printf("Set random value to b with seed = %d\n", seed);
       PASE_Vector_set_random_value( pase_b, 1);
       printf("b = [%f, %f].\n", par_b->local_vector->data[0], par_b->local_vector->data[1]);
       printf("\n");
       //Inner product
       printf("Inner product\n");
       PASE_REAL prod;
       PASE_Vector_inner_product(pase_x, pase_b, &prod);
       printf("(x,b) = %f.\n", prod);
       printf("\n");
       //Add b = x + b
       printf("Add b = x + b\n");
       printf("b = [%f, %f].\n", par_b->local_vector->data[0], par_b->local_vector->data[1]);
       printf("\n");
       //Scale b = a * b 
       PASE_REAL scale = 10.0;
       printf("Scale b = %1.2f * b\n", scale);
       PASE_Vector_scale( scale, pase_b);
       printf("b = [%f, %f].\n", par_b->local_vector->data[0], par_b->local_vector->data[1]);
       printf("\n");
       //Destroy
       PASE_Vector_destroy(pase_x);
       PASE_Vector_destroy(pase_b);
   }



   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);
   
   hypre_EndTiming(global_time_index);
   hypre_FinalizeTiming(global_time_index);
   hypre_ClearTiming();

   /* Finalize MPI*/
   MPI_Finalize();

   return(0);
}
