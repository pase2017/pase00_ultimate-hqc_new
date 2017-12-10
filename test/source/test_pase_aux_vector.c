/*
   mv

   Compile with: make mv

   Test run:   ./mv -n 10 

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

#include <unistd.h>
#include <math.h>
#include "pase_vector.h"
#include "pase_aux_vector.h"
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

void GetTwoTestHypreParVector(HYPRE_IJVector *x, HYPRE_IJVector *y, PASE_INT n);        //默认设置两个测试 HYPRE_IJVector x = [1,...,1] 和 y = [2,...,2].
void GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n, PASE_INT *block_size); //可以从命令行中, 得到向量长度 n
void PrintPaseAuxVector(PASE_AUX_VECTOR aux_x, char *vector_name);                      //打印 HYPRE_ParVector

PASE_INT main(PASE_INT argc, char *argv[])
{
  PASE_INT myid, num_procs;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  PASE_INT n          = 3; //向量长度
  PASE_INT block_size = 1; //辅助空间维数
  GetCommandLineInfo(argc, argv, &n, &block_size);

  HYPRE_IJVector x, y;
  HYPRE_ParVector par_x, par_y;
  GetTwoTestHypreParVector(&x, &y, n);
  HYPRE_IJVectorGetObject(x, (void **) &par_x);
  HYPRE_IJVectorGetObject(y, (void **) &par_y);
  
  //PASE AUX VECTOR 测试
  {
    PASE_Printf(PASE_COMM_WORLD, "=============================================================\n");
    PASE_Printf(PASE_COMM_WORLD, "This is a test program for PASE_AUX_VECTOR\n");
    PASE_Printf(PASE_COMM_WORLD, "=============================================================\n\n");

    // create
    PASE_VECTOR pase_x = PASE_Vector_create((void*)par_x, PACKAGE_HYPRE);
    PASE_VECTOR pase_y = PASE_Vector_create((void*)par_y, PACKAGE_HYPRE);
    PASE_SCALAR *block_x = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
    PASE_SCALAR *block_y = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
    memset(block_x, 0, block_size*sizeof(PASE_SCALAR));
    memset(block_y, 0, block_size*sizeof(PASE_SCALAR));
    PASE_AUX_VECTOR aux_x = PASE_Aux_vector_create(pase_x, block_x, block_size);
    PASE_AUX_VECTOR aux_y = PASE_Aux_vector_create(pase_y, block_y, block_size);
    PASE_AUX_VECTOR aux_z = PASE_Aux_vector_create_by_aux_vector(aux_x);

    // initial value
    PASE_Printf(PASE_COMM_WORLD, "Initial value\n\n");
    PrintPaseAuxVector(aux_x, "x");
    PrintPaseAuxVector(aux_y, "y");
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");
    
    // copy
    PASE_Printf(PASE_COMM_WORLD, "Copy x to z\n\n");
    PASE_Aux_vector_copy(aux_x, aux_z);
    PrintPaseAuxVector(aux_z, "z");
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");
    
    // set constant value
    PASE_REAL constant = 3.5;
    PASE_Printf(PASE_COMM_WORLD, "Set constant value %f to x\n\n", constant);
    PASE_Aux_vector_set_constant_value( aux_x, constant);
    PrintPaseAuxVector(aux_x, "x");
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");
    
    // set random value
    PASE_INT seed = 1;
    PASE_Printf(PASE_COMM_WORLD, "Set random value to z with seed = %d\n\n", seed);
    PASE_Aux_vector_set_random_value(aux_z, seed);
    PrintPaseAuxVector(aux_z, "z");
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");
    
    // axpy
    PASE_REAL alpha = 2.0;
    PASE_Printf(PASE_COMM_WORLD, "Perform y = %f * x + y\n\n", alpha);
    PASE_Aux_vector_axpy(alpha, aux_x, aux_y);
    PrintPaseAuxVector(aux_y, "y");
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");

    // scale
    PASE_REAL scale = 10.0;
    PASE_Printf(PASE_COMM_WORLD, "Scale y = %1.2f * y\n\n", scale);
    PASE_Aux_vector_scale( scale, aux_y);
    PrintPaseAuxVector(aux_y, "y");
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");

    // orthogonalization 
    PASE_AUX_VECTOR *aux_X = (PASE_AUX_VECTOR*)PASE_Malloc(3*sizeof(PASE_AUX_VECTOR));
    aux_X[0] = aux_x;
    aux_X[1] = aux_y;
    aux_X[2] = aux_z;
    PASE_Printf(PASE_COMM_WORLD, "orthogonalize [x, y, z]\n\n");
    PASE_Aux_vector_orthogonalize_all(aux_X, 3); //这个函数里调用了 PASE_Vector_orthogonalize
    PrintPaseAuxVector(aux_x, "x");
    PrintPaseAuxVector(aux_y, "y");
    PrintPaseAuxVector(aux_z, "z");
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");

    // inner product
    PASE_REAL **prods = (PASE_REAL**)PASE_Malloc(3*sizeof(PASE_REAL*));
    prods[0] = (PASE_REAL*)PASE_Malloc(3*sizeof(PASE_REAL));
    prods[1] = (PASE_REAL*)PASE_Malloc(3*sizeof(PASE_REAL));
    prods[2] = (PASE_REAL*)PASE_Malloc(3*sizeof(PASE_REAL));
    PASE_Printf(PASE_COMM_WORLD, "Inner product\n\n");
    PASE_Aux_vector_inner_product_some(aux_X, 0, 2, prods); //这个函数里调用了 PASE_Vector_inner_product
    PASE_Printf(PASE_COMM_WORLD, "(x,x) = %.2f, (x,y) = %.2f, (x,z) = %.2f, (y,y) = %.2f, (y,z) = %.2f, (z,z) = %.2f\n", prods[0][0], prods[0][1], prods[0][2], prods[1][1], prods[1][2], prods[2][2]);
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");

    // destroy
    PASE_Vector_destroy(pase_x);
    PASE_Vector_destroy(pase_y);
    PASE_Aux_vector_destroy(aux_x);
    PASE_Aux_vector_destroy(aux_y);
    PASE_Aux_vector_destroy(aux_z);
    PASE_Free(prods[0]); PASE_Free(prods[1]); PASE_Free(prods[2]);
    PASE_Free(prods);
    PASE_Free(aux_X);
  }

  HYPRE_IJVectorDestroy(x);
  HYPRE_IJVectorDestroy(y);
  MPI_Finalize();
  return(0);
}

void GetTwoTestHypreParVector(HYPRE_IJVector *x, HYPRE_IJVector *y, PASE_INT n)
{
  PASE_INT ilower, iupper, local_size, extra, i;
  PASE_SCALAR *x_values;
  PASE_SCALAR *y_values;
  PASE_INT *rows;
  PASE_INT num_procs, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  local_size = n/num_procs;
  extra      = n - local_size*num_procs;
  ilower     = local_size*myid;
  ilower    += hypre_min(myid, extra);
  iupper     = local_size*(myid+1);
  iupper    += hypre_min(myid+1, extra);
  iupper     = iupper - 1;
  local_size = iupper - ilower + 1;

  HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,x);
  HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper,y);
  HYPRE_IJVectorSetObjectType(*x, HYPRE_PARCSR);
  HYPRE_IJVectorSetObjectType(*y, HYPRE_PARCSR);
  HYPRE_IJVectorInitialize(*x);
  HYPRE_IJVectorInitialize(*y);
 
  x_values = (PASE_SCALAR*)calloc(local_size, sizeof(PASE_SCALAR));
  y_values = (PASE_SCALAR*)calloc(local_size, sizeof(PASE_SCALAR));
  rows     = (PASE_INT*)   calloc(local_size, sizeof(PASE_INT));

  for (i=0; i<local_size; i++)
  {
    x_values[i] = 1.0;
    y_values[i] = (double)i;
    rows[i] = ilower + i;
  }
  
  HYPRE_IJVectorSetValues(*x, local_size, rows, x_values);
  HYPRE_IJVectorSetValues(*y, local_size, rows, y_values);
  
  free(x_values);
  free(y_values);
  free(rows);

  HYPRE_IJVectorAssemble(*x);
  HYPRE_IJVectorAssemble(*y);
}

void GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n, PASE_INT *block_size)
{
  PASE_INT arg_index = 0;
  PASE_INT print_usage = 0;
  PASE_INT myid;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  while(arg_index < argc) {
    if(0 == strcmp(argv[arg_index], "-n")) {
      arg_index++;
      *n = atoi(argv[arg_index++]);
    } else if(0 == strcmp(argv[arg_index], "-block_size")) {
      arg_index++;
      *block_size = atoi(argv[arg_index++]);
    } else if(0 == strcmp(argv[arg_index], "-help")) {
       print_usage = 1;
      break;
    } else {
      arg_index++;
    }
  }
  
  if((1 == print_usage) && (myid == 0)) {
    printf("\n");
    printf("Usage: %s [<options>]\n", argv[0]);
    printf("\n");
    printf("  -n <n>  : length of vector\n");
    printf("\n");
  }
  
  if(print_usage) {
    exit(-1);
  }
}

void PrintPaseAuxVector(PASE_AUX_VECTOR aux_x, char *vector_name)
{
  PASE_INT myid, np, idx_proc, local_size, i;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  PASE_VECTOR      pase_x = aux_x->vec;
  PASE_SCALAR     *block  = aux_x->block;
  HYPRE_ParVector  par_x  = (HYPRE_ParVector)pase_x->vector_data;
  local_size = par_x->local_vector->size;
  if(NULL != vector_name) {
    PASE_Printf(MPI_COMM_WORLD, "%s = ", vector_name);
  }
  PASE_Printf(MPI_COMM_WORLD, "[");

  for(idx_proc = 0; idx_proc < np; idx_proc++) {
    if(myid == idx_proc) {
      for(i = 0; i<local_size; i++) {
        printf("\t%.2f", par_x->local_vector->data[i]);
        fflush(stdout);
      }
    } else {
	usleep(50);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  PASE_Printf(MPI_COMM_WORLD, "\t|");
  for(i = 0; i<aux_x->block_size; ++i) {
    PASE_Printf(MPI_COMM_WORLD, "\t%.2f", block[i]);
  }
  PASE_Printf(MPI_COMM_WORLD, "\t]^T\n");
}
