#include "pase_matrix.h"
#include "pase_vector.h"
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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


void GetTwoTestHypreParCSRMatrix(HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, PASE_INT n);
void GetTwoTestHypreParVector(HYPRE_IJVector *x, HYPRE_IJVector *y, PASE_INT n);
void GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n);
void PrintPaseMatrix(PASE_MATRIX pase_A, char *matrix_name);
void PrintPaseVector(PASE_VECTOR pase_x, char *vector_name);

PASE_INT 
main (PASE_INT argc, char *argv[])
{
  PASE_INT myid, num_procs;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  PASE_INT n = 4;
  GetCommandLineInfo(argc, argv, &n);

  HYPRE_IJMatrix A, B;
  HYPRE_ParCSRMatrix parcsr_A, parcsr_B;
  GetTwoTestHypreParCSRMatrix(&A, &B, n);
  HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
  HYPRE_IJMatrixGetObject(B, (void**) &parcsr_B);

  HYPRE_IJVector x, y;
  HYPRE_ParVector par_x, par_y;
  GetTwoTestHypreParVector(&x, &y, n);
  HYPRE_IJVectorGetObject(x, (void **) &par_x);
  HYPRE_IJVectorGetObject(y, (void **) &par_y);

  //PASE MATRIX 测试
  {
    //Create
    PASE_MATRIX pase_A = PASE_Matrix_create((void*)parcsr_A, DATA_FORM_HYPRE); 
    PASE_MATRIX pase_B = PASE_Matrix_create((void*)parcsr_B, DATA_FORM_HYPRE); 
    PASE_VECTOR pase_x = PASE_Vector_create((void*)par_x, DATA_FORM_HYPRE);
    PASE_VECTOR pase_y = PASE_Vector_create((void*)par_y, DATA_FORM_HYPRE);
    PASE_VECTOR pase_z = PASE_Vector_create_by_matrix_and_vector_data_operator(pase_A, pase_x->ops);
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n");
    PASE_Printf(MPI_COMM_WORLD, "This is a test program for PASE_MATRIX\n");
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n");
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "Initial\n");
    PrintPaseMatrix(pase_A, "A");
    PrintPaseMatrix(pase_B, "B");
    PrintPaseVector(pase_x, "x");
    PrintPaseVector(pase_y, "y");
    PASE_Printf(MPI_COMM_WORLD, "-------------------------------------------------------------\n\n");

    //Get matrix global numbers of rows and cols
    PASE_INT nrow, ncol;
    PASE_Matrix_get_global_nrow(pase_A, &nrow);
    PASE_Matrix_get_global_ncol(pase_A, &ncol);
    PASE_Printf(MPI_COMM_WORLD, "Get the information of A: nrow = %d, ncol = %d\n", nrow, ncol);
    PASE_Printf(MPI_COMM_WORLD, "-------------------------------------------------------------\n\n");

    //Transpose matrix
    PASE_Printf(MPI_COMM_WORLD, "Transpose AT = A^T\n\n");
    PASE_MATRIX pase_AT = PASE_Matrix_transpose(pase_A);
    PrintPaseMatrix(pase_AT, "AT");
    PASE_Printf(MPI_COMM_WORLD, "-------------------------------------------------------------\n\n");

    //Multiply two matrices
    PASE_Printf(MPI_COMM_WORLD, "Multiply C = AT * B\n\n");
    PASE_MATRIX pase_C = PASE_Matrix_multiply_matrix(pase_AT, pase_B);
    PrintPaseMatrix(pase_C, "C");
    PASE_Printf(MPI_COMM_WORLD, "-------------------------------------------------------------\n\n");

    //Multiply matrixT and matrix
    PASE_Printf(MPI_COMM_WORLD, "Multiply D = A^T * B\n\n");
    PASE_MATRIX pase_D = PASE_MatrixT_multiply_matrix(pase_A, pase_B);
    PrintPaseMatrix(pase_D, "D");
    PASE_Printf(MPI_COMM_WORLD, "-------------------------------------------------------------\n\n");
     
    //Multiply matrix and vector
    PASE_Printf(MPI_COMM_WORLD, "Multiply y = AT * x\n\n");
    PASE_Matrix_multiply_vector(pase_AT, pase_x, pase_y);
    PrintPaseVector(pase_y, "y");
    PASE_SCALAR a = 1.0;
    PASE_SCALAR b = 2.0;
    PASE_Printf(MPI_COMM_WORLD, "\nGeneral multiply y = %.4f * AT * x + %.4f * y\n\n", a, b);
    PASE_Matrix_multiply_vector_general(a, pase_AT, pase_x, b, pase_y);
    PrintPaseVector(pase_y, "y");
    PASE_Printf(MPI_COMM_WORLD, "-------------------------------------------------------------\n\n");

    //Multiply matrixT and vector
    PASE_Printf(MPI_COMM_WORLD, "Multiply z = A^T * x\n\n");
    PASE_MatrixT_multiply_vector(pase_A, pase_x, pase_z);
    PrintPaseVector(pase_z, "z");
    PASE_Printf(MPI_COMM_WORLD, "-------------------------------------------------------------\n\n");

    // orthogonalization 
    PASE_VECTOR *pase_X = (PASE_VECTOR*)PASE_Malloc(3*sizeof(PASE_VECTOR));
    PASE_Vector_set_random_value(pase_z, 1);
    PASE_Printf(PASE_COMM_WORLD, "set random values to z\n\n");
    PrintPaseVector(pase_z, "z");
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(PASE_COMM_WORLD, "B-orthogonalize [x, y, z]\n\n");
    pase_X[0] = pase_x;
    pase_X[1] = pase_y;
    pase_X[2] = pase_z;
    PASE_Vector_orthogonalize_general_all(pase_X, 3, pase_B); //这个函数里调用了 PASE_Vector_orthogonalize_general
    PrintPaseVector(pase_x, "x");
    PrintPaseVector(pase_y, "y");
    PrintPaseVector(pase_z, "z");
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");

    // inner product
    PASE_REAL **prods = (PASE_REAL**)PASE_Malloc(3*sizeof(PASE_REAL*));
    prods[0] = (PASE_REAL*)PASE_Malloc(3*sizeof(PASE_REAL));
    prods[1] = (PASE_REAL*)PASE_Malloc(3*sizeof(PASE_REAL));
    prods[2] = (PASE_REAL*)PASE_Malloc(3*sizeof(PASE_REAL));
    PASE_Printf(PASE_COMM_WORLD, "B-inner product\n\n");
    PASE_Vector_inner_product_general_some(pase_X, 0, 2, pase_B, prods); //这个函数里调用了 PASE_Vector_inner_product_general
    PASE_Printf(PASE_COMM_WORLD, "(x,x) = %.2f, (x,y) = %.2f, (x,z) = %.2f, (y,y) = %.2f, (y,z) = %.2f, (z,z) = %.2f\n", prods[0][0], prods[0][1], prods[0][2], prods[1][1], prods[1][2], prods[2][2]);
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");

    // destroy
    PASE_Matrix_destroy(pase_A);
    PASE_Matrix_destroy(pase_B);
    PASE_Matrix_destroy(pase_C);
    PASE_Matrix_destroy(pase_D);
    PASE_Matrix_destroy(pase_AT);
    PASE_Vector_destroy(pase_x);
    PASE_Vector_destroy(pase_y);
    PASE_Vector_destroy(pase_z);
    PASE_Free(prods[0]); PASE_Free(prods[1]); PASE_Free(prods[2]);
    PASE_Free(prods);
    PASE_Free(pase_X);
  }

  HYPRE_IJMatrixDestroy(A);
  HYPRE_IJMatrixDestroy(B);
  HYPRE_IJVectorDestroy(x);
  HYPRE_IJVectorDestroy(y);
  MPI_Finalize();
  return 0;
}

void 
GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n)
{
  PASE_INT arg_index = 0;
  PASE_INT print_usage = 0;
  PASE_INT myid;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  while(arg_index < argc) {
    if(0 == strcmp(argv[arg_index], "-n")) {
      arg_index++;
      *n = atoi(argv[arg_index++]);
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

void 
GetTwoTestHypreParCSRMatrix(HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, PASE_INT n)
{
  PASE_INT i;
  PASE_INT ilower, iupper;
  PASE_INT local_size, extra;
  PASE_INT myid, num_procs;

  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  local_size   = n / num_procs;
  extra        = n - local_size*num_procs;
  ilower       = local_size * myid;
  ilower      += hypre_min(myid, extra);
  iupper       = local_size * (myid+1);
  iupper      += hypre_min(myid+1, extra);
  iupper       = iupper - 1;
  local_size   = iupper - ilower + 1;

  HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, A);
  HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, B);
  HYPRE_IJMatrixSetObjectType(*A, HYPRE_PARCSR);
  HYPRE_IJMatrixSetObjectType(*B, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(*A);
  HYPRE_IJMatrixInitialize(*B);

  /* Now go through my local rows and set the matrix entries.
    Each row has at most 5 entries. For example, if n=3:

    A = [M -I 0; -I M -I; 0 -I M]
    M = [4 -1 0; -1 4 -1; 0 -1 4]

    Note that here we are setting one row at a time, though
    one could set all the rows together (see the User's Manual).
    */
    PASE_INT nnz;
    PASE_REAL values[5];
    PASE_INT cols[5];

    for(i = ilower; i <= iupper; i++) {
      nnz = 0;
      cols[nnz]     = i;
      values[nnz]   = 2.0;
      nnz++;
      if(i > 0) {
        cols[nnz]   = i-1;
        values[nnz] = (double)-i;
        nnz++;
      }
      if(i < n-1) {
        cols[nnz]   = i+1;
        values[nnz] = (double)i;  
        nnz++;
      }

      /* Set the values for row i */
      HYPRE_IJMatrixSetValues(*A, 1, &nnz, &i, cols, values);
    }
    for(i = ilower; i <= iupper; i++) {
      nnz = 0;
      cols[nnz]     = i;
      values[nnz]   = 2.0;
      nnz++;
      if(i > 0) {
        cols[nnz]   = i-1;
        values[nnz] = 1;
        nnz++;
      }
      if(i < n-1) {
        cols[nnz]   = i+1;
        values[nnz] = 1;  
        nnz++;
      }

       /* Set the values for row i */
       HYPRE_IJMatrixSetValues(*B, 1, &nnz, &i, cols, values);
    }

  HYPRE_IJMatrixAssemble(*A);
  HYPRE_IJMatrixAssemble(*B);
}


void 
GetTwoTestHypreParVector(HYPRE_IJVector *x, HYPRE_IJVector *y, PASE_INT n)
{
  PASE_INT     ilower, iupper, local_size, extra, i;
  PASE_SCALAR *x_values;
  PASE_SCALAR *y_values;
  PASE_INT    *rows;
  PASE_INT     num_procs, myid;
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

  for(i=0; i<local_size; i++) {
    x_values[i] = 1.0;
    y_values[i] = (double)i;
    rows[i]     = ilower + i;
  }
  
  HYPRE_IJVectorSetValues(*x, local_size, rows, x_values);
  HYPRE_IJVectorSetValues(*y, local_size, rows, y_values);
  
  free(x_values);
  free(y_values);
  free(rows);

  HYPRE_IJVectorAssemble(*x);
  HYPRE_IJVectorAssemble(*y);
}

void PrintPaseMatrix(PASE_MATRIX pase_A, char *matrix_name)
{
  PASE_INT myid, np, idx_proc, size_local_row, idx_col, idx_local_row, idx_local_col, flag_colnz;
  PASE_INT row_start_diag, row_end_diag, row_start_offd, row_end_offd; 
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  HYPRE_ParCSRMatrix A = (HYPRE_ParCSRMatrix)pase_A->matrix_data;
  size_local_row = A->last_row_index-A->first_row_index+1;

  if(NULL != matrix_name) {
    PASE_Printf(MPI_COMM_WORLD, "%s = \n", matrix_name);
  }
  for(idx_proc = 0; idx_proc < np; idx_proc++) {
    if(myid == idx_proc) {
      for(idx_local_row = 0; idx_local_row < size_local_row; idx_local_row++) {
	for(idx_col = 0; idx_col < A->global_num_cols; idx_col++) {
	  flag_colnz = 0;
	  row_start_diag = A->diag->i[idx_local_row];
	  row_end_diag   = A->diag->i[idx_local_row+1];
	  row_start_offd = A->offd->i[idx_local_row];
	  row_end_offd   = A->offd->i[idx_local_row+1];
	  for(idx_local_col = row_start_diag; idx_local_col < row_end_diag; idx_local_col++) {
	    if(idx_col == (A->diag->j[idx_local_col] + A->first_row_index)) {
	      printf("%.2f\t", A->diag->data[idx_local_col]);
	      fflush(stdout);
	      flag_colnz = 1;
	      break;
	    }
	  }
	  for(idx_local_col = row_start_offd; idx_local_col<row_end_offd; idx_local_col++) {
	    if(idx_col == A->col_map_offd[A->offd->j[idx_local_col]]) {
	      printf("%.2f\t", A->offd->data[idx_local_col]);
	      fflush(stdout);
	      flag_colnz = 1;
	      break;
	    }
	  }
	  if(1 != flag_colnz) {
	      printf("%.2f\t", 0.0);
	      fflush(stdout);
	  }
	}
	printf("\n");
      } 
    } else {
      usleep(100);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  PASE_Printf(MPI_COMM_WORLD, "\n");
}

void PrintPaseVector(PASE_VECTOR pase_x, char *vector_name)
{
  PASE_INT myid, np, idx_proc, local_size, i;
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  HYPRE_ParVector par_x = (HYPRE_ParVector)pase_x->vector_data;
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
  PASE_Printf(MPI_COMM_WORLD, "\t]^T\n");
}
