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
#include "pase_aux_vector.h"
#include "pase_aux_matrix.h"
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


void GetTestHypreParCSRMatrix(HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, HYPRE_IJMatrix *R, PASE_INT n);
void GetTestHypreParVector(HYPRE_IJVector *x, PASE_INT n, PASE_INT block_size);
void GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n, PASE_INT *block_size);
void PrintPaseAuxMatrix(PASE_AUX_MATRIX aux_A, char *matrix_name);
void PrintPaseMatrix(PASE_MATRIX pase_A, char *matrix_name);
void PrintPaseAuxVector(PASE_AUX_VECTOR aux_x, char *vector_name);
void PrintPaseVector(PASE_VECTOR pase_x, char *vector_name);
void PrintPaseVectors(PASE_VECTOR *pase_x, PASE_INT block_size, char *vector_name);

PASE_INT main (PASE_INT argc, char *argv[])
{
  PASE_INT myid, num_procs;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  PASE_INT n          = 4;  //原矩阵行数
  PASE_INT block_size = 1;  //辅助空间维数
  GetCommandLineInfo(argc, argv, &n, &block_size);

  HYPRE_IJMatrix A_h, B_h, R_hH;
  HYPRE_ParCSRMatrix parcsr_A_h, parcsr_B_h, parcsr_R_hH;
  GetTestHypreParCSRMatrix(&A_h, &B_h, &R_hH, n);
  HYPRE_IJMatrixGetObject(A_h, (void**) &parcsr_A_h);
  HYPRE_IJMatrixGetObject(B_h, (void**) &parcsr_B_h);
  HYPRE_IJMatrixGetObject(R_hH, (void**) &parcsr_R_hH);

  HYPRE_IJVector *x_h = (HYPRE_IJVector*)PASE_Malloc(block_size*sizeof(HYPRE_IJVector));
  HYPRE_ParVector *par_x_h = (HYPRE_ParVector*)PASE_Malloc(block_size*sizeof(HYPRE_ParVector)) ;
  GetTestHypreParVector(x_h, n, block_size);

  //PASE MATRIX 测试
  {
    //Initial: get PASE_MATRIX pase_A_h, pase_B_h, pase_R_hH, pase_P_Hh
    PASE_MATRIX pase_A_h = PASE_Matrix_create((void*)parcsr_A_h, 1); 
    PASE_MATRIX pase_B_h = PASE_Matrix_create((void*)parcsr_B_h, 1); 
    PASE_MATRIX pase_R_hH = PASE_Matrix_create((void*)parcsr_R_hH, 1); 
    PASE_MATRIX pase_P_Hh = PASE_Matrix_transpose(pase_R_hH);
    PASE_INT idx_block = 0;
    PASE_VECTOR *pase_x_h = (PASE_VECTOR*)PASE_Malloc(block_size*sizeof(PASE_VECTOR));
    for(idx_block = 0; idx_block < block_size; idx_block++) {
      HYPRE_IJVectorGetObject(x_h[idx_block], (void **) &(par_x_h[idx_block]));
      pase_x_h[idx_block] = PASE_Vector_create((void*)par_x_h[idx_block], 1);
    }
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n");
    PASE_Printf(MPI_COMM_WORLD, "This is a test program for PASE_AUX_MATRIX\n");
    PASE_Printf(MPI_COMM_WORLD, "=============================================================\n");
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(MPI_COMM_WORLD, "Initial\n\n");
    PrintPaseMatrix(pase_A_h, "A_h");
    PrintPaseMatrix(pase_B_h, "B_h");
    PrintPaseMatrix(pase_R_hH, "R_hH");
    PrintPaseMatrix(pase_P_Hh, "P_Hh");
    PrintPaseVectors(pase_x_h, block_size, "x_h");
    PASE_Printf(MPI_COMM_WORLD, "-------------------------------------------------------------\n\n");

    //Get A_H = R_hH * A_h * P_Hh, B_H = R_hH * A_h * P_Hh
    PASE_Printf(MPI_COMM_WORLD, "Get A_H = R_hH * A_h * P_Hh, B_H = R_hH * A_h * P_Hh\n\n");
    PASE_MATRIX tmp       = PASE_Matrix_multiply_matrix(pase_R_hH, pase_A_h);
    PASE_MATRIX pase_A_H  = PASE_Matrix_multiply_matrix(tmp, pase_P_Hh);
    PASE_Matrix_destroy(tmp);
    tmp = PASE_Matrix_multiply_matrix(pase_R_hH, pase_B_h);
    PASE_MATRIX pase_B_H  = PASE_Matrix_multiply_matrix(tmp, pase_P_Hh);
    PASE_Matrix_destroy(tmp);
    PrintPaseMatrix(pase_A_H, "A_H");
    PrintPaseMatrix(pase_B_H, "B_H");
    PASE_Printf(MPI_COMM_WORLD, "-------------------------------------------------------------\n\n");

    //Get aux_A_H = [A_H a    ], aux_B_H = [B_H b   ], aux_x_H = [1.0, ..., 1.0]
    //              [a^T alpha]            [b^T beta]
    PASE_AUX_MATRIX aux_A_H = PASE_Aux_matrix_create(pase_A_H, pase_R_hH, pase_A_h, pase_x_h, block_size);
    PASE_AUX_MATRIX aux_B_H = PASE_Aux_matrix_create(pase_B_H, pase_R_hH, pase_B_h, pase_x_h, block_size);
    PASE_AUX_VECTOR aux_x_H = PASE_Aux_vector_create_by_aux_matrix(aux_A_H);
    PASE_AUX_VECTOR aux_y_H = PASE_Aux_vector_create_by_aux_matrix(aux_A_H);
    PASE_AUX_VECTOR aux_z_H = PASE_Aux_vector_create_by_aux_matrix(aux_A_H);
    PASE_Aux_vector_set_constant_value(aux_x_H, 1.0);
    PASE_Printf(MPI_COMM_WORLD, "Create aux_A_H = [A_H a    ], where a     = R_hH  * A_h * x_h\n");
    PASE_Printf(MPI_COMM_WORLD, "                 [a^T alpha]        alpha = x_h^T * A_h * x_h\n\n");
    PrintPaseAuxMatrix(aux_A_H, "aux_A_H");
    PASE_Printf(MPI_COMM_WORLD, "\nCreate aux_B_H = [B_H b    ], where b     = R_hH  * B_h * x_h\n");
    PASE_Printf(MPI_COMM_WORLD, "                 [b^T beta ]        beta  = x_h^T * B_h * x_h\n\n");
    PrintPaseAuxMatrix(aux_B_H, "aux_B_H");
    PASE_Printf(MPI_COMM_WORLD, "Create aux_X = [1.0, ..., 1.0]\n\n");
    PrintPaseAuxVector(aux_x_H, "aux_x_H");
    PASE_Printf(MPI_COMM_WORLD, "-------------------------------------------------------------\n\n");

    //Multiply matrix and vector
    PASE_Printf(MPI_COMM_WORLD, "Multiply aux_y_H = aux_A_H * aux_x_H\n\n");
    PASE_Aux_matrix_multiply_aux_vector(aux_A_H, aux_x_H, aux_y_H);
    PrintPaseAuxVector(aux_y_H, "aux_y_H");
    PASE_SCALAR a = 1.0;
    PASE_SCALAR b = 2.0;
    PASE_Printf(MPI_COMM_WORLD, "\nGeneral multiply aux_y_H = %.2f * aux_A_H * aux_x_H + %.2f * aux_y_H\n\n", a, b);
    PASE_Aux_matrix_multiply_aux_vector_general(a, aux_A_H, aux_x_H, b, aux_y_H);
    PrintPaseAuxVector(aux_y_H, "aux_y_H");
    PASE_Printf(MPI_COMM_WORLD, "-------------------------------------------------------------\n\n");

    // orthogonalization 
    PASE_AUX_VECTOR *aux_X_H = (PASE_AUX_VECTOR*)PASE_Malloc(3*sizeof(PASE_AUX_VECTOR));
    PASE_Aux_vector_set_random_value(aux_z_H, 1);
    PASE_Printf(PASE_COMM_WORLD, "set random values to aux_z_H\n\n");
    PrintPaseAuxVector(aux_z_H, "aux_z_H");
    PASE_Printf(MPI_COMM_WORLD, "\n");
    PASE_Printf(PASE_COMM_WORLD, "B-orthogonalize [aux_x_H, aux_y_H, aux_z_H]\n\n");
    aux_X_H[0] = aux_x_H;
    aux_X_H[1] = aux_y_H;
    aux_X_H[2] = aux_z_H;
    PASE_Aux_vector_orthogonalize_general_all(aux_X_H, 3, aux_B_H); //这个函数里调用了 PASE_Vector_orthogonalize_general
    PrintPaseAuxVector(aux_x_H, "aux_x_H");
    PrintPaseAuxVector(aux_y_H, "aux_y_H");
    PrintPaseAuxVector(aux_z_H, "aux_z_H");
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");

    // inner product
    PASE_REAL **prods = (PASE_REAL**)PASE_Malloc(3*sizeof(PASE_REAL*));
    prods[0] = (PASE_REAL*)PASE_Malloc(3*sizeof(PASE_REAL));
    prods[1] = (PASE_REAL*)PASE_Malloc(3*sizeof(PASE_REAL));
    prods[2] = (PASE_REAL*)PASE_Malloc(3*sizeof(PASE_REAL));
    PASE_Printf(PASE_COMM_WORLD, "B-inner product\n\n");
    PASE_Aux_vector_inner_product_general_some(aux_X_H, 0, 2, aux_B_H, prods); //这个函数里调用了 PASE_Vector_inner_product_general
    PASE_Printf(PASE_COMM_WORLD, "(aux_x,aux_x) = %.2f, (aux_x,aux_y) = %.2f, (aux_x,aux_z) = %.2f\n", prods[0][0], prods[0][1], prods[0][2]);
    PASE_Printf(PASE_COMM_WORLD, "(aux_y,aux_y) = %.2f, (aux_y,aux_z) = %.2f\n", prods[1][1], prods[1][2]);
    PASE_Printf(PASE_COMM_WORLD, "(aux_z,aux_z) = %.2f\n", prods[2][2]);
    PASE_Printf(PASE_COMM_WORLD, "-------------------------------------------------------------\n\n");

    // destroy
    PASE_Matrix_destroy(pase_A_h);
    PASE_Matrix_destroy(pase_B_h);
    PASE_Matrix_destroy(pase_A_H);
    PASE_Matrix_destroy(pase_B_H);
    PASE_Aux_vector_destroy(aux_x_H);
    PASE_Aux_vector_destroy(aux_y_H);
    PASE_Aux_vector_destroy(aux_z_H);
    PASE_Free(prods[0]); PASE_Free(prods[1]); PASE_Free(prods[2]);
    PASE_Free(prods);
    PASE_Free(aux_X_H);
  }

  HYPRE_IJMatrixDestroy(A_h);
  HYPRE_IJMatrixDestroy(B_h);
  HYPRE_IJMatrixDestroy(R_hH);
  //HYPRE_IJVectorDestroy(x);
  MPI_Finalize();
  return 0;
}

void 
GetCommandLineInfo(PASE_INT argc, char **argv, PASE_INT *n, PASE_INT *block_size)
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

void 
GetTestHypreParCSRMatrix(HYPRE_IJMatrix *A, HYPRE_IJMatrix *B, HYPRE_IJMatrix *R, PASE_INT n)
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
    values[nnz]   = 1.0;
    nnz++;

    /* Set the values for row i */
    HYPRE_IJMatrixSetValues(*B, 1, &nnz, &i, cols, values);
  }

  HYPRE_IJMatrixAssemble(*A);
  HYPRE_IJMatrixAssemble(*B);

  PASE_INT jlower, jupper;
  n = n / 2;
  local_size   = n / num_procs;
  extra        = n - local_size*num_procs;
  jlower       = local_size * myid;
  jlower      += hypre_min(myid, extra);
  jupper       = local_size * (myid+1);
  jupper      += hypre_min(myid+1, extra);
  jupper       = jupper - 1;
  local_size   = jupper - jlower + 1;
  HYPRE_IJMatrixCreate(MPI_COMM_WORLD, jlower, jupper, ilower, iupper, R);
  HYPRE_IJMatrixSetObjectType(*R, HYPRE_PARCSR);
  HYPRE_IJMatrixInitialize(*R);
  for(i = jlower; i <= jupper; i++) {
    nnz       = 1;
    cols[0]   = i;
    values[0] = 1.0;

    /* Set the values for row i */
    HYPRE_IJMatrixSetValues(*R, 1, &nnz, &i, cols, values);
  }
  HYPRE_IJMatrixAssemble(*R);
}

void 
GetTestHypreParVector(HYPRE_IJVector *x, PASE_INT n, PASE_INT block_size)
{
  PASE_INT     ilower, iupper, local_size, extra, i, idx_block;
  PASE_SCALAR *x_values;
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

  x_values = (PASE_SCALAR*)calloc(local_size, sizeof(PASE_SCALAR));
  rows     = (PASE_INT*)   calloc(local_size, sizeof(PASE_INT));

  for(idx_block = 0; idx_block < block_size; idx_block++) {
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &(x[idx_block]));
    HYPRE_IJVectorSetObjectType(x[idx_block], HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x[idx_block]);
    for(i = 0; i < local_size; i++) {
      x_values[i] = pow(i, idx_block);;
      rows[i]     = ilower + i;
    }
    HYPRE_IJVectorSetValues(x[idx_block], local_size, rows, x_values);
    HYPRE_IJVectorAssemble(x[idx_block]);
  }

  free(x_values);
  free(rows);
}

void 
PrintPaseAuxMatrix(PASE_AUX_MATRIX aux_A, char *matrix_name)
{
  PASE_INT myid, np, idx_proc, size_local_row, idx_col, idx_local_row, idx_local_col, flag_colnz, idx_block, i;
  PASE_INT row_start_diag, row_end_diag, row_start_offd, row_end_offd; 
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  HYPRE_ParCSRMatrix A = (HYPRE_ParCSRMatrix)aux_A->mat->matrix_data;
  HYPRE_ParVector par_tmp;
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
	printf("|\t");
	for(idx_block = 0; idx_block<aux_A->block_size; idx_block++) {
	  par_tmp = (HYPRE_ParVector)aux_A->vec[idx_block]->vector_data;
	  printf("%.2f\t", par_tmp->local_vector->data[idx_local_row]);
	}
	printf("\n");
      } 
    } else {
      usleep(100);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  PASE_INT global_col_H = 0;
  PASE_Matrix_get_global_ncol(aux_A->mat, &global_col_H);
  for(idx_col = 0; idx_col < global_col_H + aux_A->block_size + 1; idx_col++) {
    PASE_Printf(MPI_COMM_WORLD, "-   -   ");
  }
  PASE_Printf(MPI_COMM_WORLD, "\n");

  for(idx_proc = 0; idx_proc < np; idx_proc++) {
    if(myid == idx_proc) {
      for(idx_local_row = 0; idx_local_row < aux_A->block_size; idx_local_row++) {
	par_tmp = (HYPRE_ParVector)aux_A->vec[idx_local_row]->vector_data;
	for(idx_local_col = 0; idx_local_col < size_local_row; idx_local_col++) {
	  printf("%.2f\t", par_tmp->local_vector->data[idx_local_col]);
	  fflush(stdout);
	}
	printf("|\t");
	for(i = 0; i<aux_A->block_size; ++i) {
	  PASE_Printf(MPI_COMM_WORLD, "%.2f\t", aux_A->block[idx_local_row][i]);
	}
	PASE_Printf(MPI_COMM_WORLD, "\n");
      }
    } else {
      usleep(50);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  PASE_Printf(MPI_COMM_WORLD, "\n");
}

void 
PrintPaseMatrix(PASE_MATRIX pase_A, char *matrix_name)
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

  void 
PrintPaseAuxVector(PASE_AUX_VECTOR aux_x, char *vector_name)
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

  void 
PrintPaseVectors(PASE_VECTOR *pase_x, PASE_INT block_size, char *vector_set_name)
{
  PASE_INT idx_block = 0;
  char vector_name[20];
  for(idx_block = 0; idx_block < block_size; idx_block++) {
    sprintf(vector_name, "%s%d", vector_set_name, idx_block);
    PrintPaseVector(pase_x[idx_block], vector_name);
  }
}

  void 
PrintPaseVector(PASE_VECTOR pase_x, char *vector_name)
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
