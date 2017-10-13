#ifndef __PASE_AUX_MATRIX_H__
#define __PASE_AUX_MATRIX_H__

#include "pase_matrix.h"

//typedef struct PASE_AUX_MATRIX_OPERATOR_PRIVATE_ {
//
//} PASE_AUX_MATRIX_OPERATOR_PRIVATE; 
//typedef PASE_AUX_MATRIX_OPERATOR_PRIVATE * PASE_AUX_MATRIX_OPERATOR;

/*
 * aux matrix = [mat   vec  ]
 *              [vecT  block]
 */
typedef struct PASE_AUX_MATRIX_PRIVATE_ {
  PASE_MATRIX   mat;
  PASE_VECTOR  *vec;
  PASE_SCALAR **block;
  PASE_INT      block_size;
} PASE_AUX_MATRIX_PRIVATE;
typedef PASE_AUX_MATRIX_PRIVATE * PASE_AUX_MATRIX;

PASE_AUX_MATRIX PASE_Create_aux_matrix(PASE_MATRIX mat, PASE_PARAMETER param);
void PASE_Set_aux_matrix_vec_some(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h); 
void PASE_Set_aux_matrix_vec(PASE_AUX_MATRIX aux_A, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h); 
void PASE_Set_aux_matrix_block(PASE_AUX_MATRIX aux_A, PASE_MATRIX A_h, PASE_VECTOR *u_h); 
void PASE_Destroy_aux_matrix(PASE_AUX_MATRIX aux_A);
void PASE_Matrix_multiply_matrix_matrix(PASE_AUX_MATRIX aux_A, PASE_AUX_MATRIX aux_B, PASE_AUX_MATRIX aux_C);
void PASE_Matrix_multiply_matrix_vector(PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y);

#endif
