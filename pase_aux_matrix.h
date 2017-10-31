#ifndef __PASE_AUX_MATRIX_H__
#define __PASE_AUX_MATRIX_H__

#include "pase_matrix.h"
#include "pase_vector.h"

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
  PASE_INT      is_mat_owner;
} PASE_AUX_MATRIX_PRIVATE;
typedef PASE_AUX_MATRIX_PRIVATE * PASE_AUX_MATRIX;

#include "pase_aux_vector.h"

PASE_AUX_MATRIX PASE_Aux_matrix_create(PASE_MATRIX A_H, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h, PASE_INT block_size);
void PASE_Aux_matrix_set_aux_space_some(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h); 
void PASE_Aux_matrix_set_aux_space(PASE_AUX_MATRIX aux_A, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h); 
PASE_AUX_MATRIX PASE_Aux_matrix_create_by_aux_matrix(PASE_MATRIX A_H, PASE_MATRIX R_hH, PASE_AUX_MATRIX aux_A_h, PASE_AUX_VECTOR *aux_u_h, PASE_INT block_size);
void PASE_Aux_matrix_set_aux_space_some_by_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX R_hH, PASE_AUX_MATRIX aux_A_h, PASE_AUX_VECTOR *aux_u_h);
void PASE_Aux_matrix_set_aux_space_by_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_MATRIX R_hH, PASE_AUX_MATRIX aux_A_h, PASE_AUX_VECTOR *aux_u_h);
void PASE_Aux_matrix_destroy(PASE_AUX_MATRIX aux_A);
void PASE_Aux_matrix_copy(PASE_AUX_MATRIX aux_A, PASE_AUX_MATRIX aux_B);
void PASE_Aux_matrix_multiply_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_AUX_MATRIX aux_B, PASE_AUX_MATRIX aux_C);
void PASE_Aux_matrix_multiply_aux_vector(PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y);
void PASE_Aux_matrix_multiply_aux_vector_general(PASE_SCALAR a, PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_x, PASE_SCALAR b, PASE_AUX_VECTOR aux_y);
void PASE_Aux_matrix_get_comm_info(PASE_AUX_MATRIX aux_A, MPI_Comm *comm);

#endif
