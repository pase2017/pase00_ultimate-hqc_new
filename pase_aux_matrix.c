#include <stdio.h>
#include <stdlib.h>
#include "pase_matrix.h"
#include "pase_vector.h"
#include "pase_aux_matrix.h"
#include "pase_aux_vector.h"
#include "pase_config.h"
#include "pase_param.h"


PASE_AUX_MATRIX 
PASE_Aux_matrix_create(PASE_MATRIX mat, PASE_PARAMETER param)
{
    PASE_AUX_MATRIX aux_A = (PASE_AUX_MATRIX)PASE_Malloc(sizeof(PASE_AUX_MATRIX_PRIVATE));
    aux_A->mat = mat;
    aux_A->is_mat_owner   = 0;

    return aux_A;
}

void 
PASE_Aux_matrix_set_aux_spase_some(PASE_AUX_MATRIX aux_A, PASE_INT i, PASE_INT j, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h)
{
    PASE_INT k, l;
    if(NULL == aux_A->vec) {
	aux_A->vec = (PASE_VECTOR*)PASE_Malloc(aux_A->block_size*sizeof(PASE_VECTOR));
	for(k=0; k<aux_A->block_size; k++) {
	    aux_A->vec[k] = PASE_Vector_create_by_matrix(aux_A->mat);
	}
    }
    if(NULL == aux_A->block) {
	aux_A->block = (PASE_SCALAR**)PASE_Malloc(aux_A->block_size*sizeof(PASE_SCALAR*));
	for(k=0; k<aux_A->block_size; k++) {
	    aux_A->block[k] = (PASE_SCALAR*)PASE_Malloc(aux_A->block_size*sizeof(PASE_SCALAR));
	}
    }
    PASE_VECTOR workspace_h = PASE_Vector_create_by_vector(u_h[0]);
    for(k=i; k<=j; k++) {
	PASE_Matrix_multiply_vector(A_h, u_h[k], workspace_h);
	PASE_Matrix_multiply_vector(R_hH, workspace_h, aux_A->vec[k]);
	for(l=0; l<aux_A->block_size; l++) {
	    if(l >= i && l <= j) {
		PASE_Vector_inner_product(workspace_h, u_h[l], &(aux_A->block[k][l]));
	    } else {
		PASE_Vector_inner_product(workspace_h, u_h[l], &(aux_A->block[l][k]));
		PASE_Vector_inner_product(workspace_h, u_h[l], &(aux_A->block[k][l]));
	    }
	}
    }
    PASE_Vector_destroy(workspace_h);
}

void 
PASE_Aux_matrix_set_aux_space(PASE_AUX_MATRIX aux_A, PASE_MATRIX R_hH, PASE_MATRIX A_h, PASE_VECTOR *u_h)
{
    PASE_Aux_matrix_set_aux_space_some(aux_A, 0, aux_A->block_size-1, R_hH, A_h, u_h);
}


void 
PASE_Aux_matrix_destroy(PASE_AUX_MATRIX aux_A)
{
    PASE_INT i;
    if(NULL != aux_A) {
	if(NULL != aux_A->mat) {
	    PASE_Matrix_destroy(aux_A->mat);
	    aux_A->mat = NULL;
	}
	if(NULL != aux_A->vec) {
	    for(i=0; i<aux_A->block_size; i++) {
		PASE_Vector_destroy(aux_A->vec[i]);
		aux_A->vec[i] = NULL;
	    }
	    PASE_Free(aux_A->vec);
	    aux_A->vec = NULL;
	}
	if(NULL != aux_A->block) {
	    for(i=0; i<aux_A->block_size; i++) {
		PASE_Free(aux_A->block[i]);
		aux_A->block[i] = NULL;
	    }
	    PASE_Free(aux_A->block);
	    aux_A->block = NULL;
	}
	PASE_Free(aux_A);
	aux_A = NULL;
    }
}

void 
PASE_Aux_matrix_copy(PASE_AUX_MATRIX aux_A, PASE_AUX_MATRIX aux_B)
{
    PASE_INT i, j;
    PASE_Matrix_copy(aux_A->mat, aux_B->mat);
    for(i=0; i<aux_A->block_size; i++) {
	PASE_Vector_copy(aux_A->vec[i], aux_B->vec[i]);
	for(j=0; j<aux_A->block_size; j++) {
	    aux_B->block[i][j] = aux_A->block[i][j];
	}
    }
}

void 
PASE_Aux_Matrix_multiply_aux_matrix(PASE_AUX_MATRIX aux_A, PASE_AUX_MATRIX aux_B, PASE_AUX_MATRIX aux_C)
{

}

void 
PASE_Aux_Matrix_multiply_aux_vector(PASE_AUX_MATRIX aux_A, PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y)
{
    PASE_INT i, j;
    PASE_Matrix_multiply_vector(aux_A->mat, aux_x->vec, aux_y->vec);
    for(i=0; i<aux_A->block_size; i++) {
	PASE_Vector_add_vector(aux_x->block[i], aux_A->vec[i], aux_y->vec);
	PASE_Vector_inner_product(aux_A->vec[i], aux_x->vec, &(aux_y->block[i]));
	for(j=0; j<aux_A->block_size; j++) {
	    aux_y->block[i] += aux_A->block[j][i] * aux_x->block[j];
	}
    }
}

