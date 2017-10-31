#ifndef __PASE_AUX_VECTOR_H__
#define __PASE_AUX_VECTOR_H__

#include "pase_vector.h"

/*
 * aux vector = [vec  ]
 *              [block]
 */
typedef struct PASE_AUX_VECTOR_PRIVATE_ {
  PASE_VECTOR  vec;
  PASE_SCALAR *block;
  PASE_INT     block_size;
  PASE_INT     is_vec_owner;
} PASE_AUX_VECTOR_PRIVATE;
typedef PASE_AUX_VECTOR_PRIVATE * PASE_AUX_VECTOR;

//typedef struct PASE_MULTI_AUX_VECTOR_PRIVATE_ {
//  PASE_INT size;
//  PASE_AUX_VECTOR **aux_vector;
//} PASE_MULTI_AUX_VECTOR_PRIVATE;
//typedef PASE_MULTI_AUX_VECTOR_PRIVATE * PASE_MULTI_AUX_VECTOR;

#include "pase_aux_matrix.h"

PASE_AUX_VECTOR PASE_Aux_vector_create_by_aux_vector(PASE_AUX_VECTOR aux_x);
PASE_AUX_VECTOR PASE_Aux_vector_create_by_aux_matrix(PASE_AUX_MATRIX aux_A);
void PASE_Aux_vector_destroy(PASE_AUX_VECTOR aux_x);
void PASE_Aux_vector_copy(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y);
void PASE_Aux_vector_set_vec(PASE_AUX_VECTOR aux_x, PASE_VECTOR vec);
void PASE_Aux_vector_set_constant_value(PASE_AUX_VECTOR aux_x, PASE_SCALAR val);  
void PASE_Aux_vector_set_random_value(PASE_AUX_VECTOR aux_x, PASE_INT seed);  
void PASE_Aux_vector_set_block_constant(PASE_AUX_VECTOR aux_x, PASE_SCALAR val);  
void PASE_Aux_vector_set_block_random(PASE_AUX_VECTOR aux_x, PASE_INT seed);  
//void PASE_Aux_vector_set_block_value(PASE_AUX_VECTOR aux_x, PASE_INT idx, PASE_SCALAR val);  
void PASE_Aux_vector_inner_product(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y, PASE_REAL *prod);
void PASE_Aux_vector_add(PASE_SCALAR a, PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y);
void PASE_Aux_vector_scale(PASE_SCALAR a, PASE_AUX_VECTOR aux_x);

#endif
