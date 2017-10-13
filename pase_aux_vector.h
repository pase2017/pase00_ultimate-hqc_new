#ifndef __PASE_AUX_VECTOR_H__
#define __PASE_AUX_VECTOR_H__

/*
 * aux vector = [vec  ]
 *              [block]
 */
typedef struct PASE_AUX_VECTOR_PRIVATE_ {
  PASE_VECTOR  vec;
  PASE_SCALAR *block;
  PASE_INT     block_size;
} PASE_AUX_VECTOR_PRIVATE;
typedef PASE_AUX_VECTOR_PRIVATE * PASE_AUX_VECTOR;

//typedef struct PASE_MULTI_AUX_VECTOR_PRIVATE_ {
//  PASE_INT size;
//  PASE_AUX_VECTOR **aux_vector;
//} PASE_MULTI_AUX_VECTOR_PRIVATE;
//typedef PASE_MULTI_AUX_VECTOR_PRIVATE * PASE_MULTI_AUX_VECTOR;

PASE_AUX_VECTOR PASE_Create_aux_vector(PASE_AUX_MATRIX aux_A);
void PASE_Set_aux_vector_vec(PASE_AUX_VECTOR aux_x, PASE_VECTOR vec);
void PASE_Set_aux_vector_value_constant(PASE_AUX_VECTOR aux_x, PASE_SCALAR val);  
void PASE_Set_aux_vector_value_random  (PASE_AUX_VECTOR aux_x, PASE_INT seed);  
void PASE_Set_aux_vector_block_constant(PASE_AUX_VECTOR aux_x, PASE_SCALAR val);  
void PASE_Set_aux_vector_block_random  (PASE_AUX_VECTOR aux_x, PASE_INT seed);  
void PASE_Set_aux_vector_block_value   (PASE_AUX_VECTOR aux_x, PASE_INT idx, PASE_SCALAR val);  
void PASE_Destroy_aux_vector(PASE_VECTOR aux_x);

#endif
