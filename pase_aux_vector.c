#include "stdio.h"
#include "stdlib.h"
#include "pase_config.h"
#include "pase_vector.h"
#include "pase_aux_vector.h"

PASE_AUX_VECTOR 
PASE_Aux_vector_create(PASE_AUX_VECTOR aux_x)
{
    PASE_AUX_VECTOR aux_y = (PASE_AUX_VECTOR)PASE_Malloc(sizeof(PASE_AUX_VECTOR_PRIVATE));
    aux_y->vec = PASE_Vector_create_by_vector(aux_x->vec);
    aux_y->is_vec_owner = 1;
    aux_y->block_size = aux_x->block_size;
    aux_y->block = (PASE_SCALAR*)PASE_Malloc(aux_y->block_size*sizeof(PASE_SCALAR));
    return aux_y;
}

void 
PASE_Aux_vector_destroy(PASE_AUX_VECTOR aux_x)
{
    if(NULL != aux_x) {
	if(NULL != aux_x->vec) {
	    PASE_Vector_destroy(aux_x->vec);
	    aux_x->vec = NULL;
	}
	if(NULL != aux_x->block) {
	    PASE_Free(aux_x->block);
	    aux_x->block = NULL;
	}
	PASE_Free(aux_x);
	aux_x = NULL;
    }
}

void
PASE_Aux_vector_copy(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y)
{
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_copy with aux_x being NULL!\n");
	exit(-1);
    }
    if(NULL == aux_y) {
	printf("PASE ERROR: Call PASE_Aux_vector_copy with aux_y being NULL!\n");
	exit(-1);
    }
    PASE_Vector_copy(aux_x->vec, aux_y->vec);
    PASE_INT i = 0;
    for(i=0; i<aux_x->block_size; i++) {
	aux_y->block[i] = aux_x->block[i];
    }
}

void 
PASE_Aux_vector_set_vec(PASE_AUX_VECTOR aux_x, PASE_VECTOR vec)
{
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_set_vec with aux_x being NULL!\n");
	exit(-1);
    }
    if(NULL == vec) {
	printf("PASE ERROR: Call PASE_Aux_vector_set_vec with vec being NULL!\n");
	exit(-1);
    }
    if(NULL != aux_x->vec) {
	PASE_Vector_destroy(aux_x->vec);
	aux_x->vec = NULL;
    }
    aux_x->vec = vec;
}

void 
PASE_Aux_vector_set_constant_value(PASE_AUX_VECTOR aux_x, PASE_SCALAR val)
{
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_set_constant_value with aux_x being NULL!\n");
	exit(-1);
    }
    PASE_Vector_set_constant_value(aux_x->vec, val);
    PASE_Aux_vector_set_block_constant(aux_x, val);
} 

void 
PASE_Aux_vector_set_random_value(PASE_AUX_VECTOR aux_x, PASE_INT seed)
{
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_set_random_value with aux_x being NULL!\n");
	exit(-1);
    }
    PASE_Vector_set_random_value(aux_x->vec, seed);
    PASE_Aux_vector_set_block_random(aux_x, seed);
} 

void 
PASE_Aux_vector_set_block_constant(PASE_AUX_VECTOR aux_x, PASE_SCALAR val)
{
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_set_block_constant with aux_x being NULL!\n");
	exit(-1);
    }

    PASE_INT i = 0;
    for(i=0; i<aux_x->block_size; i++) {
	aux_x->block[i] = val;
    }
} 

void 
PASE_Aux_vector_set_block_random(PASE_AUX_VECTOR aux_x, PASE_INT seed)
{
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_set_block_random with aux_x being NULL!\n");
	exit(-1);
    }

    PASE_INT i = 0;
    srand(seed);
    for(i=0; i<aux_x->block_size; i++) {
	aux_x->block[i] = (double)(rand()/2147483647);
    }
} 

void PASE_Aux_vector_inner_product(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y, PASE_REAL *prod)
{
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_inner_product with aux_x being NULL!\n");
	exit(-1);
    }
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_inner_product with aux_y being NULL!\n");
	exit(-1);
    }
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_inner_product with prod being NULL!\n");
	exit(-1);
    }
    PASE_Vector_inner_product(aux_x->vec, aux_y->vec, prod);

    PASE_INT i = 0;
    for(i=0; i<aux_x->block_size; i++) {
	*prod += aux_x->block[i] * aux_y->block[i]; 
    }
}

void PASE_Aux_vector_add(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y)
{
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_add with aux_x being NULL!\n");
	exit(-1);
    }
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_add with aux_y being NULL!\n");
	exit(-1);
    }
    PASE_Vector_add_vector(aux_x->vec, aux_y->vec);

    PASE_INT i = 0;
    for(i=0; i<aux_x->block_size; i++) {
	aux_y->block[i] += aux_x->block[i];
    }
}

void PASE_Aux_vector_scale(PASE_SCALAR a, PASE_AUX_VECTOR aux_x)
{
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_scale with aux_x being NULL!\n");
	exit(-1);
    }
    PASE_Vector_scale(a, aux_x->vec);

    PASE_INT i = 0;
    for(i=0; i<aux_x->block_size; i++) {
	aux_x->block[i] *= a;
    }
}
