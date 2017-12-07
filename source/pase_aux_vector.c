#include "stdio.h"
#include "stdlib.h"
#include <string.h>
#include <math.h>
#include "pase_config.h"
#include "pase_vector.h"
#include "pase_aux_vector.h"
#include "pase_aux_matrix.h"

#define DEBUG_PASE_AUX_VECTOR 1

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_create"
PASE_AUX_VECTOR
PASE_Aux_vector_create(PASE_VECTOR vec, PASE_SCALAR *block, PASE_INT block_size)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == vec || NULL == block) {
    PASE_Error(__FUNCT__": Cannot create a new PASE AUX VECTOR without vec nor block.\n");
  }
  if(0 >= block_size) {
    PASE_Error(__FUNCT__": Cannot create a new PASE AUX VECTOR with a nonpositive block size %d.\n", block_size);
  }
#endif

  PASE_AUX_VECTOR aux_x = (PASE_AUX_VECTOR)PASE_Malloc(sizeof(PASE_AUX_VECTOR_PRIVATE));
  aux_x->vec            = vec; 
  aux_x->is_vec_owner   = PASE_NO;
  aux_x->block          = block;
  aux_x->block_size     = block_size;
  return aux_x;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_create_by_aux_vector"
PASE_AUX_VECTOR 
PASE_Aux_vector_create_by_aux_vector(PASE_AUX_VECTOR aux_x)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x) {
    PASE_Error(__FUNCT__": Cannot create a new PASE AUX VECTOR without a sample PASE AUX VECTOR.\n");
  }
#endif

  PASE_VECTOR  vec        = PASE_Vector_create_by_vector(aux_x->vec);
  PASE_INT     block_size = aux_x->block_size;
  PASE_SCALAR *block      = (PASE_SCALAR*)PASE_Malloc(block_size*sizeof(PASE_SCALAR));
  memset(block, 0, block_size*sizeof(PASE_SCALAR));

  PASE_AUX_VECTOR aux_y   = PASE_Aux_vector_create(vec, block, block_size);
  aux_y->is_vec_owner     = PASE_YES;
  return aux_y;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_destroy"
void 
PASE_Aux_vector_destroy(PASE_AUX_VECTOR aux_x)
{
  if(NULL == aux_x) return;

#if DEBUG_PASE_MATRIX
  if((PASE_YES != aux_x->is_vec_owner) &&
     (PASE_NO  != aux_x->is_vec_owner)) {
    PASE_Error(__FUNCT__": Cannot decide whether the owner of vec is.");
  }
#endif

  if(NULL != aux_x->vec && PASE_YES == aux_x->is_vec_owner) {
    PASE_Vector_destroy(aux_x->vec);
  }
  if(NULL != aux_x->block) {
    PASE_Free(aux_x->block);
  }
  PASE_Free(aux_x);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_copy"
void
PASE_Aux_vector_copy(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y)
{
#if DEBUG_PASE_AUX_VECTOR
  if(NULL == aux_x || NULL == aux_y) {
    PASE_Error(__FUNCT__": Neither the two PASE AUX VECTORs can be empty.\n");
  }
#endif
  PASE_Vector_copy(aux_x->vec, aux_y->vec);
  memcpy(aux_y->block, aux_x->block, aux_x->block_size*sizeof(PASE_SCALAR));
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_set_vec"
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

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_set_constant_value"
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

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_set_random_value"
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

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_set_block_constant"
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

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_set_block_random"
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

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_inner_product"
void 
PASE_Aux_vector_inner_product(PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y, PASE_REAL *prod)
{
    if(NULL == aux_x || NULL == aux_x->vec || NULL == aux_x->block) {
	printf("PASE ERROR: Call PASE_Aux_vector_inner_product with aux_x being NULL!\n");
	exit(-1);
    }
    if(NULL == aux_y || NULL == aux_y->vec || NULL == aux_y->block) {
	printf("PASE ERROR: Call PASE_Aux_vector_inner_product with aux_y being NULL!\n");
	exit(-1);
    }
    PASE_Vector_inner_product(aux_x->vec, aux_y->vec, prod);

    PASE_INT i = 0;
    for(i=0; i<aux_x->block_size; i++) {
	*prod += aux_x->block[i] * aux_y->block[i]; 
    }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_inner_product_some"
void 
PASE_Aux_vector_inner_product_some(PASE_AUX_VECTOR *aux_x, PASE_INT start, PASE_INT end, PASE_REAL **prod)
{
#if DEBUG_PASE_AUX_VECTOR
  if((NULL == aux_x) || (NULL == prod)) {
    PASE_Error(__FUNCT__": Vectors and products cannot be empty.\n");
  }
#endif

  PASE_INT i = 0;
  PASE_INT j = 0;
  for(i = start; i <= end; ++i) {
    for(j = start; j <= i; ++j) {
      PASE_Aux_vector_inner_product(x[j], x[i], &prod[i-start][j-start]);
      prod[j-start][i-start] = prod[i-start][j-start];
    }
  }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_norm"
void
PASE_Aux_vector_norm(PASE_AUX_VECTOR aux_x, PASE_REAL *norm)
{
    PASE_Aux_vector_inner_product(aux_x, aux_x, norm);
    *norm = sqrt(*norm);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_axpy"
void 
PASE_Aux_vector_axpy(PASE_SCALAR a, PASE_AUX_VECTOR aux_x, PASE_AUX_VECTOR aux_y)
{
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_add with aux_x being NULL!\n");
	exit(-1);
    }
    if(NULL == aux_x) {
	printf("PASE ERROR: Call PASE_Aux_vector_add with aux_y being NULL!\n");
	exit(-1);
    }
    PASE_Vector_axpy(a, aux_x->vec, aux_y->vec);

    PASE_INT i = 0;
    for(i=0; i<aux_x->block_size; i++) {
	aux_y->block[i] += a * aux_x->block[i];
    }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_scale"
void 
PASE_Aux_vector_scale(PASE_SCALAR a, PASE_AUX_VECTOR aux_x)
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

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_orthogonalize"
void
PASE_Aux_vector_orthogonalize(PASE_AUX_VECTOR *aux_x, PASE_INT i, PASE_INT start, PASE_INT end)
{
#if DEBUG_PASE_AUX_VECTOR
  if((i >= start) && (i <= end)) {
    PASE_Error(__FUNCT__": index %d cannot locate in [%d, %d].\n", i, start, end);
  }
#endif

  PASE_INT  j    = 0;
  PASE_REAL prod = 0.0;
  PASE_REAL norm = 0.0;
  for(j = start; j <= end; ++j) {
    PASE_Aux_vector_inner_product(aux_x[j], aux_x[i], &prod);
    PASE_Aux_vector_axpy(-prod, aux_x[j], aux_x[i]);
  }
  PASE_Aux_vector_norm(aux_x[i], aux_x[i], &norm);
  PASE_Aux_vector_scale(1.0/norm, aux_x[i]);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Aux_vector_orthogonalize_all"
void
PASE_Aux_vector_orthogonalize_all(PASE_AUX_VECTOR *aux_x, PASE_INT num)
{
    PASE_INT j = 0;
    for(j = 0; j < num; ++j) {
      PASE_Aux_vector_orthogonalize(aux_x, j, 0, j-1);
    }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multi_aux_vector_combination"
void
PASE_Multi_aux_vector_combination(PASE_AUX_VECTOR *aux_x, PASE_INT num_vec, PASE_SCALAR *coef, PASE_AUX_VECTOR aux_y)
{
    PASE_INT j = 0;
    PASE_Aux_vector_set_constant_value(aux_y, 0.0);
    for(j=0; j<num_vec; j++) {
        PASE_Aux_vector_axpy(coef[j], aux_x[j], aux_y);
    }
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multi_aux_vector_by_matrix"
void
PASE_Multi_aux_vector_by_matrix(PASE_AUX_VECTOR *aux_x, PASE_INT num_vec, PASE_SCALAR **mat, PASE_INT num_mat, PASE_AUX_VECTOR *aux_y)
{
    PASE_INT i;
    for(i=0; i<num_mat; i++) {
        PASE_Multi_aux_vector_combination(aux_x, num_vec, mat[i], aux_y[i]);
    }
}

