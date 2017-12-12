#ifndef __PASE_VECTOR_HYPRE_H__
#define __PASE_VECTOR_HYPRE_H__

#include "pase_config.h"

#if PASE_USE_HYPRE

void* PASE_Vector_data_create_by_vector_hypre(void *x);
void* PASE_Vector_data_create_by_matrix_hypre(void *A);
void  PASE_Vector_data_copy_hypre(void *x, void *y);
void  PASE_Vector_data_destroy_hypre(void *x);
void  PASE_Vector_data_set_constant_value_hypre(void *x, PASE_SCALAR a);
void  PASE_Vector_data_set_random_value_hypre(void *x, PASE_INT seed);
void  PASE_Vector_data_inner_product_hypre(void *x, void *y, PASE_REAL *prod);
void  PASE_Vector_data_axpy_hypre(PASE_SCALAR a, void *x, void *y);
void  PASE_Vector_data_scale_hypre(PASE_SCALAR a, void *x);
void  PASE_Vector_data_get_global_nrow_hypre(void *x, PASE_INT *nrow);

#endif

#endif
