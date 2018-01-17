#include <stdio.h>
#include <stdlib.h>
#include "pase_multigrid.h"

#ifdef PASE_USE_HYPRE
#include "pase_multigrid_hypre.h"
#endif

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multigrid_create"
/**
 * @brief 创建 PASE_MULTIGRID
 *
 * @param A      输入参数
 * @param B      输入参数
 * @param param  输入参数, 包含 AMG 分层的各个参数
 * @param ops    输入参数, 多重网格操作集合
 *
 * @return PASE_MULTIGRID
 */
PASE_MULTIGRID 
PASE_Multigrid_create(PASE_MATRIX A, PASE_MATRIX B, PASE_PARAMETER param, PASE_MULTIGRID_OPERATOR ops)
{
  PASE_MULTIGRID multigrid = (PASE_MULTIGRID)PASE_Malloc(sizeof(PASE_MULTIGRID_PRIVATE));
  if(NULL != ops) {
    multigrid->ops = (PASE_MULTIGRID_OPERATOR) PASE_Malloc(sizeof(PASE_MULTIGRID_OPERATOR_PRIVATE));
    *(multigrid->ops) = *ops;
  } else {
    multigrid->ops = PASE_Multigrid_operator_create(A->data_form);
  }

  //PASE_Multigrid_set_up(multigrid, A, B, param);

  return multigrid;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multigrid_get_amg_array"
/**
 * @brief AMG 分层
 *
 * @param multigrid  输入/输出参数 
 * @param A          输入参数
 * @param B          输入参数
 * @param param      输入参数, 包含 AMG 分层的各个参数
 */
void
PASE_Multigrid_set_up(PASE_MULTIGRID multigrid, PASE_MATRIX A, PASE_MATRIX B, PASE_PARAMETER param)
{
  void **A_array, **P_array, **R_array; 
  PASE_INT    level = 0;
  PASE_MATRIX tmp   = NULL;
  multigrid->ops->get_amg_array(A->matrix_data, 
                                param, 
                                &(A_array),
                                &(P_array),
                                &(R_array),
                                &(multigrid->actual_level),
                                &(multigrid->amg_data));
  multigrid->A     = (PASE_MATRIX*)PASE_Malloc(multigrid->actual_level*sizeof(PASE_MATRIX));
  multigrid->B     = (PASE_MATRIX*)PASE_Malloc(multigrid->actual_level*sizeof(PASE_MATRIX));
  multigrid->P     = (PASE_MATRIX*)PASE_Malloc((multigrid->actual_level-1)*sizeof(PASE_MATRIX));
  multigrid->R     = (PASE_MATRIX*)PASE_Malloc((multigrid->actual_level-1)*sizeof(PASE_MATRIX));
  multigrid->aux_A = (PASE_AUX_MATRIX*)calloc(multigrid->actual_level, sizeof(PASE_AUX_MATRIX));
  multigrid->aux_B = (PASE_AUX_MATRIX*)calloc(multigrid->actual_level, sizeof(PASE_AUX_MATRIX));
  multigrid->A[0]  = A;
  multigrid->B[0]  = B;
  for(level=1; level<multigrid->actual_level; level++) {
    multigrid->A[level]                         = PASE_Matrix_assign(A_array[level], A->ops);
    multigrid->A[level]->data_form              = A->data_form;
    multigrid->P[level-1]                       = PASE_Matrix_assign(P_array[level-1], A->ops);
    multigrid->P[level-1]->data_form            = A->data_form;
    multigrid->R[level-1]                       = PASE_Matrix_assign(R_array[level-1], A->ops);
    multigrid->R[level-1]->is_matrix_data_owner = 1;
    multigrid->R[level-1]->data_form            = A->data_form;

    /* B1 = R0 * B0 * P0 */
    tmp                                         = PASE_Matrix_multiply_matrix(multigrid->B[level-1], multigrid->P[level-1]); 
    multigrid->B[level]                         = PASE_Matrix_multiply_matrix(multigrid->R[level-1], tmp); 
    PASE_Matrix_destroy(tmp);
  }
  PASE_Free(R_array);
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multigrid_destroy"
/**
 * @brief 销毁 PASE_MULTIGRID
 *
 * @param multigrid 输入参数
 */
void 
PASE_Multigrid_destroy(PASE_MULTIGRID multigrid)
{
  PASE_INT i = 0;
  if(NULL != multigrid && multigrid->actual_level > 1) {
    if(multigrid->aux_A) {
      for(i=0; i<multigrid->actual_level; i++) {
	if(NULL != multigrid->aux_A[i]) {
	  PASE_Aux_matrix_destroy(multigrid->aux_A[i]);
	}
      }
      PASE_Free(multigrid->aux_A);
    }
    if(multigrid->aux_B) {
      for(i=0; i<multigrid->actual_level; i++) {
	PASE_Aux_matrix_destroy(multigrid->aux_B[i]);
      }
      PASE_Free(multigrid->aux_B);
    }
    if(multigrid->P) {
      for(i=0; i<multigrid->actual_level-1; i++) {
	PASE_Matrix_destroy(multigrid->P[i]);
      }
      PASE_Free(multigrid->P);
    }
    if(multigrid->R) {
      for(i=0; i<multigrid->actual_level-1; i++) {
	PASE_Matrix_destroy(multigrid->R[i]);
      }
      PASE_Free(multigrid->R);
    }
    if(multigrid->A) {
      for(i=1; i<multigrid->actual_level; i++) {
	PASE_Matrix_destroy(multigrid->A[i]);
      }
      PASE_Free(multigrid->A);
    }
    if(multigrid->B) {
      for(i=1; i<multigrid->actual_level; i++) {
	PASE_Matrix_destroy(multigrid->B[i]);
      }
      PASE_Free(multigrid->B);
    }    
    if(multigrid->amg_data) {
      multigrid->ops->destroy_amg_data(multigrid->amg_data);
    }
    if(multigrid->ops) {
      PASE_Multigrid_operator_destroy(multigrid->ops);
    }
    PASE_Free(multigrid);
  }        
}            

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multigrid_operator_assign"
/**
 * @brief 指定多重网格运算集合
 *
 * @param get_amg_array     输入参数, 函数指针, AMG 分层
 * @param destroy_amg_data  输入参数, 函数指针, 销毁 AMG 数据
 *
 * @return PASE_MULTIGRID_OPERATOR
 */
PASE_MULTIGRID_OPERATOR
PASE_Multigrid_operator_assign
(void (*get_amg_array)    (void *A, PASE_PARAMETER param, 
			   void ***A_array, 
			   void ***P_array, 
			   void ***R_array, 
			   PASE_INT *num_level, 
			   void **amg_data),
 void (*destroy_amg_data) (void *amg_data))
{
  PASE_MULTIGRID_OPERATOR ops = (PASE_MULTIGRID_OPERATOR) PASE_Malloc(sizeof(PASE_MULTIGRID_OPERATOR_PRIVATE));
  ops->get_amg_array = get_amg_array;
  ops->destroy_amg_data  = destroy_amg_data;
  return ops;
}

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multigrid_operator_create"
/**
 * @brief 依据指定格式生成多重网格运算集合, 需要在编译时指定可用的格式
 *
 * @param data_form  输入参数, 数据格式类型
 *                   1: PACKAGE_HYPRE
 *                   2: PACKAGE_JXPAMG
 *
 * @return PASE_MULTIGRID_OPERATOR 
 */
PASE_MULTIGRID_OPERATOR
PASE_Multigrid_operator_create(PASE_INT data_form)
{         
  PASE_MULTIGRID_OPERATOR ops = NULL;
  if(data_form == 1){
    ops = PASE_Multigrid_operator_assign(PASE_Multigrid_get_amg_array_hypre,
	PASE_Multigrid_destroy_amg_data_hypre);
  }        
  if(NULL == ops) {
    PASE_Error(__FUNCT__": Cannot find data_form %d.\n", data_form);
  }
  return ops;
}            

#undef  __FUNCT__
#define __FUNCT__ "PASE_Multigrid_operator_destroy"
/**
 * @brief 销毁多重网格运算集合
 *
 * @param ops 输入参数
 */
void         
PASE_Multigrid_operator_destroy(PASE_MULTIGRID_OPERATOR ops)
{            
  PASE_Free(ops);
}            

