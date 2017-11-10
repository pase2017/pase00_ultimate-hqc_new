#ifndef __PASE_MULTIGRID_H__
#define __PASE_MULTIGRID_H__

#include "pase_config.h"
#include "pase_matrix.h"
#include "pase_aux_matrix.h"

typedef struct PASE_MULTIGRID_OPERATOR_PRIVATE_ {
    void (*get_amg_array)    (void *A, PASE_PARAMETER param, void ***A_array, void ***P_array, void ***R_array, PASE_INT *num_level, void **amg_data);
    void (*destroy_amg_data) (void *amg_data);
} PASE_MULTIGRID_OPERATOR_PRIVATE;
typedef PASE_MULTIGRID_OPERATOR_PRIVATE * PASE_MULTIGRID_OPERATOR;

typedef struct PASE_MULTIGRID_PRIVATE_ {
  //PASE_INT max_level;
  PASE_INT actual_level;
  
  PASE_MATRIX *A; // A_0 (细) ---->> A_n (粗)
  PASE_MATRIX *B; // B_0 (细) ---->> B_n (粗)
  
  PASE_MATRIX *P; // 相邻网格层扩张算子 I_{k+1}^{k}, k = 0,\cdots,n-1
  PASE_MATRIX *R; // 相邻网格层限制算子 I_{k}^{k+1}, k = 0,\cdots,n-1
  
  //PASE_MATRIX *LP; // long term prolongation, 某层到最细层的扩张算子
  //PASE_MATRIX *LR; // long term restriction, 最细层到某层的限制算子
  
  PASE_AUX_MATRIX *aux_A; // 辅助空间矩阵
  PASE_AUX_MATRIX *aux_B; // 辅助空间矩阵

  PASE_MULTIGRID_OPERATOR ops; 
  void *amg_data;

} PASE_MULTIGRID_PRIVATE;
typedef PASE_MULTIGRID_PRIVATE * PASE_MULTIGRID;

PASE_MULTIGRID PASE_Multigrid_create(PASE_MATRIX A, PASE_MATRIX B, PASE_PARAMETER param, PASE_MULTIGRID_OPERATOR ops);
void PASE_Multigrid_destroy(PASE_MULTIGRID multigrid);
PASE_MULTIGRID_OPERATOR PASE_Multigrid_operator_create
    (void (*get_amg_array)    (void *A, PASE_PARAMETER param, void ***A_array, void ***P_array, void ***R_array, PASE_INT *num_level, void **amg_data),
     void (*destroy_amg_data) (void *amg_data));
PASE_MULTIGRID_OPERATOR PASE_Multigrid_operator_create_by_default(PASE_INT data_struct);
void PASE_Multigrid_operator_destroy(PASE_MULTIGRID_OPERATOR ops);

///* 不同层向量之间的转移均通过如下函数进行 */
//void PASE_Multigrid_vector_transfer(PASE_MULTIGRID mg, PASE_INT src_level, PASE_VECTOR src_vec, PASE_INT des_level, PASE_VECTOR des_vec);
//void PASE_Multigrid_vector_transfer(PASE_MULTIGRID mg, PASE_INT src_level, PASE_VECTOR src_vec, PASE_INT des_level, PASE_VECTOR des_vec)
//{
//  /**
//   * if(NULL != mu.LP) {
//   *   // 由 LP/LQ 直接做矩阵向量乘法即可得到目标向量
//   * } else {
//   *   // 由 P/Q 逐层做矩阵向量乘法得到目标向量
//   *   }
//   */   
//}


void PASE_Multigrid_get_amg_array_hypre(void *A, PASE_PARAMETER param, void ***A_array, void ***P_array, void ***R_array, PASE_INT *num_level, void **amg_data);
void PASE_Multigrid_destroy_amg_data_hypre(void *amg_data);

void PASE_EigenSolver(PASE_MATRIX *A, PASE_MATRIX *B, PASE_SCALAR *eval, PASE_VECTOR **evec, PASE_INT block_size, PASE_PARAMETER param);

#endif
