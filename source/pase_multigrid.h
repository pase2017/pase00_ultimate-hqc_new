#ifndef __PASE_MULTIGRID_H__
#define __PASE_MULTIGRID_H__

#include "pase_config.h"
#include "pase_matrix.h"
#include "pase_aux_matrix.h"
#include "pase_param.h"

/* 多重网格计算集合 */
typedef struct PASE_MULTIGRID_OPERATOR_PRIVATE_ {
  /**
   * @brief AMG 分层
   *
   * @param A          输入参数, 用于 AMG 分层的矩阵                 
   * @param param      输入参数, 其中包含 AMG 分层的各个参数
   * @param A_array    输出参数, AMG 分层得到的矩阵序列, A_array[0] 为最细, A_array[num_level-1] 为最粗, 
   *                   A_array[1] = R_array[0] * A_array[0] * P_array[0], 以此类推
   * @param P_array    输出参数, AMG 分层得到的投影矩阵序列 
   * @param R_array    输出参数, AMG 分层得到的限制矩阵序列 
   * @param num_level  输出参数, AMG 分层得到的层数
   * @param amg_data   输出参数, AMG 分层得到的存储 A_array, P_array, R_array 等数据的结构体, 用于之后销毁数据
   */
  void (*get_amg_array)    (void *A, PASE_PARAMETER param, void ***A_array, void ***P_array, void ***R_array, PASE_INT *num_level, void **amg_data);
  /**
   * @brief AMG 数据的销毁
   *
   * @param amg_data  输入参数, 指向 AMG 数据的指针
   */
  void (*destroy_amg_data) (void *amg_data);
} PASE_MULTIGRID_OPERATOR_PRIVATE;
typedef PASE_MULTIGRID_OPERATOR_PRIVATE * PASE_MULTIGRID_OPERATOR;

typedef struct PASE_MULTIGRID_PRIVATE_ {
  //PASE_INT max_level;
  PASE_INT actual_level;

  PASE_MATRIX *A;         // A_0 (细) ---->> A_n (粗)
  PASE_MATRIX *B;         // B_0 (细) ---->> B_n (粗)

  PASE_MATRIX *P;         // 相邻网格层扩张算子 I_{k+1}^{k}, k = 0,\cdots,n-1
  PASE_MATRIX *R;         // 相邻网格层限制算子 I_{k}^{k+1}, k = 0,\cdots,n-1

  //PASE_MATRIX *LP;      // long term prolongation, 某层到最细层的扩张算子
  //PASE_MATRIX *LR;      // long term restriction, 最细层到某层的限制算子

  PASE_AUX_MATRIX *aux_A; // 辅助空间矩阵
  PASE_AUX_MATRIX *aux_B; // 辅助空间矩阵

  PASE_MULTIGRID_OPERATOR ops; 
  void *amg_data;

} PASE_MULTIGRID_PRIVATE;
typedef PASE_MULTIGRID_PRIVATE * PASE_MULTIGRID;

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
PASE_MULTIGRID PASE_Multigrid_create(PASE_MATRIX A, PASE_MATRIX B, PASE_PARAMETER param, PASE_MULTIGRID_OPERATOR ops);

/**
 * @brief AMG 分层
 *
 * @param multigrid  输入/输出参数, 其中各个指针成员已申请好空间
 * @param A          输入参数
 * @param B          输入参数
 * @param param      输入参数, 包含 AMG 分层的各个参数
 */
void PASE_Multigrid_set_up(PASE_MULTIGRID multigrid, PASE_MATRIX A,  PASE_MATRIX B, PASE_PARAMETER param);

/**
 * @brief 销毁 PASE_MULTIGRID
 *
 * @param multigrid 输入参数
 */
void PASE_Multigrid_destroy(PASE_MULTIGRID multigrid);

/**
 * @brief 指定多重网格运算集合
 *
 * @param get_amg_array     输入参数, 函数指针, AMG 分层
 * @param destroy_amg_data  输入参数, 函数指针, 销毁 AMG 数据
 *
 * @return PASE_MULTIGRID_OPERATOR
 */
PASE_MULTIGRID_OPERATOR PASE_Multigrid_operator_assign
(void (*get_amg_array)    (void *A, PASE_PARAMETER param, 
			   void ***A_array, 
			   void ***P_array, 
			   void ***R_array, 
			   PASE_INT *num_level, 
			   void **amg_data),
 void (*destroy_amg_data) (void *amg_data));

/**
 * @brief 依据指定格式生成多重网格运算集合, 需要在编译时指定可用的格式
 *
 * @param data_form  输入参数, 数据格式类型
 *                   1: PACKAGE_HYPRE
 *                   2: PACKAGE_JXPAMG
 *
 * @return PASE_MULTIGRID_OPERATOR 
 */
PASE_MULTIGRID_OPERATOR PASE_Multigrid_operator_create(PASE_INT data_form);

/**
 * @brief 销毁多重网格运算集合
 *
 * @param ops 输入参数
 */
void PASE_Multigrid_operator_destroy(PASE_MULTIGRID_OPERATOR ops);

#endif
