#ifndef __PASE_MULTIGRID_HYPRE_H__
#define __PASE_MULTIGRID_HYPRE_H__

#include "pase_config.h"
#include "pase_param.h"

#if PASE_USE_HYPRE

void PASE_Multigrid_get_amg_array_hypre
    (void *A, PASE_PARAMETER param, 
     void ***A_array, 
     void ***P_array, 
     void ***R_array, 
     PASE_INT *num_level, 
     void **amg_data);
void PASE_Multigrid_destroy_amg_data_hypre(void *amg_data);

#endif

#endif
