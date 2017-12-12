#ifndef __PASE_LOBPCG_HYPRE_H__
#define __PASE_LOBPCG_HYPRE_H__

#include <mpi.h>
#include "pase_config.h"
#include "pase_pcg_hypre.h"

#if PASE_USE_HYPRE
#include "_hypre_parcsr_ls.h"
#include "HYPRE_MatvecFunctions.h"
#include "temp_multivector.h"


PASE_INT PASE_Lobpcg_setup_interpreter( mv_InterfaceInterpreter* i);
PASE_INT PASE_Lobpcg_setup_matvec(HYPRE_MatvecFunctions* mv);
PASE_INT PASE_Lobpcg_setup_interpreter_aux( mv_InterfaceInterpreter* i);
PASE_INT PASE_Lobpcg_setup_matvec_aux(HYPRE_MatvecFunctions* mv);

PASE_INT hypre_LOBPCGSetup( void *pcg_vdata, void *A, void *b, void *x );
PASE_INT hypre_LOBPCGSetupB( void *pcg_vdata, void *B, void *x );

#endif
#endif
