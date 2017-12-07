#ifndef __PASE_CONFIG_H__
#define __PASE_CONFIG_H__

//=============================================================================
/* 基本数据类型的封装 */
typedef int    PASE_INT;
typedef double PASE_DOUBLE;
typedef double PASE_REAL;

#ifdef PASE_USE_COMPLEX
typedef complex PASE_SCALAR;
typedef complex PASE_COMPLEX;
#else
typedef double  PASE_SCALAR;
#endif

//=============================================================================
/* 是否使用 HYPRE 软件包 （目前默认包含) */
#define USE_HYPRE 

#ifdef USE_HYPRE
#define PASE_USE_HYPRE 1 
#else
#define PASE_USE_HYPRE 0 
#endif

#define PASE_USE_JXPAMG 0

//=============================================================================
typedef enum { CLJP = 1, FALGOUT = 2, PMHIS = 3 } PASE_COARSEN_TYPE;
typedef enum { PACKAGE_HYPRE = 1, PACKAGE_JXPAMG = 2 } EXTERNAL_PACKAGE;

//=============================================================================
#include <stdlib.h>

#define PASE_Malloc  malloc
#define PASE_Free(a) { free(a); a = NULL; }

//=============================================================================
#define PASE_NO    0
#define PASE_YES   1
#define PASE_USER -1

//=============================================================================
#include "mpi.h"

#define PASE_COMM_WORLD MPI_COMM_WORLD
#define PASE_COMM_SELF  MPI_COMM_SELF

void PASE_Error(char *fmt, ...);
void PASE_Printf(MPI_Comm comm, char *fmt, ...);

#endif
