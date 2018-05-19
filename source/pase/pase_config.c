#include <stdarg.h>
#include <stdio.h>
#include "pase_config.h"

void PASE_Printf(MPI_Comm comm, char *fmt, ...)
{
  PASE_INT myrank = 0;

#if PASE_USE_MPI
  MPI_Comm_rank(comm, &myrank);
#endif

  if(0 == myrank) {
    va_list vp;
    va_start(vp, fmt);
    vprintf(fmt, vp);
    va_end(vp);
  }

}

void PASE_Error(char *fmt, ...)
{
  PASE_INT myrank = 0;

#if PASE_USE_MPI
  MPI_Comm_rank(PASE_COMM_WORLD, &myrank);
#endif

  if(0 == myrank) {
    printf("PASE ERROR @ ");
    va_list vp;
    va_start(vp, fmt);
    vprintf(fmt, vp);
    va_end(vp);
  }

#if PASE_USE_MPI
  MPI_Abort(PASE_COMM_WORLD, -1);
#else
  exit(-1);
#endif
}

