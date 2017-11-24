########################################################################
# Compiler and external dependences
########################################################################
CC        = mpicc
F77       = mpif77
CXX       = mpicxx
F90       = mpif90
HYPRE_DIR = ../../../hypre

########################################################################
# Compiling and linking options
########################################################################
COPTS     = -g -Wall
CINCLUDES = -I$(HYPRE_DIR)/include -I..
CDEFS     = -DHAVE_CONFIG_H -DHYPRE_TIMING
CFLAGS    = $(COPTS) $(CINCLUDES) $(CDEFS)
FOPTS     = -g
FINCLUDES = $(CINCLUDES)
FFLAGS    = $(FOPTS) $(FINCLUDES)
CXXOPTS   = $(COPTS) -Wno-deprecated
CXXINCLUDES = $(CINCLUDES) -I..
CXXDEFS   = $(CDEFS)
IFLAGS_BXX = 
CXXFLAGS  = $(CXXOPTS) $(CXXINCLUDES) $(CXXDEFS) $(IFLAGS_BXX)
IF90FLAGS = 
F90FLAGS = $(FFLAGS) $(IF90FLAGS)


LINKOPTS  = $(COPTS)
LIBS      = ../libpase.a -L$(HYPRE_DIR)/lib -lHYPRE ../lapack-3.4.2/liblapacke.a ../lapack-3.4.2/liblapack.a ../lapack-3.4.2/BLAS/blas_LINUX.a -lm -lgfortran
LFLAGS    = $(LINKOPTS) $(LIBS) -lstdc++
LFLAGS_B =\
 -L${HYPRE_DIR}/lib\
 -lbHYPREClient-C\
 -lbHYPREClient-CX\
 -lbHYPREClient-F\
 -lbHYPRE\
 -lsidl -ldl -lxml2
LFLAGS77 = $(LFLAGS)
LFLAGS90 =

########################################################################
# Rules for compiling the source files
########################################################################
.SUFFIXES: .c .f .cxx .f90

.c.o:
	$(CC) $(CFLAGS) -c $<
.f.o:
	$(F77) $(FFLAGS) -c $<
.cxx.o:
	$(CXX) $(CXXFLAGS) -c $<

########################################################################
# List of all programs to be compiled
########################################################################
ALLPROGS = serPASE_ver01 mv cg lobpcg parPASE_ver01

all: $(ALLPROGS)

default: all

########################################################################
# Example 1 (vector test)
########################################################################
vector_test: vector_test.o
	$(CC) -o vector_test.exe $^ $(LFLAGS)
	rm vector_test.o

########################################################################
# Example 2 (matrix test)
########################################################################
matrix_test: matrix_test.o 
	$(CC) -o matrix_test.exe $^ $(LFLAGS)
	rm matrix_test.o


########################################################################
# Example 3 (Multi-Correction)
########################################################################
parMG: parMG.o
	$(CC) -o parMG.exe $^ $(LFLAGS)
	rm parMG.o


########################################################################
# Example 4 (lixinji)
########################################################################
jiusuo_matrix: jiusuo_matrix.o
	$(CC) -o jiusuo_matrix.exe $^ $(LFLAGS)
	rm jiusuo_matrix.o


########################################################################
# Clean up
########################################################################
clean:
	rm -f $(ALLPROGS:=.o)
	rm -f *\.[0-9] *\.aux
