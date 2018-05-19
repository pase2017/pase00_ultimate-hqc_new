include_directories(${PASE_SOURCE_DIR}/include)

set(EXAMPLE_HYPRE_SRC "example_hypre.c")
set(BIN_EXAMPLE_HYPRE_NAME "example_hypre.exe")
set(EXECUTABLE_OUTPUT_PATH ${PASE_SOURCE_DIR}/bin)
add_executable(${BIN_EXAMPLE_HYPRE_NAME} ${EXAMPLE_HYPRE_SRC})
target_link_libraries(${BIN_EXAMPLE_HYPRE_NAME} ${LIB_PASE} ${LIB_HYPRE} ${LIB_LAPACKE} ${LIB_LAPACK} ${LIB_BLAS} "-lm" "-lgfortran")

set(EXAMPLE_USER_SRC "example_user.c")
set(BIN_EXAMPLE_USER_NAME "example_user.exe")
set(EXECUTABLE_OUTPUT_PATH ${PASE_SOURCE_DIR}/bin)
add_executable(${BIN_EXAMPLE_USER_NAME} ${EXAMPLE_USER_SRC})
target_link_libraries(${BIN_EXAMPLE_USER_NAME} ${LIB_PASE} ${LIB_HYPRE} ${LIB_LAPACKE} ${LIB_LAPACK} ${LIB_BLAS} "-lm" "-lgfortran")

