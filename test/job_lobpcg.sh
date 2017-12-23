#============================================================================================================
EXE="./parMG.exe"
N="1000"
BLOCKSIZE="100"
ATOL="-9"
NSMOOTH="2"
NP="20"

EXEC=${EXE}" -n "${N}" -block_size "${BLOCKSIZE}" -atol "${ATOL}" -nsmooth "${NSMOOTH}
echo ""
echo ${EXEC}

OUTPUT="output/10lobpcg/"
mpirun -np ${NP} ${EXEC} > ${OUTPUT}/np${NP}_n${N}_bs${BLOCKSIZE}_atol1e${ATOL}_nsmooth${NSMOOTH} 2>&1


#============================================================================================================
EXE="./parMG.exe"
N="2000"
BLOCKSIZE="100"
ATOL="-9"
NSMOOTH="2"
NP="20"

EXEC=${EXE}" -n "${N}" -block_size "${BLOCKSIZE}" -atol "${ATOL}" -nsmooth "${NSMOOTH}
echo ""
echo ${EXEC}

OUTPUT="output/10lobpcg/"
mpirun -np ${NP} ${EXEC} > ${OUTPUT}/np${NP}_n${N}_bs${BLOCKSIZE}_atol1e${ATOL}_nsmooth${NSMOOTH} 2>&1


#============================================================================================================
EXE="./parMG.exe"
N="3000"
BLOCKSIZE="100"
ATOL="-9"
NSMOOTH="2"
NP="20"

EXEC=${EXE}" -n "${N}" -block_size "${BLOCKSIZE}" -atol "${ATOL}" -nsmooth "${NSMOOTH}
echo ""
echo ${EXEC}

OUTPUT="output/10lobpcg/"
mpirun -np ${NP} ${EXEC} > ${OUTPUT}/np${NP}_n${N}_bs${BLOCKSIZE}_atol1e${ATOL}_nsmooth${NSMOOTH} 2>&1


#============================================================================================================
EXE="./parMG.exe"
N="4000"
BLOCKSIZE="100"
ATOL="-9"
NSMOOTH="2"
NP="20"

EXEC=${EXE}" -n "${N}" -block_size "${BLOCKSIZE}" -atol "${ATOL}" -nsmooth "${NSMOOTH}
echo ""
echo ${EXEC}

OUTPUT="output/10lobpcg/"
mpirun -np ${NP} ${EXEC} > ${OUTPUT}/np${NP}_n${N}_bs${BLOCKSIZE}_atol1e${ATOL}_nsmooth${NSMOOTH} 2>&1


#============================================================================================================
EXE="./parMG.exe"
N="5000"
BLOCKSIZE="100"
ATOL="-9"
NSMOOTH="2"
NP="20"

EXEC=${EXE}" -n "${N}" -block_size "${BLOCKSIZE}" -atol "${ATOL}" -nsmooth "${NSMOOTH}
echo ""
echo ${EXEC}

OUTPUT="output/10lobpcg/"
mpirun -np ${NP} ${EXEC} > ${OUTPUT}/np${NP}_n${N}_bs${BLOCKSIZE}_atol1e${ATOL}_nsmooth${NSMOOTH} 2>&1





#mpirun -np 20 ./parMG.exe -n 1000 -block_size 100 > output/direct_solve_lobpcg/np20_n1000_bs100_tol10_cg2 2>&1
#mpirun -np 20 ./parMG.exe -n 2000 -block_size 100 > output/direct_solve_lobpcg/np20_n2000_bs100_tol10_cg2 2>&1
#mpirun -np 20 ./parMG.exe -n 3000 -block_size 100 > output/direct_solve_lobpcg/np20_n3000_bs100_tol10_cg2 2>&1
#mpirun -np 20 ./parMG.exe -n 4000 -block_size 100 > output/direct_solve_lobpcg/np20_n4000_bs100_tol10_cg2 2>&1 
#mpirun -np 20 ./parMG.exe -n 5000 -block_size 100 > output/direct_solve_lobpcg/np20_n5000_bs100_tol10_cg2 2>&1
