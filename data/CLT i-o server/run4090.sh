# !/bin/bash
# sh run.sh

source ~/.bashrc
mpif90fftwopenacc tokRZ_mpi_pll_acc_merged.f90 -o case04.out
nohup mpirun -np 1 --bind-to socket --mca oob_tcp_if_include lo case04.out &
MPI_PID=$!
wait $MPI_PID

source /opt/intel/oneapi/setvars.sh --force
# mpiifortfftw wsdiagncd_allin2_mnmode.f90 -o dia_main.out
chmod +x ./dia_main.out
mpirun -n 10 ./dia_main.out

# mpiifortfftw wsdiagncd_allin2_mnmode_outBrmn.f90 -o dia_Bmn.out
chmod +x ./dia_Bmn.out
mpirun -n 10 ./dia_Bmn.out

# for d in */ ;
#     do if [ -f "$d/run4090.sh" ];
#         then echo "Entering directory $d and starting the script...";
#             # In the subprocess: enter the directory, remove \r, and start nohup.
#             (cd "$d" && sed -i 's/\r$//' run4090.sh && nohup sh run4090.sh > nohup.out 2>&1 &);
#         fi;
# done
