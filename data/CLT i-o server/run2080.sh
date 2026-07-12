# !/bin/bash
# sh run.sh

source ~/.bashrc
mpif90fftwopenacc tokRZ_mpi_pll_acc_merged.f90 -o case04.out
nohup mpirun -np 1 --bind-to socket --mca oob_tcp_if_include lo case04.out &
MPI_PID=$!
wait $MPI_PID

chmod +x ./dia_main.out
mpirun_intel -n 10 ./dia_main.out

# mpiifortfftw wsdiagncd_allin2_mnmode_outBrmn.f90 -o dia_Bmn.out
chmod +x ./dia_Bmn.out
mpirun_intel -n 10 ./dia_Bmn.out

# for d in */ ;
#     do if [ -f "$d/run.sh" ];
#         then echo "正在進入目錄 $d 並啟動腳本...";
#             # 在子進程中：進入目錄 -> 修復 \r -> 啟動 nohup
#             (cd "$d" && sed -i 's/\r$//' run.sh && nohup sh run.sh > nohup.out 2>&1 &);
#         fi;
# done
