 #!/usr/bin/env bash 
module purge

#GCC 
#module load compiler/gcc/11.2.0
module load compiler/gcc/9.3.0

#CMake
module load build/cmake/3.15.3

#mkl
module load linalg/mkl/2020_update4

#Cuda
#module load compiler/cuda/10.0
#module load compiler/cuda/10.2
#module load compiler/cuda/11.2
module load compiler/cuda/11.2

#Hwloc
module load hardware/hwloc/2.5.0

#Trace
module load trace/fxt/0.3.13

#StarPU
module load runtime/starpu/1.3.8/mpi-cuda-fxt
#module load runtime/starpu/1.3.8/mpi-cuda

#CUDNN
#module load dnn/cudnn/11.2-v8.1.1.33
#module load dnn/cudnn/9.0-v7.1
#module load dnn/cudnn/10.0-v7.5.0
#module load dnn/cudnn/10.0-v7.6.4.38
module load dnn/cudnn/11.2-v8.1.1.33         

#Python3 for Gurobi
module load language/python/3.9.6

#Gurobi
module load tools/gurobi/9.1.2

#https://gitlab.inria.fr/sed-bso/datacenter/-/blob/master/2021_11_mdb_energy_scope/2021_11_mdb_energy_scope.md
module load trace/energy_scope
#energy_scope_slurm.sh sleep 15
