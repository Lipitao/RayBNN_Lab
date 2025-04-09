#!/bin/bash -l 
#SBATCH --job-name=mnist  # Name of job 
#SBATCH --account=def-xdong   # adjust this to match the accounting group you are using 
#SBATCH --time=0-00:30           
#SBATCH --cpus-per-task=6         # CPU cores/threads 
#SBATCH --gpus-per-node=a100:1 
#SBATCH --mem=32G 
#SBATCH --output=./out_report/mnist/%x-%j.out
 
# Load modules 
module load StdEnv/2020 gcc/9.3.0 cuda/12.2 fmt/9.1.0 spdlog/1.9.2 arrayfire/3.9.0 rust/1.70.0 python/3.11.2   

# modify the following line to match your directory
rm -rf /scratch/lain1385/arrayfire/ 
mkdir /scratch/lain1385/arrayfire/ 
 
 
export AF_JIT=20 
 
 # modify the following line to match your directory
export AF_JIT_KERNEL_CACHE_DIRECTORY=/scratch/lain1385/arrayfire/ 
 
export AF_OPENCL_MAX_JIT_LEN=$AF_JIT 
 
export AF_CUDA_MAX_JIT_LEN=$AF_JIT 
 
export AF_CPU_MAX_JIT_LEN=$AF_JIT 
 
export AF_DISABLE_GRAPHICS=1 
 
export RUSTFLAGS="-Awarnings -C target-cpu=native" 
# export RUST_BACKTRACE=1

# modify the following line to match your directory
cd /home/lain1385/scratch/project/RayBNN_Lab/raybnn_v1

cargo clean
cargo run --example train_mnist