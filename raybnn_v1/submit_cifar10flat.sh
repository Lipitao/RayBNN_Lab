#!/bin/bash -l 
#SBATCH --job-name=cifar10_10fold_B  # Name of job 
#SBATCH --account=def-xdong   # adjust this to match the accounting group you are using 
#SBATCH --time=0-00:30           
#SBATCH --cpus-per-task=6         # CPU cores/threads 
#SBATCH --gpus-per-node=a100:1 
#SBATCH --mem=32G 
#SBATCH --output=./out_report/cifar10_flat/%x-%j.out
 
# Load modules 
module load StdEnv/2020 gcc/9.3.0 cuda/12.2 fmt/9.1.0 spdlog/1.9.2 arrayfire/3.9.0 rust/1.70.0 python/3.11.2   

rm -rf /scratch/lain1385/arrayfire/ 
mkdir /scratch/lain1385/arrayfire/ 
 
 
export AF_JIT=20 
 
export AF_JIT_KERNEL_CACHE_DIRECTORY=/scratch/lain1385/arrayfire/ 
 
export AF_OPENCL_MAX_JIT_LEN=$AF_JIT 
 
export AF_CUDA_MAX_JIT_LEN=$AF_JIT 
 
export AF_CPU_MAX_JIT_LEN=$AF_JIT 
 
export AF_DISABLE_GRAPHICS=1 
 
export RUSTFLAGS="-Awarnings -C target-cpu=native" 

export RUSTB 

export RUST_BACKTRACE=1

cd /scratch/lain1385/project/raybnn_v1/s
cargo clean

cargo run --example train_cifar10_flat_10fold
: '
cargo run --example figure2a --release 
cargo run --example figure2b --release 
cargo run --example figure2c --release 
cargo run --example figure2d --release 
cargo run --example figure2e --release 
cargo run --example figure2f --release 
cargo run --example figure3a --release 
cargo run --example figure3b --release 
cargo run --example figure3d --release 
cargo run --example figure4_raybnn --release

 
for i in {0..56}       
do 
    cargo run --example figure6_raybnn2   --release  $i 
done
'
