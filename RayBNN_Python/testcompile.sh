#!/bin/bash -l
#SBATCH --job-name=arrayfire   # Name of job
#SBATCH --account=def-taolu    # adjust this to match the accounting group you are using to submit jobs
#SBATCH --time=0-00:14          # 2 hours
#SBATCH --cpus-per-task=6         # CPU cores/threads
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=16G



# Load modules
module --force purge
module load StdEnv/2020 gcc/9.3.0 cuda/12.2 fmt/9.1.0 spdlog/1.9.2 arrayfire/3.9.0 rust/1.70.0 python/3.11.2 openblas


nvidia-smi


cd /scratch/brosnany/
#rm -rf ./magic/
#virtualenv magic
source /scratch/brosnany/magic/bin/activate
rm -rf /scratch/brosnany/Rust_Code/target/
cd /scratch/brosnany/Rust_Code
pip install maturin numpy patchelf
maturin develop
python3 ./example.py
python3 ./run_network.py
rm -rf /scratch/brosnany/Rust_Code/target/
maturin build -r
