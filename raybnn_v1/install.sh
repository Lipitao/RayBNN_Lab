#!/bin/bash


cd /workspace/

apt update

DEBIAN_FRONTEND=noninteractive apt dist-upgrade -y

ldconfig

DEBIAN_FRONTEND=noninteractive apt install -y build-essential git cmake libfreeimage-dev wget cmake-curses-gui libopenblas-dev libfftw3-dev liblapacke-dev opencl-headers libboost-all-dev ocl-icd-opencl-dev libglfw3-dev libfontconfig1-dev curl  git-lfs  libspdlog-dev


rm -rf ./arrayfire/

git clone --recursive https://github.com/arrayfire/arrayfire.git -b master

cd arrayfire

git checkout d2a6636

mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)

make install

ldconfig



cd /workspace/

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh  -s -- -y

source "$HOME/.cargo/env"

source $HOME/.cargo/env




export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/' >> ~/.bashrc 

export AF_JIT=20

echo 'export AF_JIT=20' >> ~/.bashrc 

export AF_JIT_KERNEL_CACHE_DIRECTORY=$HOME/.arrayfire

echo 'export AF_JIT_KERNEL_CACHE_DIRECTORY=$HOME/.arrayfire' >> ~/.bashrc 

export AF_OPENCL_MAX_JIT_LEN=$AF_JIT

echo 'export AF_OPENCL_MAX_JIT_LEN=$AF_JIT' >> ~/.bashrc 

export AF_CUDA_MAX_JIT_LEN=$AF_JIT

echo 'export AF_CUDA_MAX_JIT_LEN=$AF_JIT' >> ~/.bashrc 

export AF_CPU_MAX_JIT_LEN=$AF_JIT

echo 'export AF_CPU_MAX_JIT_LEN=$AF_JIT' >> ~/.bashrc 

export AF_DISABLE_GRAPHICS=1

echo 'export AF_DISABLE_GRAPHICS=1' >> ~/.bashrc 

export RUSTFLAGS="-Awarnings -C target-cpu=native"

echo 'export RUSTFLAGS="-Awarnings -C target-cpu=native"' >> ~/.bashrc 




source ~/.cargo/env

source ~/.bashrc


cd /tmp/
git lfs install

git clone https://huggingface.co/datasets/AlcalaDataset/AlcalaData   

cd /tmp/AlcalaData 


mkdir /workspace/RayBNN/test_data/
cp ./* /workspace/RayBNN/test_data/
cp ./* /workspace/RayBNN/python_verify/RSSI2/
cp ./meanY.dat  /workspace/RayBNN/matlab_plot/meanY.csv
cp ./stdY.dat  /workspace/RayBNN/matlab_plot/stdY.csv

