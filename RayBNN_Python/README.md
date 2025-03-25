# RayBNN_Python
Python Bindings for Rust RayBNN



```
module --force purge
module load StdEnv/2023 cuda/12.2 arrayfire/3.9.0 rust/1.85.0 python/3.13.2 openblas


cd RayBNN_Python/
pip3 install -r ./requirements.txt
cd ./Rust_Code/
maturin develop
cd ../Python_Code
python3 ./test_train.py
```

Don't submit jobs to use maturin, just install raybnn_python in the terminal on CCDB