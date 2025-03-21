#!/bin/bash
pip install scikit-learn matplotlib pandas pytorch-lightning==1.9.0
apt update
apt install wget git curl git-lfs

cd /workspace/RayBNN/python_verify/RSSI2/
#############################
#CNN model
#############################
python3 All_run.py CNNRSSI.py
python3 getresult.py

#############################
#GCN2 model
#############################
python3 All_run.py GCN2RSSI.py
python3 getresult.py

#############################
#LSTM model
#############################
python3 All_run.py LSTMRSSI.py
python3 getresult.py

#############################
#MLP model
#############################
python3 All_run.py MLPRSSI.py
python3 getresult.py

#############################
#BILSTM model
#############################
python3 All_run.py BILSTM.py
python3 getresult.py

