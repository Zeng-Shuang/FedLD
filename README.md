# Tackling Data Heterogeneity in Federated Learning via Loss Decomposition
## Pre-requisites
### Set up environment
```
cd ./FedLD
conda create -n fedld python=3.10
conda activate fedld
pip install -r requirements.txt
```
### Data Preparation
```
cd ./data
```
Download the prepared Retina dataset.
```
gdown https://drive.google.com/uc?id=1bW--_qRZnWbkb0XXvGBCSferdqXZ6pe7
```
Download the prepared COVID-FL dataset.
```
gdown https://drive.google.com/uc?id=1cuvoYvt-EVs5qtA5Xgos0yUJmfPhRbwg
```
## Train and Test
```
cd ./FedLD
python federated_main.py --train_rule FedLD --dataset retina --retina_split 1 --num_users 5 --local_bs 50 --lr 0.01 --epochs 200 --local_epoch 1 --marg_control_loss True --margin_loss_penalty 0.1 --svd True --k_proportion 0.8 --device cuda:0
```
