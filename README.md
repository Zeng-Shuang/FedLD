# FedLD (Tackling Data Heterogeneity in Federated Learning via Loss Decomposition)
Official implementation of "Tackling Data Heterogeneity in Federated Learning via Loss Decomposition" (MICCAI 2024) https://arxiv.org/pdf/2408.12300

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
## Citation
If you find our code or paper useful, please consider citing:
```
@article{zeng2024tackling,
  title={Tackling Data Heterogeneity in Federated Learning via Loss Decomposition},
  author={Zeng, Shuang and Guo, Pengxin and Wang, Shuai and Wang, Jianbo and Zhou, Yuyin and Qu, Liangqiong},
  journal={arXiv preprint arXiv:2408.12300},
  year={2024}
}
```
