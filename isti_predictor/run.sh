#!/bin/bash

#CUDA_VISIBLE_DEVICES=6 python train.py -lr 0.00001 -use_wandb True
# CUDA_VISIBLE_DEVICES=0 python train.py -lr 0.00001 -nw 0
python3 train.py -lr 0.00001 -nw 0

