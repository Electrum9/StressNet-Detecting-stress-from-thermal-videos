#!/bin/bash

#CUDA_VISIBLE_DEVICES=6 python train.py -lr 0.00001 -use_wandb True
CUDA_VISIBLE_DEVICES=1 python train.py -lr 0.00001 -nw 0

