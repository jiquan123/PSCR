#!/bin/bash

for benchmark in AGIQA1K
do  
    echo "HF_ENDPOINT=https://hf-mirror.com python -u train.py --lr=1e-5 --backbone=swin --PS --PS_method=OPS --image_size=224 --log_info=$benchmark   --benchmark=$benchmark "
    HF_ENDPOINT=https://hf-mirror.com python -u train.py --lr=1e-5 --backbone=swin --PS --PS_method=OPS --image_size=224 --log_info=$benchmark   --benchmark=$benchmark 
done



