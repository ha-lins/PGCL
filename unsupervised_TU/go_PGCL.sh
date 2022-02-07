##!/bin/bash -ex
#
#for seed in 0 1 2 3 4
#do
#  CUDA_VISIBLE_DEVICES=$1 python gsimclr_lyx.py --DS $2 --lr 0.01 --local \
#  --num-gc-layers 3 --aug $3 --weak_aug2 $4  --seed $seed
#
#done

for seed in 2 3 4 0 1
do
  CUDA_VISIBLE_DEVICES=$1 python pgcl_main.py --DS $2 --local \
   --aug $3 --weak_aug2 $4 --num-gc-layers $5 --lr $6 --epochs $7 --nmb_prototypes $8 --seed $seed
done
