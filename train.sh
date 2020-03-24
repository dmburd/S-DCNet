#!/bin/bash

train_A ()
{
    ./train.py \
        --dataset-rootdir ./ShanghaiTech \
        --part A \
        --densmaps-gt-npz ./my_gener_gt_density_maps_npz_SAVED/density_maps_part_A_\*.npz \
        --train-val-split 0.9 \
        --num-epochs 1000 \
        --num-intervals 22 
}
# ^ --num-intervals: 22 for part_A
# --pretrained-model ./run_part_A_from_scratch/checkpoints/epoch_1000.pth

train_B ()
{
    ./train.py \
        --dataset-rootdir ./ShanghaiTech \
        --part B \
        --densmaps-gt-npz ./my_gener_gt_density_maps_npz_SAVED/density_maps_part_B_\*.npz \
        --train-val-split 0.9 \
        --num-epochs 1000 \
        --num-intervals 7 
}
# ^ --num-intervals: 7 for part_B
# --pretrained-model ./run_part_B_from_scratch/checkpoints/epoch_1000.pth

train_A
# train_A or train_B
