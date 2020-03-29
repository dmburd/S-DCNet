#!/bin/bash

inference_A ()
{
    ./inference.py \
        --checkpoint ./run_part_A_from_scratch_googlecolab/checkpoints/epoch_0310_selected.pth \
        --images-dir ./dir_for_inference_A \
        --export yes
}

inference_B ()
{
    ./inference.py \
        --checkpoint ./run_part_B_from_scratch_googlecolab/checkpoints/epoch_0430_selected.pth \
        --images-dir ./dir_for_inference_B \
        --export yes
}

inference_A
