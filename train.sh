#!/bin/bash

# First run prepare_feature_banks.py; see REFERENCE_FLOW.md for commands.
# This entry trains FM first, then freezes the field and trains the readout.
CUDA_VISIBLE_DEVICES=0 python train.py \
        --method=reference_flow \
        --real_bank_path="<real_bank.pt>" \
        --name=reference_flow_two_stage \
        --wang2020_data_path="<data_path>" \
        --checkpoints_dir="<ckpt_dir>" \
        --data_mode=sd1_4 \
        --arch="CLIP:ViT-L/14"  \
        --lr=0.00005 \
        --feature_layers 11 12 13 14 15 \
        --flow_steps=4 \
        --flow_epochs=5 \
        --batch_size=256 \
        --niter=5
