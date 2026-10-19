#!/bin/bash

# First run prepare_feature_banks.py; see REFERENCE_FLOW.md for commands.
# Add --fake_bank_path="<fake_bank.pt>" below to enable interference.
CUDA_VISIBLE_DEVICES=0 python train.py \
        --method=reference_flow \
        --real_bank_path="<real_bank.pt>" \
        --name=for_ltd_test \
        --wang2020_data_path="<data_path>" \
        --checkpoints_dir="<ckpt_dir>" \
        --data_mode="wang2020 or sd1_4"  \
        --arch="CLIP:ViT-L/14"  \
        --lr=0.00005 \
        --fix_backbone \
        --select_k=5 \
        --batch_size=256 \
        --niter=5
