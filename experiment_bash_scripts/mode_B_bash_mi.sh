#!/bin/bash

printf "Running Mode B test with MI_eeg dataset...\n"

# Test Mode B
python ../dvae.py \
  --dataset MI_eeg \
  --mode-b-frozen-backbone \
  --backbone 'cbramod' \
  --backbone-weights '/home/burger/canWeReally/weights/cbramod_pretrained_weights.pth' \
  --data-file '/home/burger/canWeReally/data/processed_data/MI_eeg_cbramod.pt' \
  --epochs 100 \
  --batch-size 512 \
  --save-dir ../experiments/Mode_B_MI_frozen_backbone_cbramod_test 


