#!/bin/bash

printf "Running Mode A test with MI_eeg dataset...\n"

# Test Mode A
python ../dvae.py \
  --data-file /home/burger/canWeReally/data/processed_data/MI_eeg.pt \
  --dataset MI_eeg \
  --mode-a-raw-learnable \
  --epochs 100 \
  --batch-size 512 \
  --save-dir ../experiments/Mode_A_MI_raw_linear_cbramod


