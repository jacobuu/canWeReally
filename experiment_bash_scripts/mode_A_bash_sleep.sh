#!/bin/bash

printf "Running Mode A test with sleepedfx dataset...\n"

# Test Mode A
python ../dvae.py \
  --data-file /home/burger/canWeReally/data/processed_data/sleepedfx_cbramod_data.pt \
  --dataset sleepedfx \
  --mode-a-raw-learnable \
  --epochs 100 \
  --batch-size 32 \
  --save-dir ../experiments/test_mode_a


python dvae.py \
  --data-file /home/burger/canWeReally/data/processed_data/MI_eeg_cbramod.pt \
  --dataset sleepedfx \
  --mode-a-raw-learnable \
  --epochs 100 \
  --batch-size 32 \
  --save-dir ../experiments/test_mode_a