#!/bin/bash

printf "Running Mode F test with MI_eeg dataset but on finetuned weights using the entire training dataset ...\n"

# Test Mode B
python ../dvae.py \
  --dataset MI_eeg \
  --mode-f-classifier-only \
  --backbone 'cbramod' \
  --backbone-weights '/home/burger/canWeReally/experiments/test_jacopo_1/last_model_weights.pt' \
  --data-file '/home/burger/canWeReally/data/processed_data/MI_eeg_cbramod.pt' \
  --epochs 100 \
  --batch-size 128 \
  --save-dir ../experiments/test_jacopo_MLP_simpleHEAD_lr.05_longrun \
  --lr 0.001 \

  # --save-dir ../experiments/Mode_B_MI_frozen_FT_all_backbone_cbra_tEnc_genSameSize_kl.001

