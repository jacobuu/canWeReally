#!/bin/bash

printf "Running Mode B test with MI_eeg dataset but on finetuned weights using the entire training dataset ...\n"

# Test Mode B
python ../dvae.py \
  --dataset MI_eeg \
  --mode-b-frozen-backbone \
  --backbone 'cbramod' \
  --backbone-weights '/home/burger/canWeReally/weights/finetuned/epoch40_acc_0.62291_kappa_0.49715_f1_0.62318.pth' \
  --data-file '/home/burger/canWeReally/data/processed_data/MI_eeg_cbramod.pt' \
  --epochs 100 \
  --batch-size 512 \
  --save-dir ../experiments/tmp_test

  # --save-dir ../experiments/Mode_B_MI_frozen_FT_all_backbone_cbra_tEnc_genSameSize_kl.001


