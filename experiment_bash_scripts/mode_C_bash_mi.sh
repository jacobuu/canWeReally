#!/bin/bash

printf "Running Mode C test with MI_eeg dataset...\n This means two-stage training: Stage1(Finetune Backbone on T_A) ->  Stage2(DVAE on T_V)\n"

# Test Mode C
python ../dvae.py \
  --dataset MI_eeg \
  --mode-c-two-stage \
  --backbone 'cbramod' \
  --backbone-weights '/home/burger/canWeReally/weights/cbramod_pretrained_weights.pth' \
  --data-file '/home/burger/canWeReally/data/processed_data/MI_eeg_cbramod.pt' \
  --epochs 100 \
  --batch-size 512 \
  --stage1-epochs 10 \
  --save-dir ../experiments/Mode_C_MI_two_stage_cbramod \
  --run-name Mode_C_MI_2stage_cbra_test


