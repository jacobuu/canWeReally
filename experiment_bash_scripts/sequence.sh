printf "Running Mode B test with MI_eeg dataset...\n"

# Test Mode B
python ../dvae.py \
  --dataset MI_eeg \
  --mode-b-frozen-backbone \
  --backbone 'cbramod' \
  --backbone-weights '/home/burger/canWeReally/weights/cbramod_pretrained_weights.pth' \
  --data-file '/home/burger/canWeReally/data/processed_data/MI_eeg_cbramod.pt' \
  --epochs 500 \
  --batch-size 128 \
  --save-dir ../experiments/Mode_B_MI_frozen_backbone_cbramod_500epochs_sameEncSize




printf "Running Mode C test with MI_eeg dataset...\n This means two-stage training: Stage1(Finetune Backbone on T_A) ->  Stage2(DVAE on T_V)\n"

# Test Mode C
python ../dvae.py \
  --dataset MI_eeg \
  --mode-c-two-stage \
  --backbone 'cbramod' \
  --backbone-weights '/home/burger/canWeReally/weights/cbramod_pretrained_weights.pth' \
  --data-file '/home/burger/canWeReally/data/processed_data/MI_eeg_cbramod.pt' \
  --epochs 500 \
  --batch-size 128 \
  --stage1-epochs 10 \
  --save-dir ../experiments/Mode_C_MI_two_stage_cbramod_500epochs_sameEncSize


#!/bin/bash

printf "Running Mode A test with MI_eeg dataset...\n"

# Test Mode A
python ../dvae.py \
  --data-file /home/burger/canWeReally/data/processed_data/MI_eeg.pt \
  --dataset MI_eeg \
  --mode-a-raw-learnable \
  --epochs 500 \
  --batch-size 128 \
  --save-dir ../experiments/Mode_A_MI_raw_linear_cbramod_500epochs_sameEncSize
