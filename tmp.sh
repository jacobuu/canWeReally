python dvae.py \
    --data-file 'data/processed_data/MI_eeg_cbramod.pt' \
    --dataset MI_eeg \
    --mode-b-frozen-backbone \
    --backbone cbramod \
    --backbone-weights 'weights/cbramod_pretrained_weights.pth' \
    --save-dir 'experiments/test4_mi_mode_b' \
    --batch-size 512