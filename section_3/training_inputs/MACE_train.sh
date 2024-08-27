#!/bin/bash

echo STARTING AT `date`

source /home/chong/mace_venv/bin/activate

python /home/chong/mace/mace/scripts/run_train.py \
    --name="clust10_increm1" \
    --train_file="clust10_increm1_train.xyz" \
    --valid_file="clust10_increm1_valid.xyz" \
    --test_file="clust10_increm1_test.xyz" \
    --E0s='{14:-7881.32677981007}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=5.5 \
    --batch_size=10 \
    --max_num_epochs=2000 \
    --eval_interval=1 \
    --scheduler_patience=10 \
    --patience=100 \
    --restart_latest \
    --device=cuda \
    --energy_key="free_energy" \
    --ema \
    --swa \
    --start_swa=10000 \
    --num_swa=100 \

echo FINISHED at `date`
