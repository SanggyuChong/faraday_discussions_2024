#!/bin/bash

echo STARTING AT `date`

python /home/chong/mace_llpr/mace/scripts/run_train.py \
    --name="CG_water" \
    --train_file="CG_water_train_1k_0.xyz" \
    --valid_file="CG_water_val_1k.xyz" \
    --model="MACE" \
    --hidden_irreps="64x0e + 64x1o" \
    --r_max=6.0 \
    --batch_size=10 \
    --max_num_epochs=50 \
    --eval_interval=1 \
    --restart_latest \
    --device="cuda" \
    --ema \
    --correlation=3 \
    --max_ell=3 \
    --gate="tanh" \
    --scaling="rms_forces_scaling" \
    --E0s="average" \
    --loss="forces_only" \
    --seed=20240413 \

echo FINISHED at `date`
