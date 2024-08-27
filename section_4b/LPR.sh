#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:1
#SBATCH --mem 32G
#SBATCH --time 12:00:00 

echo STARTING AT `date`

module load cuda

source /home/chong/mts_venv/bin/activate


for buffer in 5.0 3.0 1.0 0.5 0.0

do

python -u lpr_md_refined.py 0.1 $buffer

done



for sig in 0 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5

do

python -u lpr_md_refined.py $sig 0.0

done

