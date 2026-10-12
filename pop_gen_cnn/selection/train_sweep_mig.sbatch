#!/bin/bash
#SBATCH --job-name=sweep_full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=3:00:00

REPO=/vast/home/f/fejiroa/projects/Mathieson_Research_Upenn/pop_gen_cnn
PROJ=$REPO/selection
DATA=$REPO/data/final.split.up.seln.big.npz

cd "$PROJ"
mkdir -p logs results

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV=sweepcnn
if ! conda env list | grep -qE "^$ENV\s"; then
    echo "Creating conda env $ENV"
    conda create -y -n "$ENV" python=3.12
fi
conda activate "$ENV"
pip install -q -r requirements.txt

export KERAS_BACKEND=torch

python -u train_sweep_cnn.py --data "$DATA" --out results/run_${SLURM_JOB_ID} --epochs 3 --seed ${SEED:-1}
