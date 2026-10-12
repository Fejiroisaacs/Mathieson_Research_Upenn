#!/bin/bash
#SBATCH --job-name=sweep_cnn
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# ---- edit these two paths ----
REPO=/vast/home/f/fejiroa/projects/Mathieson_Research_Upenn/pop_gen_cnn
PROJ=$REPO/selection
DATA=$REPO/data/final.split.up.seln.big.npz
# ------------------------------

cd "$PROJ"
mkdir -p logs results

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sweepcnn
pip install -q -r requirements.txt 

export KERAS_BACKEND=torch
nvidia-smi
python -c "import torch, keras; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('keras', keras.__version__)"

# Quick smoke test first?  add:  --max-chunks 2 --epochs 1
python -u train_sweep_cnn.py --data "$DATA" --out results/run_${SLURM_JOB_ID} --epochs 3
