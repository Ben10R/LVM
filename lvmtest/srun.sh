#!/bin/bash
#SBATCH --job-name=lvm-job

#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --time=7:00:00

#SBATCH --output=%j/output.log
#SBATCH --error=%j/error.log

#SBATCH --chdir=/mnt/home/av00440@ens.ad.etsmtl.ca/lvmtest

nvidia-smi

source .lvmenv/bin/activate  # Activate venv
# Export env variables
export HF_HOME=/mnt/home/av00440@ens.ad.etsmtl.ca/lvmtest/hfcache
export WANDB_API_KEY=
export HF_TOKEN=

python cmd_app.py --checkpoint="Ben10x/lvm_100m" --input_dir='prompt' --output_dir='output/'
