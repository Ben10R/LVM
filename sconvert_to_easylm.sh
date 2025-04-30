#!/bin/bash
#SBATCH --job-name=lvm-job

#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --time=7:00:00

#SBATCH --output=0convert-%j/output.log
#SBATCH --error=0convert-%j/error.log

#SBATCH --chdir=/srv/nfs/logti-hyper-c1/av00440@ens.ad.etsmtl.ca/LVM

nvidia-smi

source .lvmenv/bin/activate  # Activate venv
# Export env variables
export HF_HOME=/srv/nfs/logti-hyper-c1/av00440@ens.ad.etsmtl.ca/LVM/hfcache
export WANDB_API_KEY=
export HF_TOKEN=

python -m EasyLM.models.llama.hf2jax --checkpoint_dir='Emma02/LVM_ckpts' --output_file='./lvm-ckpts/checkpoints'