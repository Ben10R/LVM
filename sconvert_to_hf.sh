#!/bin/bash
#SBATCH --job-name=lvm-job

#SBATCH --gres=gpu:1
#SBATCH --mem=1G
#SBATCH --time=7:00:00

#SBATCH --output=%j/output.log
#SBATCH --error=%j/error.log

#SBATCH --chdir=/srv/nfs/logti-hyper-c1/av00440@ens.ad.etsmtl.ca/LVM

nvidia-smi
gpu

source .lvmenv/bin/activate  # Activate venv
# Export env variables
export HF_HOME=/srv/nfs/logti-hyper-c1/av00440@ens.ad.etsmtl.ca/LVM/hfcache
export WANDB_API_KEY=
export HF_TOKEN=

python -m EasyLM.models.llama.convert_easylm_to_hf --load_checkpoint='trainstate_params::./output/848/streaming_train_state' --model_size='vqlm_100m' --output_dir='./output-hf/ckpt-848/'