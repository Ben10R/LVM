#!/bin/bash
#SBATCH --job-name=lvm-train

#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=7:00:00

#SBATCH --output=0from-ckpt-%j/output.log
#SBATCH --error=0from-ckpt-%j/error.log

#SBATCH --chdir=/srv/nfs/logti-hyper-c1/av00440@ens.ad.etsmtl.ca/LVM

nvidia-smi

source .lvmenv/bin/activate
# Export env variables
export HF_HOME=/srv/nfs/logti-hyper-c1/av00440@ens.ad.etsmtl.ca/LVM/hfcache
export WANDB_API_KEY=
export HF_TOKEN=

python -u -m EasyLM.models.llama.llama_train \
    --mesh_dim='1,1,-1' \
    --dtype='bf16' \
    --load_checkpoint='trainstate_params::lvm-ckpts/checkpoints-no-stream' \
    --total_steps=38355 \
    --log_freq=10 \
    --save_model_freq=1000 \
    --save_milestone_freq=2000 \
    --load_llama_config='vqlm_300m' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=1.5e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-5 \
    --optimizer.adamw_optimizer.lr_warmup_steps=8000 \
    --optimizer.adamw_optimizer.lr_decay_steps=288000 \
    --optimizer.accumulate_gradient_steps=4 \
    --train_dataset.type='json' \
    --train_dataset.text_processor.fields='{tokens}' \
    --train_dataset.json_dataset.path='/srv/nfs/logti-hyper-c1/av00440@ens.ad.etsmtl.ca/LVM/dataset.jsonl' \
    --train_dataset.json_dataset.seq_length=4096 \
    --train_dataset.json_dataset.batch_size=16 \
    --train_dataset.json_dataset.tokenizer_processes=8 \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.output_dir='./output/' \
    --logger.wandb_dir='./wandb' \
    --logger.notes='' \
    --logger.experiment_id=$SLURM_JOB_ID