#!/bin/bash

#SBATCH --job-name=train_generator
#SBATCH -D .
#SBATCH --output=/data/jingbo_yang/KVMemory/sbatch/O-%x.%j
#SBATCH --error=/data/jingbo_yang/KVMemory/sbatch/E-%x.%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:8           # number of GPUs per node
#SBATCH --cpus-per-task=128         # number of cores per tasks
#SBATCH --time=48:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --signal=SIGUSR1@90     #Signal is sent 90s before time expires
######################
### Set enviroment ###
######################


source /data/jingbo_yang/.bashrc
conda activate kvm
cd /data/jingbo_yang/KVCompress


# accelerate launch --config_file /data/jingbo_yang/.cache/huggingface/accelerate/step2.yaml --main_process_port 25671 compress_pretraining_trainer.py
accelerate launch --config_file config/step2.yaml --main_process_port 25671 compress_pretraining_trainer.py
