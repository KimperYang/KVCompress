#!/bin/bash
# startTime=$(date +%s) #mark the start of job 

# output_dir=$1
# run_name=$2

# export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"

conda activate kvm

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunkaug_prompt/hqa2_eval.py --run "chunkaug_qa_20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunkaug_prompt/nq2_eval.py --run "chunkaug_qa_20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunkaug_prompt/tqa_eval.py --run "chunkaug_qa_20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunkaug_prompt/wiki_eval.py --run "chunkaug_qa_20k" --ckpt 1122 &


python scripts/evaluation/chunkaug_prompt/tqa_eval.py --run "chunkaug_10_qa_20k" --ckpt 1122
# python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_10_qa_20k" --ckpt 1122 --pos 1
# python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_10_qa_20k" --ckpt 1122 --pos 2
# python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_10_qa_20k" --ckpt 1122 --pos 3
# python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_10_qa_20k" --ckpt 1122 --pos 4
# python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_10_qa_20k" --ckpt 1122 --pos 5
# python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_10_qa_20k" --ckpt 1122 --pos 6
# python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_10_qa_20k" --ckpt 1122 --pos 7
# python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_10_qa_20k" --ckpt 1122 --pos 8
# python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_10_qa_20k" --ckpt 1122 --pos 9

# accelerate launch --config_file config/step2.yaml --main_process_port 25671 chunkaug_ptr_trainer.py