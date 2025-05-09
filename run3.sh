#!/bin/bash
# startTime=$(date +%s) #mark the start of job 

# output_dir=$1
# run_name=$2

# export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"

# conda activate kvm

# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunkaug_prompt/hqa2_eval.py --run "chunkaug_qa_link_20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunkaug_prompt/nq2_eval.py --run "chunkaug_qa_link_20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/chunkaug_prompt/tqa_eval.py --run "chunkaug_qa_link_20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/chunkaug_prompt/wiki_eval.py --run "chunkaug_qa_link_20k" --ckpt 561 &

CUDA_VISIBLE_DEVICES=6
python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561 --pos 0
python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561 --pos 1
python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561 --pos 2
python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561 --pos 3
python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561 --pos 4
python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561 --pos 5
python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561 --pos 6
python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561 --pos 7
python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561 --pos 8
python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561 --pos 9
python scripts/evaluation/ratioaug_link_prompt/hqa.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561
python scripts/evaluation/ratioaug_link_prompt/wiki_eval.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561
python scripts/evaluation/ratioaug_link_prompt/tqa_eval.py --run "ratioaug_10_qa_compress_link_continue" --ckpt 561

# accelerate launch --config_file config/step2.yaml --main_process_port 25671 chunkaug_link_trainer.py