# python preprocess.py

# !/bin/bash
startTime=$(date +%s) #mark the start of job 

output_dir=$1
run_name=$2

export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"

accelerate launch --config_file config/step2.yaml --main_process_port 25671 ratioaug_pretraining_trainer.py
# accelerate launch --config_file config/step2.yaml --main_process_port 25671 compress_mix_trainer.py --dataset "qa"

# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunk_nopadding/tqa_eval.py --run "chunkcomp_qa" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunk_nopadding/hqa2_eval.py --run "chunkcomp_qa" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunk_nopadding/nq2_eval.py --run "chunkcomp_qa" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunk_nopadding/wiki_eval.py --run "chunkcomp_qa" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/chunk_nopadding_kvlink/tqa_eval.py --run "chunkcomp_qa_link" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunk_nopadding_kvlink/nq2_eval.py --run "chunkcomp_qa_link" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/chunk_nopadding_kvlink/hqa2_eval.py --run "chunkcomp_qa_link" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/chunk_nopadding_kvlink/wiki_eval.py --run "chunkcomp_qa_link" --ckpt 1122 &

# wait
# accelerate launch --config_file config/step2.yaml --main_process_port 25671 ratio_compress_mix_trainer.py --dataset "qa_compress"
# accelerate launch --config_file config/step2.yaml --main_process_port 25671 ratio_compress_mix_trainer.py --dataset "qa_compress_link"