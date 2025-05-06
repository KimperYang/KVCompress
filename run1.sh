#!/bin/bash
startTime=$(date +%s) #mark the start of job 

output_dir=$1
run_name=$2

export WANDB_API_KEY="297fefc6714432e38b47736829a56f96e540206a"

python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_qa_compress" --ckpt 1122 --pos 0
python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_qa_compress" --ckpt 1122 --pos 1
python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_qa_compress" --ckpt 1122 --pos 2
python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_qa_compress" --ckpt 1122 --pos 3
python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_qa_compress" --ckpt 1122 --pos 4
python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_qa_compress" --ckpt 1122 --pos 5
python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_qa_compress" --ckpt 1122 --pos 6
python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_qa_compress" --ckpt 1122 --pos 7
python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_qa_compress" --ckpt 1122 --pos 8
python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_qa_compress" --ckpt 1122 --pos 9
python scripts/evaluation/ratioaug_prompt/hqa.py --run "ratioaug_25_qa_compress" --ckpt 1122
python scripts/evaluation/ratioaug_prompt/wiki_eval.py --run "ratioaug_25_qa_compress" --ckpt 1122
python scripts/evaluation/ratioaug_prompt/tqa_eval.py --run "ratioaug_25_qa_compress" --ckpt 1122