# python preprocess.py
# accelerate launch --config_file config/step2.yaml --main_process_port 25671 ratio_compress_pretraining_trainer.py
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/kvlink/tqa_eval.py --run "block_qa" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/kvlink/hqa2_eval.py --run "block_qa" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/kvlink/nq2_eval.py --run "block_qa" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/kvlink/wiki_eval.py --run "block_qa" --ckpt 1122 &

wait