# python preprocess.py
accelerate launch --config_file config/step2.yaml --main_process_port 25671 anchor_trainer.py --dataset "qa"
accelerate launch --config_file config/step2.yaml --main_process_port 25671 anchor_trainer.py --dataset "qa_link"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/anllm/tqa_eval.py --run "training_res/anchor_continue_qa_10k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/anllm/hqa2_eval.py --run "training_res/anchor_continue_qa_10k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/anllm/nq2_eval.py --run "training_res/anchor_continue_qa_10k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/anllm/wiki_eval.py --run "training_res/anchor_continue_qa_10k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/anllm_link/tqa_eval.py --run "training_res/anchor_continue_qa_link_10k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/anllm_link/hqa2_eval.py --run "training_res/anchor_continue_qa_link_10k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/anllm_link/nq2_eval.py --run "training_res/anchor_continue_qa_link_10k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/anllm_link/wiki_eval.py --run "training_res/anchor_continue_qa_link_10k" --ckpt 561 &

wait

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratio_kvlink_prompt/tqa_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratio_kvlink_prompt/hqa2_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratio_kvlink_prompt/nq2_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratio_kvlink_prompt/wiki_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/ratio_prompt/tqa_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/ratio_prompt/hqa2_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/ratio_prompt/nq2_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/ratio_prompt/wiki_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 561 &

# wait