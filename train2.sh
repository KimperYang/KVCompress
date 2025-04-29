# python preprocess.py
# accelerate launch --config_file config/step2.yaml --main_process_port 25671 anchor_ptr_trainer.py
accelerate launch --config_file config/step2.yaml --main_process_port 25671 anchor_trainer.py --dataset "qa"
accelerate launch --config_file config/step2.yaml --main_process_port 25671 anchor_trainer.py --dataset "qa_link"
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/anllm/hqa.py --run "training_res/anchor_continue_qa_10k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/anllm/musique.py --run "training_res/anchor_continue_qa_10k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/anllm/nq.py --run "training_res/anchor_continue_qa_10k" --ckpt 561 --pos 0 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/anllm/wiki.py --run "training_res/anchor_continue_qa_10k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/anllm_link/hqa.py --run "training_res/anchor_continue_qa_link_10k" --ckpt 561 --reencode 5&
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/anllm_link/musique.py --run "training_res/anchor_continue_qa_link_10k" --ckpt 561 --reencode 5&
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/anllm_link/nq.py --run "training_res/anchor_continue_qa_link_10k" --ckpt 561 --pos 0 --reencode 5&
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/anllm_link/wiki.py --run "training_res/anchor_continue_qa_link_10k" --ckpt 561 --reencode 5&

# wait

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratio_kvlink_prompt/tqa_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratio_kvlink_prompt/hqa2_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratio_kvlink_prompt/nq2_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratio_kvlink_prompt/wiki_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/ratio_prompt/tqa_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/ratio_prompt/hqa2_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/ratio_prompt/nq2_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 561 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/ratio_prompt/wiki_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 561 &

# wait