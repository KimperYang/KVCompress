# python preprocess.py
# accelerate launch --config_file config/step2.yaml --main_process_port 25671 anchor_ptr_trainer.py
# accelerate launch --config_file config/step2.yaml --main_process_port 25671 anchor_trainer.py --dataset "qa"
# accelerate launch --config_file config/step2.yaml --main_process_port 25671 ratio_compress_qa_trainer.py --dataset "qa_compress"

accelerate launch --config_file config/4gpu_step8.yaml --main_process_port 25671 ratio_compress_qa_sum5.py --reencode 5
accelerate launch --config_file config/4gpu_step8.yaml --main_process_port 25671 ratio_compress_qa_sum5.py --reencode 0

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_mix_reencode_0" --ckpt 6000 --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_mix_reencode_0" --ckpt 6000 --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_mix_reencode_0" --ckpt 6000 --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_mix_reencode_0" --ckpt 6000 --pos 3 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_mix_reencode_0" --ckpt 6000 --pos 4 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_mix_reencode_0" --ckpt 6000 --pos 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_mix_reencode_0" --ckpt 6000 --pos 6 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_mix_reencode_0" --ckpt 6000 --pos 7 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_mix_reencode_0" --ckpt 6000 --pos 8 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/ratioaug_prompt/nq.py --run "ratioaug_25_mix_reencode_0" --ckpt 6000 --pos 9 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_25_mix_reencode_5" --ckpt 6000 --pos 8 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_25_mix_reencode_5" --ckpt 6000 --pos 9 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_25_mix_reencode_5" --ckpt 6000 --pos 4 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_25_mix_reencode_5" --ckpt 6000 --pos 5 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_25_mix_reencode_5" --ckpt 6000 --pos 6 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_25_mix_reencode_5" --ckpt 6000 --pos 7 &

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_25_mix_reencode_5" --ckpt 6000 --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_25_mix_reencode_5" --ckpt 6000 --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_25_mix_reencode_5" --ckpt 6000 --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratioaug_link_prompt/nq.py --run "ratioaug_25_mix_reencode_5" --ckpt 6000 --pos 3 &