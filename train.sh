# python preprocess.py
accelerate launch --config_file config/step2.yaml --main_process_port 25671 compress_mix_trainer.py --dataset "qa"
accelerate launch --config_file config/step2.yaml --main_process_port 25671 compress_mix_trainer.py --dataset "qa_link"

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunk_nopadding_prompt/tqa_eval.py --run "chunkcomp_mix_qa_3k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunk_nopadding_prompt/hqa2_eval.py --run "chunkcomp_mix_qa_3k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunk_nopadding_prompt/nq2_eval.py --run "chunkcomp_mix_qa_3k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunk_nopadding_prompt/wiki_eval.py --run "chunkcomp_mix_qa_3k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/chunk_nopadding_kvlink_prompt/tqa_eval.py --run "chunkcomp_mix_qa_link_3k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/chunk_nopadding_kvlink_prompt/nq2_eval.py --run "chunkcomp_mix_qa_link_3k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/chunk_nopadding_kvlink_prompt/hqa2_eval.py --run "chunkcomp_mix_qa_link_3k" --ckpt 561 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/chunk_nopadding_kvlink_prompt/wiki_eval.py --run "chunkcomp_mix_qa_link_3k" --ckpt 561 &

wait
# accelerate launch --config_file config/step2.yaml --main_process_port 25671 ratio_compress_mix_trainer.py --dataset "qa_compress"
# accelerate launch --config_file config/step2.yaml --main_process_port 25671 ratio_compress_mix_trainer.py --dataset "qa_compress_link"