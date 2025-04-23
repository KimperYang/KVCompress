python preprocess.py
accelerate launch --config_file config/step2.yaml --main_process_port 25671 anchor_trainer.py
# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunk_nopadding_kvlink_prompt/tqa_eval.py --run "compress_chunk_qa_kvlink_nopadding_multichunk20k_epoch2" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunk_nopadding_kvlink_prompt/hqa2_eval.py --run "compress_chunk_qa_kvlink_nopadding_multichunk20k_epoch2" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunk_nopadding_kvlink_prompt/nq2_eval.py --run "compress_chunk_qa_kvlink_nopadding_multichunk20k_epoch2" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunk_nopadding_kvlink_prompt/wiki_eval.py --run "compress_chunk_qa_kvlink_nopadding_multichunk20k_epoch2" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/chunk_nopadding_prompt/tqa_eval.py --run "compress_chunk_qa_nopadding_multichunk20k_epoch2" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/chunk_nopadding_prompt/hqa2_eval.py --run "compress_chunk_qa_nopadding_multichunk20k_epoch2" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/chunk_nopadding_prompt/nq2_eval.py --run "compress_chunk_qa_nopadding_multichunk20k_epoch2" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/chunk_nopadding_prompt/wiki_eval.py --run "compress_chunk_qa_nopadding_multichunk20k_epoch2" --ckpt 1122 &

# wait

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratio_kvlink_prompt/tqa_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratio_kvlink_prompt/hqa2_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratio_kvlink_prompt/nq2_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratio_kvlink_prompt/wiki_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/ratio_prompt/tqa_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/ratio_prompt/hqa2_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/ratio_prompt/nq2_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/ratio_prompt/wiki_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 1122 &

# wait