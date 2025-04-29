# python preprocess.py
accelerate launch --config_file config/debug.yaml --main_process_port 25671 chunkaug_ptr_trainer.py
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