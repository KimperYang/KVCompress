conda activate kvm
# cd /data/jingbo_yang/KVCompress

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunk_nopadding_prompt/tqa_eval.py --run "chunkcomp_mix_3k" --ckpt 3000 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunk_nopadding_prompt/hqa2_eval.py --run "chunkcomp_mix_3k" --ckpt 3000 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunk_nopadding_prompt/nq2_eval.py --run "chunkcomp_mix_3k" --ckpt 3000 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunk_nopadding_prompt/wiki_eval.py --run "chunkcomp_mix_3k" --ckpt 3000 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/chunk_nopadding_kvlink_prompt/tqa_eval.py --run "chunkcomp_link_mix_3k" --ckpt 3000 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunk_nopadding_kvlink_prompt/nq2_eval.py --run "chunkcomp_link_mix_3k" --ckpt 3000 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/chunk_nopadding_kvlink_prompt/hqa2_eval.py --run "chunkcomp_link_mix_3k" --ckpt 3000 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/chunk_nopadding_kvlink_prompt/wiki_eval.py --run "chunkcomp_link_mix_3k" --ckpt 3000 &

wait
