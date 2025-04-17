conda activate kvm
# cd /data/jingbo_yang/KVCompress

# CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunk_nopadding/tqa_eval.py --run "chunkcomp_qa" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunk_nopadding/hqa2_eval.py --run "chunkcomp_qa" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunk_nopadding/nq2_eval.py --run "chunkcomp_qa" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunk_nopadding/wiki_eval.py --run "chunkcomp_qa" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/chunk_nopadding_kvlink/tqa_eval.py --run "chunkcomp_qa_link" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/chunk_nopadding_kvlink/nq2_eval.py --run "chunkcomp_qa_link" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/chunk_nopadding_kvlink/hqa2_eval.py --run "chunkcomp_qa_link" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/chunk_nopadding_kvlink/wiki_eval.py --run "chunkcomp_qa_link" --ckpt 1122 &

wait
