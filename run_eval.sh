conda activate kvm
# cd /data/jingbo_yang/KVCompress

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunk_nopadding/tqa_eval.py --run "compress_chunk_qa_nopadding_multichunk30k_epoch2" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunk_nopadding/hqa2_eval.py --run "compress_chunk_qa_nopadding_multichunk30k_epoch2" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunk_nopadding/nq2_eval.py --run "compress_chunk_qa_nopadding_multichunk30k_epoch2" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunk_nopadding/wiki_eval.py --run "compress_chunk_qa_nopadding_multichunk30k_epoch2" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/upper/tqa_upper.py --run "upper_2epoch" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/upper/nq2_upper.py --run "upper_2epoch" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/upper/hqa2_upper.py --run "upper_2epoch" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/upper/wiki_upper.py --run "upper_2epoch" --ckpt 1122 &
wait