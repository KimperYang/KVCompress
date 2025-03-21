conda activate kvm
# cd /data/jingbo_yang/KVCompress

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratio_kvlink/tqa_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/ratio_kvlink/hqa2_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratio_kvlink/nq2_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratio_kvlink/wiki_eval.py --run "ratio_compress_qa_kvlink_multichunk20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/upper/tqa_upper.py --run "upper_2epoch" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/upper/nq2_upper.py --run "upper_2epoch" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/upper/hqa2_upper.py --run "upper_2epoch" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/upper/wiki_upper.py --run "upper_2epoch" --ckpt 1122 &
wait