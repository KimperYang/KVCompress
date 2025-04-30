conda activate kvm
# cd /data/jingbo_yang/KVCompress

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/anllm_link/nq.py --run "anchor_5_qa_qa_link_10k" --ckpt 1122 --reencode 5 --pos 0 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/anllm_link/nq.py --run "anchor_5_qa_qa_link_10k" --ckpt 1122 --reencode 5 --pos 1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/anllm_link/nq.py --run "anchor_5_qa_qa_link_10k" --ckpt 1122 --reencode 5 --pos 2 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/anllm_link/nq.py --run "anchor_5_qa_qa_link_10k" --ckpt 1122 --reencode 5 --pos 3 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/anllm_link/nq.py --run "anchor_5_freeze_qa_link_10k" --ckpt 561 --reencode 5 --pos 0 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/anllm_link/nq.py --run "anchor_5_freeze_qa_link_10k" --ckpt 561 --reencode 5 --pos 1 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/anllm_link/nq.py --run "anchor_5_freeze_qa_link_10k" --ckpt 561 --reencode 5 --pos 2 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/anllm_link/nq.py --run "anchor_5_freeze_qa_link_10k" --ckpt 561 --reencode 5 --pos 3 &

wait
