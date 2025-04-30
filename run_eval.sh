conda activate kvm
# cd /data/jingbo_yang/KVCompress

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/anllm/tqa.py --run "anchor_5_qa_qa_10k" --ckpt 1122  &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/anllm/hqa.py --run "anchor_5_qa_qa_10k" --ckpt 1122   &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/anllm/musique.py --run "anchor_5_qa_qa_10k" --ckpt 1122   &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/anllm/wiki.py --run "anchor_5_qa_qa_10k" --ckpt 1122   &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/anllm_link/tqa.py --run "anchor_5_freeze_qa_link_10k" --ckpt 561 --reencode 5 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/anllm_link/hqa.py --run "anchor_5_freeze_qa_link_10k" --ckpt 561 --reencode 5  &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/anllm_link/musique.py --run "anchor_5_freeze_qa_link_10k" --ckpt 561 --reencode 5  &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/anllm_link/wiki.py --run "anchor_5_freeze_qa_link_10k" --ckpt 561 --reencode 5  &

wait
