conda activate kvm
# cd /data/jingbo_yang/KVCompress

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunkaug_prompt/hqa.py --run "chunkaug_25_qa" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunkaug_prompt/musique.py --run "chunkaug_25_qa" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunkaug_prompt/tqa_eval.py --run "chunkaug_25_qa" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunkaug_prompt/wiki_eval.py --run "chunkaug_25_qa" --ckpt 1122 &