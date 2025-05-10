conda activate kvm
# cd /data/jingbo_yang/KVCompress

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_25_qa_8gpu" --ckpt 1122 --pos 0
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunkaug_link_prompt/nq.py --run "chunkaug_25_qa_link_8gpu" --ckpt 1122 --pos 0
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_25_qa_8gpu" --ckpt 1122 --pos 1
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunkaug_link_prompt/nq.py --run "chunkaug_25_qa_link_8gpu" --ckpt 1122 --pos 1

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_25_qa_8gpu" --ckpt 1122 --pos 2
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunkaug_link_prompt/nq.py --run "chunkaug_25_qa_link_8gpu" --ckpt 1122 --pos 2
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_25_qa_8gpu" --ckpt 1122 --pos 3
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunkaug_link_prompt/nq.py --run "chunkaug_25_qa_link_8gpu" --ckpt 1122 --pos 3

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_25_qa_8gpu" --ckpt 1122 --pos 4
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunkaug_link_prompt/nq.py --run "chunkaug_25_qa_link_8gpu" --ckpt 1122 --pos 4
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_25_qa_8gpu" --ckpt 1122 --pos 5
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunkaug_link_prompt/nq.py --run "chunkaug_25_qa_link_8gpu" --ckpt 1122 --pos 5

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_25_qa_8gpu" --ckpt 1122 --pos 6
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunkaug_link_prompt/nq.py --run "chunkaug_25_qa_link_8gpu" --ckpt 1122 --pos 6
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_25_qa_8gpu" --ckpt 1122 --pos 7
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunkaug_link_prompt/nq.py --run "chunkaug_25_qa_link_8gpu" --ckpt 1122 --pos 7

wait

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_25_qa_8gpu" --ckpt 1122 --pos 8
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/chunkaug_link_prompt/nq.py --run "chunkaug_25_qa_link_8gpu" --ckpt 1122 --pos 8
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/chunkaug_prompt/nq.py --run "chunkaug_25_qa_8gpu" --ckpt 1122 --pos 9
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/chunkaug_link_prompt/nq.py --run "chunkaug_25_qa_link_8gpu" --ckpt 1122 --pos 9

wait
