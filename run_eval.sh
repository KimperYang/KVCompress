conda activate kvm
# cd /data/jingbo_yang/KVCompress

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratio_kvlink5/tqa_eval.py --run "ratio_50_compress_qa_kvlink5_multichunk20k_1e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/ratio_kvlink5/hqa2_eval.py --run "ratio_50_compress_qa_kvlink5_multichunk20k_1e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratio_kvlink5/nq2_eval.py --run "ratio_50_compress_qa_kvlink5_multichunk20k_1e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratio_kvlink5/wiki_eval.py --run "ratio_50_compress_qa_kvlink5_multichunk20k_1e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/ratio_kvlink5/tqa_eval.py --run "ratio_50_compress_qa_kvlink5_multichunk20k_2e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/ratio_kvlink5/nq2_eval.py --run "ratio_50_compress_qa_kvlink5_multichunk20k_2e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/ratio_kvlink5/hqa2_eval.py --run "ratio_50_compress_qa_kvlink5_multichunk20k_2e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/ratio_kvlink5/wiki_eval.py --run "ratio_50_compress_qa_kvlink5_multichunk20k_2e-5" --ckpt 1122 &
wait