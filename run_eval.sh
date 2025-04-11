conda activate kvm
# cd /data/jingbo_yang/KVCompress

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratio_kvlink/tqa_eval.py --run "ratio_50_compress_qa_kvlink_multichunk20k_2e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/ratio_kvlink/hqa2_eval.py --run "ratio_50_compress_qa_kvlink_multichunk20k_2e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratio_kvlink/nq2_eval.py --run "ratio_50_compress_qa_kvlink_multichunk20k_2e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratio_kvlink/wiki_eval.py --run "ratio_50_compress_qa_kvlink_multichunk20k_2e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/chunk_nopadding_kvlink/tqa_eval.py --run "compress_chunk_qa_kvlink_nopadding_multichunk20k_1e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/chunk_nopadding_kvlink/nq2_eval.py --run "compress_chunk_qa_kvlink_nopadding_multichunk20k_1e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/chunk_nopadding_kvlink/hqa2_eval.py --run "compress_chunk_qa_kvlink_nopadding_multichunk20k_1e-5" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/chunk_nopadding_kvlink/wiki_eval.py --run "compress_chunk_qa_kvlink_nopadding_multichunk20k_1e-5" --ckpt 1122 &
wait