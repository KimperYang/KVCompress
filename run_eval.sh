conda activate kvm
# cd /data/jingbo_yang/KVCompress

CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratio/tqa_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/ratio/hqa2_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratio/nq2_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratio/wiki_eval.py --run "ratio_compress_qa_multichunk20k" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=4 python scripts/evaluation/chunk_nopadding_kvlink/tqa_eval.py --run "compress_chunk20_qa_kvlink_nopadding_multichunk20k_epoch2" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=5 python scripts/evaluation/chunk_nopadding_kvlink/nq2_eval.py --run "compress_chunk20_qa_kvlink_nopadding_multichunk20k_epoch2" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=6 python scripts/evaluation/chunk_nopadding_kvlink/hqa2_eval.py --run "compress_chunk20_qa_kvlink_nopadding_multichunk20k_epoch2" --ckpt 1122 &
# CUDA_VISIBLE_DEVICES=7 python scripts/evaluation/chunk_nopadding_kvlink/wiki_eval.py --run "compress_chunk20_qa_kvlink_nopadding_multichunk20k_epoch2" --ckpt 1122 &
wait