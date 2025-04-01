accelerate launch --config_file config/step2.yaml --main_process_port 25671 ratio_compress_pretraining_trainer.py
accelerate launch --config_file config/step2.yaml --main_process_port 25671 ratio_compress_qa_trainer.py
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/ratio_kvlink/tqa_eval.py --run "ratio_compress_qa_kvlink_linkchunk20k" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/ratio_kvlink/hqa2_eval.py --run "ratio_compress_qa_kvlink_linkchunk20k" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluation/ratio_kvlink/nq2_eval.py --run "ratio_compress_qa_kvlink_linkchunk20k" --ckpt 1122 &
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/ratio_kvlink/wiki_eval.py --run "ratio_compress_qa_kvlink_linkchunk20k" --ckpt 1122 &