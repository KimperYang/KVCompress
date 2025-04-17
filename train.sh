# python preprocess.py
accelerate launch --config_file config/step2.yaml --main_process_port 25671 compress_qa_trainer.py
accelerate launch --config_file config/step2.yaml --main_process_port 25671 compress_qa_trainer.py --dataset "qa_compress"
accelerate launch --config_file config/step2.yaml --main_process_port 25671 compress_qa_trainer.py --dataset "qa_compress_link"