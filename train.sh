accelerate launch --config_file config/step2.yaml --main_process_port 25671 ratio_compress_pretraining_trainer.py
accelerate launch --config_file config/step2.yaml --main_process_port 25671 ratio_compress_qa_trainer.py
