# python preprocess.py
accelerate launch --config_file config/step2.yaml --main_process_port 25671 compress_qa_trainer.py