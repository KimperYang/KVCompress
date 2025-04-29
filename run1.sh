#!/bin/bash
startTime=$(date +%s) #mark the start of job 

output_dir=$1
run_name=$2

conda activate kvm

accelerate launch --config_file config/step2.yaml --main_process_port 25671 chunkaug_ptr_trainer.py