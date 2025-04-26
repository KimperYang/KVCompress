"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/h100_config.yaml \
    --main_process_port 25678 block_attn_trainer.py

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/single_gpu.yaml \
    --main_process_port 25678 block_attn_trainer.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/fsdp.yaml \
    --main_process_port 25678 block_attn_trainer.py
"""
import os
from typing import Tuple

import datasets
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from functools import partial

from src.data.input_preprocessor import custom_collate_anchor, AnchorPreprocessor
from src.training.custom_trainer import AnchorTrainer


def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: AnchorPreprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["qa", "qa_link"]:
        data_path = "dataset_cache/processed/compress_qa"
        if data_component_name == "qa":
            preprocessor_fn = preprocessor.process_qa
        elif data_component_name == "qa_link":
            preprocessor_fn = preprocessor.process_qa_link
        else:
            raise NotImplementedError()
        remove_columns=['prompt', 'question', 'answers', 'generated', 'inputs', 'documents']
        num_shards = 512
    else:
        raise NotImplementedError()
    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)

    # streaming_train_dataset = data_component["train"].to_iterable_dataset(num_shards=num_shards)
    streaming_train_dataset = data_component["train"]
    training_data = streaming_train_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=96,
        batched=False,
        load_from_cache_file=True
    )

    eval_dataset = data_component["test"]
    eval_data = eval_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=96,
        batched=False,
        load_from_cache_file=True
    )

    return training_data, eval_data


def main():
    batch_size_per_device = 4
    parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
    parser.add_argument('--dataset', type=str, required=True, help='Path under training_res')

    args = parser.parse_args()

    dataset = args.dataset

    global_tokenizer = AutoTokenizer.from_pretrained("training_res/anchor_qa_10k/checkpoint-1122")
    global_model = AutoModelForCausalLM.from_pretrained(
        "training_res/anchor_qa_10k/checkpoint-1122",
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        # use_flash_attention_2=True,
    )

    preprocessor = AnchorPreprocessor(
        tokenizer=global_tokenizer,
        max_len=4096,
        anchor_id=128011
    )

    train_dataset, eval_dataset = load_from_disk_then_process(dataset, preprocessor)
    # data_component = datasets.load_from_disk("dataset_cache/processed/fineweb/anchor")
    # train_dataset, eval_dataset = data_component["train"], data_component["test"]

    os.environ["WANDB_PROJECT"]="kvcompress"
    os.environ["WANDB_WATCH"]="false"

    training_args = TrainingArguments(
        output_dir=f"training_res/anchor_continue_{dataset}_10k",
        report_to="wandb",
        run_name=f"anchor_continue_{dataset}_10k",
        per_device_train_batch_size= batch_size_per_device,
        num_train_epochs=1,
        # max_steps=4000,
        logging_dir="training_res/logs",
        logging_steps=10,
        # save_steps=1000,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        learning_rate=5e-6,
        do_eval=True,
        per_device_eval_batch_size = batch_size_per_device,
        evaluation_strategy="epoch",  # Add this line
        # eval_steps=5000,
        gradient_checkpointing=True,
        save_total_limit=1,
        # overwrite_output_dir = False
        remove_unused_columns=False,
        # split_batches=True,
        dispatch_batches=False,
        eval_on_start=False,
        seed = 42
    )

    trainer = AnchorTrainer(
        model=global_model,
        tokenizer=global_tokenizer,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = custom_collate_anchor
    )

    trainer.train()
    # trainer.train(resume_from_checkpoint = True)

if __name__ == "__main__":
    main()
