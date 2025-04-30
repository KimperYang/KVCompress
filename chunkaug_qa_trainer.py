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
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from functools import partial

from src.data.input_preprocessor import custom_collate_chunkaug, chunkaug_attention_preprocessor
from src.training.custom_trainer import ChunkAugTrainer


def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: chunkaug_attention_preprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text_multichunk"]:
        data_path = f"dataset_cache/processed/fineweb/{data_component_name}"
        if data_component_name == "text_multichunk":
            preprocessor_fn = preprocessor.process_pretraining_multichunk_completion_compress
            data_path = "dataset_cache/processed/fineweb/text_min2048"
        else:
            raise NotImplementedError()
        remove_columns = [
            "text", "id", "dump", "url", "date",
            "file_path", "language", "language_score", "token_count",
        ]
        num_shards = 512
    elif data_component_name in ["qa", "qa_link"]:
        if data_component_name == "qa":
            preprocessor_fn = preprocessor.process_qa
            data_path = "dataset_cache/processed/compress_qa"
        elif data_component_name == "qa_link":
            preprocessor_fn = preprocessor.process_qa_link
            data_path = "dataset_cache/processed/compress_qa"
        else:
            raise NotImplementedError()
        remove_columns=['prompt', 'question', 'answers', 'generated', 'inputs', 'documents']
        num_shards = 32
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

    # compress_tokens = list(range(128011, 128061))
    compress_tokens = list(range(128011, 128061))

    global_tokenizer = AutoTokenizer.from_pretrained("training_res/chunkaug_20k/checkpoint-20000")
    global_model = AutoModelForCausalLM.from_pretrained(
        "training_res/chunkaug_20k/checkpoint-20000",
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        # use_flash_attention_2=True,
    )

    preprocessor = chunkaug_attention_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096,
        compress_tokens=compress_tokens,
        chunk_size=100,
        chunk_end_token=128253,
        do_shuffle=True,
        link_token_num = 5,
        max_memory_num = 10
    )

    train_dataset, eval_dataset = load_from_disk_then_process("qa_link", preprocessor)
    # wandb.init()
    os.environ["WANDB_PROJECT"]="kvcompress"
    os.environ["WANDB_WATCH"]="false"
    # run = wandb.init(
    #     project="kvcompress",    # Specify your project
    # )

    training_args = TrainingArguments(
        output_dir="training_res/chunkaug_qa_link_20k",
        report_to="wandb",
        run_name="chunkaug_qa_link_20k",
        per_device_train_batch_size= batch_size_per_device,
        num_train_epochs=2,
        logging_dir="training_res/logs",
        logging_steps=10,
        save_steps=1000,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        learning_rate=5e-6,
        do_eval=True,
        per_device_eval_batch_size = batch_size_per_device,
        evaluation_strategy="epoch",  # Add this line
        gradient_checkpointing=True,
        save_total_limit=1,
        # overwrite_output_dir = False
        remove_unused_columns=False,
        # split_batches=True,
        dispatch_batches=False,
        eval_on_start=False,
        seed = 42
    )

    trainer = ChunkAugTrainer(
        model=global_model,
        tokenizer=global_tokenizer,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = custom_collate_chunkaug
    )

    trainer.train()
    # trainer.train(resume_from_checkpoint = True)

if __name__ == "__main__":
    main()
