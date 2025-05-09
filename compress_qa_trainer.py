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

from src.data.input_preprocessor import custom_collate_compress, compress_attention_preprocessor
from src.training.custom_trainer import CustomTrainerCompressAttn


def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: compress_attention_preprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text", "text_compress"]:
        if data_component_name == "text":
            preprocessor_fn = preprocessor.process_pretraining
            data_path = "dataset_cache/processed/fineweb/text"
        elif data_component_name == "text_compress":
            preprocessor_fn = preprocessor.process_pretraining_instruct_compress
            data_path = "dataset_cache/processed/fineweb/text_compress"
        else:
            raise NotImplementedError()
        remove_columns = [
            "text", "id", "dump", "url", "date",
            "file_path", "language", "language_score", "token_count",
        ]
        num_shards = 512
    elif data_component_name in ["qa", "qa_compress", "qa_link", "qa_link_fix"]:
        if data_component_name == "qa":
            preprocessor_fn = preprocessor.process_qa_chunk_nopadding_compress
            data_path = "dataset_cache/processed/compress_qa"
        elif data_component_name == "qa_compress":
            preprocessor_fn = preprocessor.process_qa_chunk_compress
            data_path = "dataset_cache/processed/compress_qa"
        elif data_component_name == "qa_link":
            preprocessor_fn = preprocessor.process_qa_chunk_nopadding_kvlink
            data_path = "dataset_cache/processed/compress_qa"
        elif data_component_name == "qa_link_fix":
            preprocessor_fn = preprocessor.process_qa_chunk_nopadding_kvlink_fix
            data_path = "dataset_cache/processed/compress_qa"
        else:
            raise NotImplementedError()
        remove_columns=['prompt', 'question', 'answers', 'generated', 'inputs', 'documents']
        num_shards = 32
    else:
        raise NotImplementedError()
    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)

    streaming_train_dataset = data_component["train"]
    # streaming_train_dataset = data_component["train"]
    training_data = streaming_train_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=16,
        batched=False,
    )

    eval_dataset = data_component["test"]
    eval_data = eval_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=16,
        batched=False,
        # load_from_cache_file=False
    )

    return training_data, eval_data


def main():
    batch_size_per_device = 4
    compress_tokens = list(range(128011, 128061))

    parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
    parser.add_argument('--dataset', type=str, required=True, help='Path under training_res')

    args = parser.parse_args()

    dataset = args.dataset


    global_tokenizer = AutoTokenizer.from_pretrained("training_res/compress_chunk_qa_nopadding_multichunk20k_epoch2/checkpoint-1122")
    global_model = AutoModelForCausalLM.from_pretrained(
        "training_res/compress_chunk_qa_nopadding_multichunk20k_epoch2/checkpoint-1122",
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        # use_flash_attention_2=True,
    )

    preprocessor = compress_attention_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096,
        compress_tokens=compress_tokens,
        chunk_size=100,
        chunk_end_token=128253,
        do_shuffle=True,
        # link_token_num=5,
        # max_memory_num=1
    )

    train_dataset, eval_dataset = load_from_disk_then_process(dataset, preprocessor)

    os.environ["WANDB_PROJECT"]="kvcompress"
    os.environ["WANDB_WATCH"]="false"

    training_args = TrainingArguments(
        output_dir=f"training_res/continue_{dataset}",
        report_to="wandb",
        run_name=f"compress_chunk_{len(compress_tokens)}_{dataset}",
        per_device_train_batch_size= batch_size_per_device,
        num_train_epochs=1,
        # max_steps=2500,
        logging_dir="training_res/logs",
        logging_steps=10,
        # save_steps=5000,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        learning_rate=5e-6,
        do_eval=True,
        per_device_eval_batch_size = batch_size_per_device,
        evaluation_strategy="epoch",
        # eval_steps=1000,
        gradient_checkpointing=True,
        save_total_limit=1,
        # overwrite_output_dir = False
        remove_unused_columns=False,
        # split_batches=True,
        dispatch_batches=False,
        eval_on_start=True,
        seed = 42
    )

    trainer = CustomTrainerCompressAttn(
        model=global_model,
        tokenizer=global_tokenizer,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = custom_collate_compress
    )

    trainer.train()

if __name__ == "__main__":
    main()
