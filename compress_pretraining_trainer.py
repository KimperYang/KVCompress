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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from functools import partial

from src.data.input_preprocessor import custom_collate_compress, compress_attention_preprocessor
from src.training.custom_trainer import CustomTrainerCompressAttn


def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: compress_attention_preprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text_singlechunk", "text_multichunk", "text_multichunk2"]:
        data_path = f"dataset_cache/processed/fineweb/{data_component_name}"
        if data_component_name == "text_multichunk":
            preprocessor_fn = preprocessor.process_pretraining_multichunk_completion_compress
            data_path = "dataset_cache/processed/fineweb/text_min2048"
        elif data_component_name == "text_singlechunk":
            preprocessor_fn = preprocessor.process_pretraining_singlechunk_completion_compress
            data_path = "dataset_cache/processed/fineweb/text_min500"
        elif data_component_name == "text_multichunk2":
            preprocessor_fn = preprocessor.process_pretraining_multichunk2_batch
            data_path = "dataset_cache/processed/fineweb/text_min500"
        else:
            raise NotImplementedError()
        remove_columns = [
            "text", "id", "dump", "url", "date",
            "file_path", "language", "language_score", "token_count",
        ]
        num_shards = 512
    else:
        raise NotImplementedError()
    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)

    # streaming_train_dataset = data_component["train"].to_iterable_dataset(num_shards=num_shards)
    streaming_train_dataset = data_component["train"].select(range(0, 3200000))
    training_data = streaming_train_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=96,
        batched=True,
        load_from_cache_file=True
    )

    eval_dataset = data_component["test"]
    eval_data = eval_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=96,
        batched=True,
        load_from_cache_file=True
    )

    return training_data, eval_data


def main():
    batch_size_per_device = 4

    # compress_tokens = list(range(128011, 128061))
    compress_tokens = list(range(128011, 128031))

    global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    global_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
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
        do_shuffle=True
    )

    # train_dataset, eval_dataset = load_from_disk_then_process("text_multichunk2", preprocessor)
    data_component = datasets.load_from_disk("dataset_cache/processed/fineweb/mapped_text_multichunk_20_chunkcomp")
    train_dataset, eval_dataset = data_component["train"], data_component["test"]

    os.environ["WANDB_PROJECT"]="kvcompress"
    os.environ["WANDB_WATCH"]="false"

    training_args = TrainingArguments(
        output_dir="training_res/compress_chunk20_pretrain_multichunk20k",
        report_to="wandb",
        run_name=f"compress_chunk_{len(compress_tokens)}_pretrain_multichunk20k",
        per_device_train_batch_size= batch_size_per_device,
        # num_train_epochs=2,
        max_steps=20000,
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
        evaluation_strategy="steps",  # Add this line
        eval_steps=5000,
        gradient_checkpointing=True,
        save_total_limit=1,
        # overwrite_output_dir = False
        remove_unused_columns=False,
        # split_batches=True,
        dispatch_batches=False,
        eval_on_start=False,
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

    # trainer.train()
    trainer.train(resume_from_checkpoint = True)

if __name__ == "__main__":
    main()
