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
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

from src.data.input_preprocessor import custom_collate_upper, upper_attention_preprocessor


def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: upper_attention_preprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text"]:
        if data_component_name == "text":
            preprocessor_fn = preprocessor.process_pretraining
            data_path = "/data/jingbo_yang/KVCompress/dataset_cache/processed/fineweb/text"
        else:
            raise NotImplementedError()
        remove_columns = [
            "text", "id", "dump", "url", "date",
            "file_path", "language", "language_score", "token_count",
        ]
        num_shards = 512
    elif data_component_name in ["qa", "qa2"]:
        if data_component_name == "qa":
            preprocessor_fn = preprocessor.process_qa
            data_path = "/data/jingbo_yang/KVMemory/dataset_cache/processed/block_qa/qa"
        elif data_component_name == "qa2":
            preprocessor_fn = preprocessor.process_qa
            data_path = "/data/jingbo_yang/KVMemory/dataset_cache/processed/block_qa/qa_mem"
        else:
            raise NotImplementedError()
        remove_columns=['prompt', 'question', 'answers', 'generated', 'inputs', 'documents']
        num_shards = 32
    else:
        raise NotImplementedError()
    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)

    streaming_train_dataset = data_component["train"].to_iterable_dataset(num_shards=num_shards)
    training_data = streaming_train_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        # num_proc=16,
        batched=False,
    )

    eval_dataset = data_component["test"]
    eval_data = eval_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=16,
        batched=False,
        load_from_cache_file=False
    )

    return training_data, eval_data


def main():
    batch_size_per_device = 4

    global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    global_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        # use_flash_attention_2=True,
    )

    preprocessor = upper_attention_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096,
        do_shuffle=True
    )

    qa2_train, qa2_eval = load_from_disk_then_process("qa2", preprocessor)
    qa_train, qa_eval = load_from_disk_then_process("qa", preprocessor)

    train_dataset = datasets.interleave_datasets(
        [qa_train, qa2_train],
        probabilities=[0.50, 0.50],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    eval_dataset = datasets.DatasetDict({
        "qa_eval": qa_eval,
        "qa2_eval": qa2_eval
    })



    os.environ["WANDB_PROJECT"]="kvcompress"
    os.environ["WANDB_WATCH"]="false"

    training_args = TrainingArguments(
        output_dir=f"training_res/upper",
        report_to="wandb",
        run_name=f"upper_bsz{batch_size_per_device}_5e-6",
        per_device_train_batch_size= batch_size_per_device,
        # num_train_epochs=3,
        max_steps=2500,
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
        evaluation_strategy="steps",
        eval_steps=1000,
        gradient_checkpointing=True,
        save_total_limit=1,
        # overwrite_output_dir = False
        remove_unused_columns=False,
        # split_batches=True,
        dispatch_batches=False,
        eval_on_start=True,
        seed = 42
    )

    trainer = Trainer(
        model=global_model,
        tokenizer=global_tokenizer,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = custom_collate_upper
    )

    trainer.train()

if __name__ == "__main__":
    main()
