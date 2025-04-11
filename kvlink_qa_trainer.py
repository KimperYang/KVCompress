"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file /data/jingbo_yang/.cache/huggingface/accelerate/step2.yaml \
--main_process_port 25671 ratio_compress_pretraining_trainer.py
"""
import os
from typing import Tuple

import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from functools import partial

from src.data.input_preprocessor import custom_collate_compress, kvlink_preprocessor
from src.training.custom_trainer import CustomTrainerCompressAttn


def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: kvlink_preprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    # if data_component_name in ["text_singlechunk", "text_multichunk"]:
    #     data_path = f"dataset_cache/processed/fineweb/{data_component_name}"
    #     if data_component_name == "text_singlechunk":
    #         preprocessor_fn = preprocessor.process_pretraining_singlechunk_completion_compress
    #         data_path = "dataset_cache/processed/fineweb/text"
    #     elif data_component_name == "text_multichunk":
    #         preprocessor_fn = preprocessor.process_pretraining_multichunk_completion_compress
    #         data_path = "dataset_cache/processed/fineweb/text_min2048"
    #     else:
    #         raise NotImplementedError()
    #     remove_columns = [
    #         "text", "id", "dump", "url", "date",
    #         "file_path", "language", "language_score", "token_count",
    #     ]
    #     num_shards = 512
    if data_component_name in ["qa_link"]:
        data_path = "dataset_cache/processed/compress_qa"
        if data_component_name == "qa_link":
            preprocessor_fn = preprocessor.process_qa_kvlink
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
        num_proc=16,
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

    preprocessor = kvlink_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096,
        do_shuffle=True,
        link_token_num = 1,
        max_chunk_num = 10,
    )

    train_dataset, eval_dataset = load_from_disk_then_process("qa_link", preprocessor)

    os.environ["WANDB_PROJECT"]="kvcompress"
    os.environ["WANDB_WATCH"]="false"

    training_args = TrainingArguments(
        output_dir="training_res/kvlink_qa2",
        report_to="wandb",
        run_name="kvlink_qa2",
        per_device_train_batch_size= batch_size_per_device,
        num_train_epochs=2,
        logging_dir="training_res/logs",
        logging_steps=10,
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
        remove_unused_columns=False,
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
