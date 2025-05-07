"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file /data/jingbo_yang/.cache/huggingface/accelerate/step2.yaml \
--main_process_port 25671 ratio_compress_pretraining_trainer.py
"""
import os
from typing import Tuple

import datasets
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from functools import partial

from src.data.input_preprocessor import custom_collate_compress, sum_compress_preprocessor
from src.training.custom_trainer import CustomTrainerCompressAttn

def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: sum_compress_preprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text", "text_mem", "text_inst"]:
        data_path = f"../tmp_memory/dataset_cache/processed/fineweb/{data_component_name}"
        if data_component_name == "text":
            preprocessor_fn = preprocessor.process_text
        else:
            raise NotImplementedError()
        remove_columns = [
            "text", "id", "dump", "url", "date",
            "file_path", "language", "language_score", "token_count",
        ]
        num_shards = 512
        if data_component_name in ["text_mem", "text_inst"]:
            remove_columns.append("num_tokens")
    elif data_component_name in ["sft", "sft_mem"]:
        data_path = f"../tmp_memory/dataset_cache/processed/daringanteater/{data_component_name}"
        if data_component_name == "sft":
            preprocessor_fn = preprocessor.process_sft
        elif data_component_name == "sft_mem":
            preprocessor_fn = preprocessor.process_sftmem
        else:
            raise NotImplementedError()
        remove_columns=["system", "mask", "dataset", "conversations"]
        num_shards = 32
    elif data_component_name in ["tulu"]:
        data_path = "../tmp_memory/dataset_cache/processed/tulu/sft"
        if data_component_name == "tulu":
            preprocessor_fn = preprocessor.process_tulu
        else:
            raise NotImplementedError()
        remove_columns=["id", "messages", "source"]
        num_shards = 32
    elif data_component_name in ["qa", "qa_mem"]:
        data_path = f"../tmp_memory/dataset_cache/processed/block_qa/{data_component_name}"
        if data_component_name == "qa":
            preprocessor_fn = preprocessor.process_qa
        elif data_component_name == "qa_mem":
            preprocessor_fn = preprocessor.process_qamem
        else:
            raise NotImplementedError()
        remove_columns=['prompt', 'question', 'answers', 'generated', 'inputs', 'documents']
        num_shards = 32
    elif data_component_name in ["xsum"]:
        data_path = f"../tmp_memory/dataset_cache/processed/xsum/{data_component_name}"
        preprocessor_fn = preprocessor.process_xsum
        remove_columns=['document', 'summary', 'id']
        num_shards = 32
    else:
        raise NotImplementedError()
    data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)
    # print(data_component.cleanup_cache_files())

    streaming_train_dataset = data_component["train"].to_iterable_dataset(num_shards=num_shards)
    # streaming_train_dataset = data_component["train"]
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
        num_proc=96,
        batched=False,
        # load_from_cache_file=False
    )

    return training_data, eval_data


def main():

    parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
    parser.add_argument('--reencode', type=int, required=True, help='Path under training_res')

    args = parser.parse_args()

    reencode = args.reencode

    batch_size_per_device = 2
    compress_tokens = [128011] * 400
    ratio = 0.25

    global_tokenizer = AutoTokenizer.from_pretrained("training_res/ratioaug_25_singlechunk20k/checkpoint-20000")
    global_model = AutoModelForCausalLM.from_pretrained(
        "training_res/ratioaug_25_singlechunk20k/checkpoint-20000",
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        # use_flash_attention_2=True,
    )

    preprocessor = sum_compress_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096,
        special_token_start=compress_tokens[-1] + 1,
        mem_start=128254,
        mem_end=128255,
        reencode_num=reencode,
        do_shuffle=True,
        compression_tokens=compress_tokens,
        chunk_end_token=128253,
        ratio=ratio,
    )


    ptr_train, ptr_eval = load_from_disk_then_process("text", preprocessor)
    sft_train, sft_eval = load_from_disk_then_process("tulu", preprocessor)
    sft_mem_train, sft_mem_eval = load_from_disk_then_process("sft_mem", preprocessor)
    qa_train, qa_eval = load_from_disk_then_process("qa", preprocessor)
    qa_mem_train, qa_mem_eval = load_from_disk_then_process("qa_mem", preprocessor)
    xsum_train, xsum_eval = load_from_disk_then_process("xsum", preprocessor)

    train_dataset = datasets.interleave_datasets(
        [sft_mem_train, sft_train, ptr_train, qa_train, qa_mem_train, xsum_train],
        probabilities=[0.25, 0.30, 0.20, 0.10, 0.10, 0.05],
        seed=42,
        stopping_strategy="all_exhausted",
    )

    eval_dataset = datasets.DatasetDict({
        "text": ptr_eval,
        "sft": sft_eval,
        "sftmem": sft_mem_eval,
        "qa": qa_eval,
        "qamem": qa_mem_eval,
        "xsum": xsum_eval,
    })

    os.environ["WANDB_PROJECT"]="kvcompress"
    os.environ["WANDB_WATCH"]="false"

    training_args = TrainingArguments(
        output_dir=f"training_res/ratioaug_25_mix_reencode_{reencode}",
        report_to="wandb",
        run_name=f"ratioaug_{int(ratio * 100)}_mix_reencode_{reencode}",
        per_device_train_batch_size= batch_size_per_device,
        # num_train_epochs=2,
        max_steps=6000,
        logging_dir="training_res/logs",
        logging_steps=10,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        bf16=True,
        learning_rate=5e-6,
        do_eval=True,
        per_device_eval_batch_size = batch_size_per_device,
        evaluation_strategy="steps",  # Add this line
        eval_steps=1000,
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
