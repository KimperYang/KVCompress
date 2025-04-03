"""
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/h100_config.yaml \
    --main_process_port 25678 block_attn_trainer.py

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/single_gpu.yaml \
    --main_process_port 25678 block_attn_trainer.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file configs/fsdp.yaml \
    --main_process_port 25678 block_attn_trainer.py
"""
from typing import Tuple

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.data.input_preprocessor import kvlink_preprocessor

def load_from_disk_then_process(
    data_component_name: str,
    preprocessor: kvlink_preprocessor,
) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
    """
    load the downloaded data from disk and then pair it with the preprocessor
    """
    if data_component_name in ["text_singlechunk", "text_multichunk"]:
        if data_component_name == "text_multichunk":
            preprocessor_fn = preprocessor.process_pretraining_kvlink
            data_path = "dataset_cache/processed/fineweb/text_min2048"
        else:
            raise NotImplementedError()
        remove_columns = [
            "text", "id", "dump", "url", "date",
            "file_path", "language", "language_score", "token_count",
        ]
        num_shards = 512
    elif data_component_name in ["qa_link"]:
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

    streaming_train_dataset = data_component["train"]
    # streaming_train_dataset = data_component["train"]
    training_data = streaming_train_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=96,
        batched=False,
    )

    eval_dataset = data_component["test"]
    eval_data = eval_dataset.map(
        preprocessor_fn,
        remove_columns=remove_columns,
        num_proc=96,
        batched=False,
        load_from_cache_file=False
    )

    return training_data, eval_data

# def load_from_disk_then_process(
#     data_component_name: str,
#     preprocessor,
# ) -> Tuple[datasets.IterableDataset, datasets.Dataset]:
#     """
#     load the downloaded data from disk and then pair it with the preprocessor
#     """
#     if data_component_name in ["text_singlechunk", "text_multichunk", "text_multichunk2", "text_multichunk_kvlink"]:
#         data_path = f"dataset_cache/processed/fineweb/{data_component_name}"
#         if data_component_name == "text_multichunk":
#             preprocessor_fn = preprocessor.process_pretraining_multichunk_completion_compress
#             data_path = "dataset_cache/processed/fineweb/text_min2048"
#         elif data_component_name == "text_singlechunk":
#             preprocessor_fn = preprocessor.process_pretraining_singlechunk_completion_compress
#             data_path = "dataset_cache/processed/fineweb/text_min500"
#         elif data_component_name == "text_multichunk2":
#             preprocessor_fn = preprocessor.process_pretraining_multichunk2_batch
#             data_path = "dataset_cache/processed/fineweb/text_min500"
#         elif data_component_name == "text_multichunk_kvlink":
#             preprocessor_fn = preprocessor.process_pretraining_multichunk_kvlink_completion_compress
#             data_path = "dataset_cache/processed/fineweb/text_min2048"
#         else:
#             raise NotImplementedError()
#         remove_columns = [
#             "text", "id", "dump", "url", "date",
#             "file_path", "language", "language_score", "token_count",
#         ]
#         num_shards = 512
#     else:
#         raise NotImplementedError()
#     data_component: datasets.DatasetDict = datasets.load_from_disk(data_path)

#     # streaming_train_dataset = data_component["train"].to_iterable_dataset(num_shards=num_shards)
#     # streaming_train_dataset = data_component["train"].select(range(0, 3200000))
#     streaming_train_dataset = data_component["train"]
#     training_data = streaming_train_dataset.map(
#         preprocessor_fn,
#         remove_columns=remove_columns,
#         num_proc=96,
#         batched=False
#     )

#     eval_dataset = data_component["test"]
#     eval_data = eval_dataset.map(
#         preprocessor_fn,
#         remove_columns=remove_columns,
#         num_proc=96,
#         batched=False
#     )

#     return training_data, eval_data


def main():

    global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # compress_tokens = list(range(128011, 128061))
    # preprocessor = compress_attention_preprocessor(
    #     tokenizer=global_tokenizer,
    #     max_len=4096,
    #     compress_tokens=compress_tokens,
    #     chunk_size=100,
    #     chunk_end_token=128253,
    #     do_shuffle=True
    # )

    # compress_tokens = list(range(128011, 128211))
    # ratio = 0.5
    # preprocessor = compress_ratio_preprocessor(
    #     tokenizer=global_tokenizer,
    #     max_len=4096,
    #     compress_tokens=compress_tokens,
    #     compress_ratio=ratio,
    #     chunk_end_token=128253,
    #     do_shuffle=True,
    #     max_chunk_num=20,
    # )

    preprocessor = kvlink_preprocessor(
        tokenizer=global_tokenizer,
        max_len=4096,
        do_shuffle=True,
        link_token_num = 1,
        max_chunk_num = 20,
    )

    train_set, test_set = load_from_disk_then_process("text_multichunk_kvlink", preprocessor)
    dataset = datasets.DatasetDict({'train': train_set, 'test': test_set})
    shards = {'train': 128, 'test': 4}
    dataset.save_to_disk("dataset_cache/processed/fineweb/mapped_text_kvlink", num_shards=shards, num_proc=128)


if __name__ == "__main__":
    main()
