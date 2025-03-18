import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from src.data.attention import make_segment_mask_with_two_rules
from src.data.input_preprocessor import compress_ratio_preprocessor

# data_component = datasets.load_from_disk("dataset_cache/processed/fineweb/text_min2048")

# global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# ratio=0.5
# compress_tokens = list(range(128011, 128211))
# preprocessor = compress_ratio_preprocessor(
#     tokenizer=global_tokenizer,
#     max_len=4096,
#     compress_tokens=compress_tokens,
#     compress_ratio=ratio,
#     chunk_end_token=128253,
#     do_shuffle=True
# )

# eval_dataset = data_component["test"]
# eval_data = eval_dataset.map(
#     preprocessor.process_pretraining_multichunk_completion_compress,
#     # num_proc=16,
#     batched=False,
#     load_from_cache_file=False
# )


# def process_text_chunks(
#     text_tokens: list,
#     compress_tokens: list,
#     chunk_size: int = 100,
#     chunk_end_token: int = 128253,
#     max_length: int = 1024
# ) -> list:
#     """
#     Processes `text_tokens` by splitting into chunks of `chunk_size`.
#     After each chunk, we append:
#       1) chunk_end_token (default=128253),
#       2) all `special_tokens`,
#       3) the same chunk tokens again.

#     We accumulate these processed chunks in a single list until
#     adding one more chunk would exceed `max_length`.
    
#     Args:
#         text_tokens (list): A list of integer token IDs for the text.
#         special_tokens (list): A list of integer token IDs to append.
#         chunk_size (int): The size of each chunk. Defaults to 100.
#         chunk_end_token (int): The special token (e.g. 128253) appended to mark the chunk end.
#         max_length (int): The max length of the final output sequence.

#     Returns:
#         list: The processed token list.
#     """
#     output_sequence = []
#     segment_ids_1 = []
#     segment_ids_2 = []
#     labels = []
#     position_ids = []
#     chunk_counter = 0
#     # 1. Split text_tokens into slices of size `chunk_size`.
#     for i in range(0, len(text_tokens), 2 * chunk_size):

#         chunk_counter += 1
#         chunk1 = text_tokens[i : i + chunk_size]
#         chunk2 = text_tokens[i + chunk_size : i + 2 * chunk_size]
#         chunk1_len = len(chunk1)
#         chunk2_len = len(chunk2)  # could be < chunk_size for the last chunk

#         if chunk2_len < chunk_size:
#             break  # no more tokens

#         # 2. Build the processed chunk
#         #    chunk + [chunk_end_token] + self.compress_tokens + chunk
#         processed_chunk = chunk1 + [chunk_end_token] + compress_tokens + chunk2

#         # 3. Check if adding this processed chunk would exceed max_length
#         if len(output_sequence) + len(processed_chunk) > max_length:
#             # If we can't add this chunk without exceeding,
#             # we stop processing further.
#             break

#         segment_ids_1.extend([chunk_counter] * len(processed_chunk))

#         segment_ids_2.extend([1] * (chunk1_len + 1) + [2] * len(compress_tokens) + [3] * chunk2_len)

#         labels.extend([-100] * (chunk1_len + 1 + len(compress_tokens)) + chunk2)

#         position_ids.extend(list(range(-chunk1_len - 1, len(compress_tokens) + chunk2_len)))

#         output_sequence.extend(processed_chunk)

#     return {
#         "input_ids": output_sequence,
#         "segment_ids_rule1": segment_ids_1,
#         "segment_ids_rule2": segment_ids_2,
#         "labels": labels,
#         "position_ids": position_ids,
#     }


# if __name__ == "__main__":
#     # Example: We have 10 tokens total.
#     # We'll use chunk_size=3 to demonstrate chunk-splitting.
#     text_tokens = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
#     special_tokens = [999, 1000]  # e.g. some special markers

#     result = process_text_chunks(
#         text_tokens=text_tokens,
#         compress_tokens=special_tokens,
#         chunk_size=2,
#         chunk_end_token=128253,
#         max_length=50  # for demonstration
#     )



#     print("Output length:", len(result["input_ids"]))
#     for k, v in result.items():
#         print(f"{k} = {v}")

seg_id_1 = torch.tensor([[1,1,2,2,0,4,4]])
seg_id_2 = torch.tensor([[3,3,3,3,3,3,3]])
mask = make_segment_mask_with_two_rules(
    source_segments_1=seg_id_1,
    target_segments_1=seg_id_1,
    source_segments_2=seg_id_2,
    target_segments_2=seg_id_2,
    dtype=torch.bfloat16,           # For printing, let's keep float
    add_causal_lm_mask=True     # No causal mask for clarity
)

print(mask)

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from src.data.input_preprocessor import custom_collate_compress, compress_attention_preprocessor
# import datasets

# data_component = datasets.load_from_disk("/data/jingbo_yang/KVMemory/dataset_cache/processed/block_qa/qa_mem")['test']

# compress_tokens = list(range(128011, 128016))

# global_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# global_model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-3.2-1B",
#     torch_dtype=torch.bfloat16,
#     attn_implementation='sdpa',
#     # use_flash_attention_2=True,
# )

# preprocessor = compress_attention_preprocessor(
#     tokenizer=global_tokenizer,
#     max_len=4096,
#     compress_tokens=compress_tokens,
#     chunk_size=100,
#     chunk_end_token=128253,
#     do_shuffle=True
# )

# training_data = data_component.map(
#     preprocessor.process_qa_chunk_compress,
#     # num_proc=16,
#     batched=False,
# )
import random

seq_len = 4000
max_chunk_num = 20
max_prefix_length = random.randint(0.2 * seq_len, 0.8 * seq_len)
chunk_num = random.randint(1, max_chunk_num)
total_chunk_len = 0
for _ in range(chunk_num):
    chunk_len = random.randint(20, 400)
    total_chunk_len += chunk_len
    if total_chunk_len > max_prefix_length:
        break
