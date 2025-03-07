import torch
from src.data.attention import make_segment_mask_with_two_rules

def process_text_chunks(
    text_tokens: list,
    compress_tokens: list,
    chunk_size: int = 100,
    chunk_end_token: int = 128253,
    max_length: int = 1024
) -> list:
    """
    Processes `text_tokens` by splitting into chunks of `chunk_size`.
    After each chunk, we append:
      1) chunk_end_token (default=128253),
      2) all `special_tokens`,
      3) the same chunk tokens again.

    We accumulate these processed chunks in a single list until
    adding one more chunk would exceed `max_length`.
    
    Args:
        text_tokens (list): A list of integer token IDs for the text.
        special_tokens (list): A list of integer token IDs to append.
        chunk_size (int): The size of each chunk. Defaults to 100.
        chunk_end_token (int): The special token (e.g. 128253) appended to mark the chunk end.
        max_length (int): The max length of the final output sequence.

    Returns:
        list: The processed token list.
    """
    output_sequence = []
    segment_ids_1 = []
    segment_ids_2 = []
    labels = []
    position_ids = []
    chunk_counter = 0
    # 1. Split text_tokens into slices of size `chunk_size`.
    for i in range(0, len(text_tokens), 2 * chunk_size):

        chunk_counter += 1
        chunk1 = text_tokens[i : i + chunk_size]
        chunk2 = text_tokens[i + chunk_size : i + 2 * chunk_size]
        chunk1_len = len(chunk1)
        chunk2_len = len(chunk2)  # could be < chunk_size for the last chunk

        if chunk2_len < chunk_size:
            break  # no more tokens

        # 2. Build the processed chunk
        #    chunk + [chunk_end_token] + self.compress_tokens + chunk
        processed_chunk = chunk1 + [chunk_end_token] + compress_tokens + chunk2

        # 3. Check if adding this processed chunk would exceed max_length
        if len(output_sequence) + len(processed_chunk) > max_length:
            # If we can't add this chunk without exceeding,
            # we stop processing further.
            break

        segment_ids_1.extend([chunk_counter] * len(processed_chunk))

        segment_ids_2.extend([1] * (chunk1_len + 1) + [2] * len(compress_tokens) + [3] * chunk2_len)

        labels.extend([-100] * (chunk1_len + 1 + len(compress_tokens)) + chunk2)

        position_ids.extend(list(range(-chunk1_len - 1, len(compress_tokens) + chunk2_len)))

        output_sequence.extend(processed_chunk)

    return {
        "input_ids": output_sequence,
        "segment_ids_rule1": segment_ids_1,
        "segment_ids_rule2": segment_ids_2,
        "labels": labels,
        "position_ids": position_ids,
    }


if __name__ == "__main__":
    # Example: We have 10 tokens total.
    # We'll use chunk_size=3 to demonstrate chunk-splitting.
    text_tokens = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    special_tokens = [999, 1000]  # e.g. some special markers

    result = process_text_chunks(
        text_tokens=text_tokens,
        compress_tokens=special_tokens,
        chunk_size=2,
        chunk_end_token=128253,
        max_length=50  # for demonstration
    )



    print("Output length:", len(result["input_ids"]))
    for k, v in result.items():
        print(f"{k} = {v}")

# mask = make_segment_mask_with_two_rules(
#     source_segments_1=torch.tensor([[0, 0, 0, -1]]),
#     target_segments_1=torch.tensor([[0, 0, 0, -1]]),
#     source_segments_2=torch.tensor([[3, 3, 3, -1]]),
#     target_segments_2=torch.tensor([[3, 3, 3, -1]]),
#     dtype=torch.bfloat16,           # For printing, let's keep float
#     add_causal_lm_mask=True     # No causal mask for clarity
# )

# print(mask)

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