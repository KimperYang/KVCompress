import torch
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.attention import make_segment_mask_with_two_rules
def filter_kv(past_key_values, segment_ids_2):
    num_layers = len(past_key_values)
    filtered_past_key_values = ()

    T = past_key_values[0][0].shape[2] 
    mask = torch.ones(T, dtype=bool)

    mask[segment_ids_2[0] == 1] = False

    for layer_id in range(num_layers):
        tem_key = past_key_values[layer_id][0]
        tem_value = past_key_values[layer_id][1]

        filtered_key = tem_key[:, :, mask, :]
        filtered_value = tem_value[:, :, mask, :]

        filtered_past_key_values += ((filtered_key, filtered_value),)

    return filtered_past_key_values

def filter_id(input_ids, segment_ids_2):  

    T = input_ids.shape[1] 
    mask = torch.ones(T, dtype=bool)
    mask[segment_ids_2[0] == 1] = False

    return input_ids[:, mask]

tokenizer = AutoTokenizer.from_pretrained("training_res/compress_pretrain/checkpoint-20000")
model = AutoModelForCausalLM.from_pretrained(
    "training_res/compress_pretrain/checkpoint-20000",
    torch_dtype=torch.bfloat16,
    attn_implementation='sdpa',
    # use_flash_attention_2=True,
)
model.to('cuda')
test_data = datasets.load_from_disk("/data/jingbo_yang/KVMemory/dataset_cache/processed/fineweb/text")["test"]

for i in range(1):
    example = test_data[i]
    compress_tokens = list(range(128011, 128061))
    text_tokens = tokenizer(example['text'], add_special_tokens= False).input_ids[:100]
    text_tokens.extend([128253])
    text_tokens.extend(compress_tokens)

    segment_ids_1 = torch.tensor([[1] * len(text_tokens)])
    segment_ids_2 = torch.tensor([[1] * 101 + [2] * len(compress_tokens)])

    input_ids = torch.tensor([text_tokens], device=model.device)
    position_ids = torch.tensor([list(range(-101, len(compress_tokens)))], device=model.device)
    mask = make_segment_mask_with_two_rules(
        source_segments_1=segment_ids_1,
        target_segments_1=segment_ids_1,
        source_segments_2=segment_ids_2,
        target_segments_2=segment_ids_2,
        dtype=torch.bfloat16,
        add_causal_lm_mask=True
    ).unsqueeze(1).to(model.device)

    with torch.no_grad():
        prefill_output = model(input_ids = input_ids, attention_mask = mask, position_ids = position_ids)
        prefill_kv = prefill_output.past_key_values
    filtered_id = filter_id(input_ids, segment_ids_2=segment_ids_2)
    filtered_kv = filter_kv(prefill_kv, segment_ids_2=segment_ids_2)
    filtered_position = filter_id(position_ids, segment_ids_2=segment_ids_2)
    print(filtered_id.shape, filtered_kv[0][0].shape)
    print(filtered_position)

    generate_ids = torch.cat([filtered_id, torch.tensor([[text_tokens[0]]], device=model.device)], dim=1)
    with torch.no_grad():
        generated = model.generate(input_ids=generate_ids, 
                                   cache_position=filtered_position, 
                                   past_key_values=filtered_kv,
                                   max_new_tokens=100,
                                    do_sample=False,
                                    use_cache=True)
        print(text_tokens[:100])
        print(generated[0][50:])