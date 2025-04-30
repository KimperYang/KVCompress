import os
import torch
import string
import argparse
import json
import regex
import datasets

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.attention import make_chunk_aug_mask

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

def normalize_answer(s) -> str:
    """Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument('--run', type=str, required=True, help='Path under training_res')
parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')

args = parser.parse_args()

run_name = args.run
ckpt = args.ckpt

file_path = "data/wiki/dev.json"
with open(file_path, 'r') as file:
    data = json.load(file)

tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")
model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)
model.to('cuda')

system = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an AI assistant. "
          "Always ground your answers in the retrieved documents and do not add unsupported details. If the documents lack sufficient information, indicate that.")

total_num = len(data)
correct_num = 0
res_list = []

pad_token = 128004
chunk_size = 100
chunk_end_token = 128253
global_start_token = 128254
global_end_token = 128255
compress_tokens = list(range(128011, 128061))

link_token_num = 5
link_token_start = compress_tokens[-1] + 1
link_tokens = [
    [
        link_token_start + idx * link_token_num + offset
        for offset in range(link_token_num)
    ]
    for idx in range(10)
]

for i in range(total_num):
    print("Processing sample:", str(i))

    input_ids = []
    segment_ids_1 = []
    segment_ids_2 = []
    chunk_index_ids = []
    position_ids = []

    sys_ids = tokenizer(system, add_special_tokens=False).input_ids + [global_start_token]
    sys_len = len(sys_ids)

    input_ids.extend(sys_ids)
    segment_ids_1.extend([0] * sys_len)
    segment_ids_2.extend([3] * sys_len)
    position_ids.extend(list(range(sys_len)))
    chunk_index_ids.extend([-1] * sys_len)

    current_index = sys_len

    for j in range(0,10):
        title = data[i]['context'][j][0]
        text = " ".join(data[i]['context'][j][1])
        tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids + [chunk_end_token]

        chunk_idx = 1
        for idx in range(0, len(tem_id), chunk_size):
            chunk_id = tem_id[idx : idx + chunk_size]
            chunk_len = len(chunk_id)
            
            segment_ids_1.extend([chunk_idx] * (chunk_len + 1 + len(compress_tokens)))
            segment_ids_2.extend([1] * (chunk_len + 1) + [2] * len(compress_tokens))
            chunk_index_ids.extend([j] * (chunk_len + 1 + len(compress_tokens)))
            position_ids.extend(list(range(current_index - chunk_len - 1, current_index + len(compress_tokens))))
            input_ids.extend(chunk_id + [chunk_end_token] + compress_tokens)

            current_index += len(compress_tokens)
            chunk_idx += 1

        input_ids += link_tokens[j]
        segment_ids_1 += [0] * link_token_num
        segment_ids_2 += [3] * link_token_num
        chunk_index_ids += [-1] * link_token_num
        position_ids += list(range(current_index, current_index + link_token_num))
        current_index += link_token_num

    user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + data[i]['question'] + "<|eot_id|>"
    user_id = [global_end_token] + tokenizer(user, add_special_tokens=False).input_ids
    user_len = len(user_id)
    segment_ids_1.extend([0] * user_len)
    segment_ids_2.extend([3] * user_len)
    chunk_index_ids.extend([-1] * user_len)
    position_ids.extend(list(range(current_index, current_index + user_len)))
    input_ids.extend(user_id)
    current_index += user_len

    input_ids = torch.tensor([input_ids], device = model.device)
    position_ids = torch.tensor([position_ids], device=model.device)
    segment_ids_1 = torch.tensor([segment_ids_1])
    segment_ids_2 = torch.tensor([segment_ids_2])
    chunk_index_ids = torch.tensor([chunk_index_ids])

    mask = make_chunk_aug_mask(
        source_seg1=segment_ids_1,
        target_seg1=segment_ids_1,
        source_seg2=segment_ids_2,
        target_seg2=segment_ids_2,
        source_chunk=chunk_index_ids,
        target_chunk=chunk_index_ids,
        dtype=torch.bfloat16,
        add_causal_lm_mask=True
    ).unsqueeze(1).to(model.device)

    with torch.no_grad():
        prefill_output = model(input_ids = input_ids, attention_mask = mask, position_ids = position_ids)
        prefill_kv = prefill_output.past_key_values
    filtered_ids = filter_id(input_ids, segment_ids_2=segment_ids_2)
    filtered_kv = filter_kv(prefill_kv, segment_ids_2=segment_ids_2)
    filtered_position = filter_id(position_ids, segment_ids_2=segment_ids_2)

    prefix_ids = tokenizer("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    generate_ids = torch.cat([filtered_ids, prefix_ids], dim=1)

    with torch.no_grad():
        generated = model.generate(input_ids=generate_ids, 
                                   cache_position=filtered_position, 
                                   past_key_values=filtered_kv,
                                   max_new_tokens=200,
                                    do_sample=False,
                                    use_cache=True)
    
        generated_seq = tokenizer.batch_decode(generated, skip_special_tokens=True)

        response = generated_seq[0].split('assistant\n\n')[-1]
        print(response)

        score = best_subspan_em(response, [data[i]["answer"]])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": data[i]["question"], "response": response, "gold_answer": data[i]["answer"], "Score": score})
        print("Correct progress", correct_num)
    
if not os.path.exists(f"result/{run_name}"):
    os.makedirs(f"result/{run_name}")

accuracy = correct_num / total_num

file_name = f"result/{run_name}/wiki_full_ckpt{ckpt}_{accuracy}.jsonl"

with open(file_name, 'w', encoding='utf-8') as f:
    for entry in res_list:
        json_line = json.dumps(entry)
        f.write(json_line + '\n')