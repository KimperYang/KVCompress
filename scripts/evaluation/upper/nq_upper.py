import os
import torch
import string
import argparse
import json
import regex

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.attention import make_segment_mask_with_two_rules

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
parser.add_argument('--pos', type=int, required=True, help='Position value')

args = parser.parse_args()

run_name = args.run
ckpt = args.ckpt
pos = args.pos

if pos in [0, 4, 9]:
    file_path = f'data/nq/nq-open-10_{pos}.jsonl'
else:
    file_path = path_or_buf='data/nq/nq-open-10_0.jsonl'

data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line)
        data.append(record)

tokenizer = AutoTokenizer.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}")
model = AutoModelForCausalLM.from_pretrained(f"training_res/{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)
model.to('cuda')

system = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an AI assistant. "
          "Always ground your answers in the retrieved documents and do not add unsupported details. If the documents lack sufficient information, indicate that.")

total_num = 500
correct_num = 0
res_list = []

for i in range(total_num):
    print("Processing sample:", str(i))

    input_ids = []

    sys_ids = tokenizer(system, add_special_tokens=False).input_ids
    sys_len = len(sys_ids)

    input_ids.extend(sys_ids)

    doc_list = []
    for k in range(0,10):
        title = data[i]["ctxs"][k]["title"]
        text = data[i]["ctxs"][k]["text"]
        doc_list.append({'title': title, 'text':text})

    if pos not in [0,4,9]:
        ground_truth = doc_list.pop(0)
        doc_list.insert(pos, ground_truth)

    for j in range(0,10):
        title = doc_list[j]["title"]
        text = doc_list[j]["text"]
        tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids

        input_ids.extend(tem_id)

    user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + data[i]['question'] + "<|eot_id|>"
    user_id = tokenizer(user, add_special_tokens=False).input_ids
    user_len = len(user_id)
    input_ids.extend(user_id)

    input_ids = torch.tensor([input_ids], device = model.device)

    with torch.no_grad():
        prefill_output = model(input_ids = input_ids)
        prefill_kv = prefill_output.past_key_values

    prefix_ids = tokenizer("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    generate_ids = torch.cat([input_ids, prefix_ids], dim=1)

    with torch.no_grad():
        generated = model.generate(input_ids=generate_ids, 
                                   past_key_values=prefill_kv,
                                   max_new_tokens=200,
                                    do_sample=False,
                                    use_cache=True)
    
        generated_seq = tokenizer.batch_decode(generated, skip_special_tokens=True)

        response = generated_seq[0].split('assistant\n\n')[-1]
        print(response)

        score = best_subspan_em(response, data[i]["answers"])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": data[i]["question"], "response": response, "gold_answer": data[i]["answers"], "Score": score})
        print("Correct progress", correct_num)
    
if not os.path.exists(f"result/{run_name}"):
    os.makedirs(f"result/{run_name}")

accuracy = correct_num / total_num

file_name = f"result/{run_name}/NQ_ckpt{ckpt}_at{pos}_{accuracy}.jsonl"

with open(file_name, 'w', encoding='utf-8') as f:
    for entry in res_list:
        json_line = json.dumps(entry)
        f.write(json_line + '\n')