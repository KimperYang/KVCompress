import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd    
import json
import datetime
import string
from typing import List
from tqdm import tqdm
import regex
from src.data.attention import make_anchor_attention
import argparse

parser = argparse.ArgumentParser(description="Run script with specified ckpt and pos.")
parser.add_argument('--run', type=str, required=True, help='Path under training_res')
parser.add_argument('--ckpt', type=int, required=True, help='Checkpoint number')

args = parser.parse_args()

run_name = args.run
ckpt = args.ckpt

file_path = "data/wiki/dev.json"
with open(file_path, 'r') as file:
    data = json.load(file)

global_tokenizer = AutoTokenizer.from_pretrained(f"{run_name}/checkpoint-{ckpt}")

global_model = AutoModelForCausalLM.from_pretrained(f"{run_name}/checkpoint-{ckpt}", torch_dtype=torch.bfloat16)

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

def normalize_answer(s: str) -> str:
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

def main():
    global_model.to('cuda')

    input_ids = []
    segment_ids_1 = []
    segment_ids_2 = []
    chunk_ids = []

    anchor_id=list(range(128011, 128016))
    anchor_num = len(anchor_id)
    mem_start=128254
    mem_end=128255

    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a intelligent AI assistant. Please answer questions based on the user's instruction. Below are some reference documents that may help you in answering the user's question.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    total_num = len(data)
    correct_num = 0
    res_list = []

    for i in tqdm(range(total_num)):
        input_ids = []
        segment_ids_1 = []
        segment_ids_2 = []
        chunk_ids = []

        doc_list = []

        for j in range(0,10):
            title = data[i]['context'][j][0]
            text = " ".join(data[i]['context'][j][1])
            doc_list.append({'title': title, 'text':text})

        sys_ids = global_tokenizer(template, add_special_tokens=False).input_ids
        system_input_ids = sys_ids + [mem_start]
        sys_len = len(system_input_ids)

        input_ids.extend(system_input_ids)
        segment_ids_1.extend([0] * sys_len)
        segment_ids_2.extend([0] * sys_len)
        chunk_ids.extend([-1] * sys_len)


        for idx in range(0,10):
            title = doc_list[idx]['title']
            text = doc_list[idx]['text']
            context = f"Document [{idx+1}](Title: {title}) {text}\n"

            sentences = context.split(". ")
            for j in range(len(sentences)):
                tem_id = global_tokenizer(sentences[j], add_special_tokens=False).input_ids
                
                input_ids += tem_id + anchor_id
                segment_ids_1 += [j+1] * (len(tem_id) +  anchor_num)
                segment_ids_2 += [1] * len(tem_id) + [2] *  anchor_num
                chunk_ids += [idx] * (len(tem_id) +  anchor_num)


        user_prompt = data[i]['question'] + "<|eot_id|>"
        user_id = [mem_end] + global_tokenizer(user_prompt, add_special_tokens=False).input_ids
        user_len = len(user_id)
        segment_ids_1.extend([0] * user_len)
        segment_ids_2.extend([0] * user_len)
        chunk_ids.extend([-1] * user_len)
        input_ids.extend(user_id)

        input_ids = torch.tensor([input_ids], device=global_model.device)
        segment_ids_1 = torch.tensor([segment_ids_1], device=global_model.device)
        segment_ids_2 = torch.tensor([segment_ids_2], device=global_model.device)
        chunk_ids = torch.tensor([chunk_ids], device=global_model.device)

        mask = make_anchor_attention(
            first_segments=segment_ids_1,
            second_segments=segment_ids_2,
            chunk_ids=chunk_ids,
            dtype=torch.bfloat16,           # For printing, let's keep float
            allow_self=True     # No causal mask for clarity
        )
        attention_mask = mask.unsqueeze(1).to(global_model.device)

        with torch.no_grad():
            outputs = global_model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                use_cache=True
            )
            
            past_key_values = outputs.past_key_values

        filtered_ids = filter_id(input_ids, segment_ids_2)
        filtered_kv = filter_kv(past_key_values, segment_ids_2)

        asst_prompt = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        asst_ids = global_tokenizer(asst_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(global_model.device)

        generate_ids = torch.cat([filtered_ids, asst_ids], dim = 1)

        global_model.eval()
        with torch.no_grad():

            outputs = global_model.generate(
                input_ids=generate_ids, 
                past_key_values=filtered_kv,
                max_new_tokens=200,
                do_sample=False,
                use_cache=True)
        # print(outputs)
        generated_seq = global_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response = generated_seq[0].split('assistant\n\n')[-1]
        print(data[i]['question'])
        print(response)

        score = best_subspan_em(response, [data[i]["answer"]])

        correct_num = correct_num + int(score)

        res_list.append({"id": str(i),"question": data[i]['question'], "response": response, "gold_answer": [data[i]["answer"]], "Score": score})
        print("Accuracy", correct_num / (i+1))

    accuracy = correct_num / total_num
    print(accuracy)

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d-%H%M%S")

    file_name = f"result/anllm/wiki_ckpt{ckpt}_{accuracy}_{time_str}.jsonl"

    with open(file_name, 'w', encoding='utf-8') as f:
        for entry in res_list:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

    print(f"Dumped at {file_name}")

if __name__ == "__main__":
    main()
