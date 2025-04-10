import os
import torch
import string
import argparse
import json
import regex
import datasets

from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

def do_stats(dataset_name, tokenizer):
    if dataset_name == "hqa":
        file_path = "data/block_eval/hqa/eval.jsonl"
    elif dataset_name == "tqa":
        file_path = "data/block_eval/tqa/eval.jsonl"
    elif dataset_name == "wiki":
        file_path = "data/wiki/dev.json"
    elif dataset_name == "nq":
        file_path = "data/block_eval/nq/eval.jsonl"
    else:
        raise NotImplementedError

    if dataset_name == "wiki":
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]

    context_len = []
    context_num = []

    for i in range(len(data)):
        if dataset_name == "hqa":

            context_num.append(len(data[i]['documents']))
            accumulated_len = 0
            for j in range(len(data[i]['documents'])):
                title = data[i]['documents'][j]['title']
                text = data[i]['documents'][j]['text']
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                accumulated_len += len(tem_id)
            context_len.append(accumulated_len / len(data[i]['documents']))
        
        elif dataset_name == "tqa":

            context_num.append(len(data[i]['documents']))
            accumulated_len = 0
            for j in range(len(data[i]['documents'])):
                title = data[i]['documents'][j]['title']
                text = data[i]['documents'][j]['text']
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                accumulated_len += len(tem_id)
            context_len.append(accumulated_len / len(data[i]['documents']))

        elif dataset_name == "wiki":

            context_num.append(len(data[i]['context']))
            accumulated_len = 0
            for j in range(len(data[i]['context'])):
                title = data[i]['context'][j][0]
                text = " ".join(data[i]['context'][j][1])
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                accumulated_len += len(tem_id)
            context_len.append(accumulated_len / len(data[i]['context']))

        elif dataset_name == "nq":

            context_num.append(len(data[i]['documents']))
            accumulated_len = 0
            for j in range(len(data[i]['documents'])):
                title = data[i]['documents'][j]['title']
                text = data[i]['documents'][j]['text']
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                accumulated_len += len(tem_id)
            context_len.append(accumulated_len / len(data[i]['documents']))

    return sum(context_num) / len(context_num), sum(context_len) / len(context_len)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
print(do_stats("hqa", tokenizer))
print(do_stats("tqa", tokenizer))
print(do_stats("nq", tokenizer))
print(do_stats("wiki", tokenizer))