import os
import torch
import string
import argparse
import json
import regex
import datasets
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    elif dataset_name == "train":
        file_path = "dataset_cache/processed/compress_qa"
    else:
        raise NotImplementedError

    if dataset_name == "wiki":
        with open(file_path, 'r') as file:
            data = json.load(file)
    elif dataset_name == "train":
        data = datasets.load_from_disk(file_path)['train']
    else:
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]

    # context_len = []
    context_num = []
    chunk_num_list = []
    doc_length_list = []

    for i in tqdm(range(len(data))):

        if dataset_name == "hqa":

            context_num.append(len(data[i]['documents']))
            chunk_num = 0
            for j in range(len(data[i]['documents'])):
                title = data[i]['documents'][j]['title']
                text = data[i]['documents'][j]['text']
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                doc_length_list.append(len(tem_id))
                chunk_num += math.ceil(len(tem_id) / 100)
            chunk_num_list.append(chunk_num)
        
        elif dataset_name == "tqa":

            context_num.append(len(data[i]['documents']))
            accumulated_len = 0
            chunk_num = 0
            for j in range(len(data[i]['documents'])):
                title = data[i]['documents'][j]['title']
                text = data[i]['documents'][j]['text']
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                doc_length_list.append(len(tem_id))
                chunk_num += math.ceil(len(tem_id) / 100)
            chunk_num_list.append(chunk_num)
            

        elif dataset_name == "wiki":

            context_num.append(len(data[i]['context']))
            accumulated_len = 0
            chunk_num = 0
            for j in range(len(data[i]['context'])):
                title = data[i]['context'][j][0]
                text = " ".join(data[i]['context'][j][1])
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                doc_length_list.append(len(tem_id))
                chunk_num += math.ceil(len(tem_id) / 100)
            chunk_num_list.append(chunk_num)

        elif dataset_name == "nq":

            context_num.append(len(data[i]['documents']))
            accumulated_len = 0
            chunk_num = 0
            for j in range(len(data[i]['documents'])):
                title = data[i]['documents'][j]['title']
                text = data[i]['documents'][j]['text']
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                doc_length_list.append(len(tem_id))
                chunk_num += math.ceil(len(tem_id) / 100)
            chunk_num_list.append(chunk_num)
            
        
        elif dataset_name == "train":

            context_num.append(len(data[i]['documents']))
            accumulated_len = 0
            chunk_num = 0
            for j in range(len(data[i]['documents'])):
                title = data[i]['documents'][j]['title']
                text = data[i]['documents'][j]['text']
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                doc_length_list.append(len(tem_id))
                chunk_num += math.ceil(len(tem_id) / 100)
            chunk_num_list.append(chunk_num)
            

    # if dataset_name == "train":
    return chunk_num_list, doc_length_list
    # return sum(context_num) / len(context_num), sum(context_len) / len(context_len)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Checkpoint number')
args = parser.parse_args()
dataset_name = args.dataset

chunk_num_list, doc_length_list = do_stats(dataset_name, tokenizer)
# print("Max chunk num: ", max(chunk_num_list))
# print("Avg chunk num: ", sum(chunk_num_list) / len(chunk_num_list))


# data = chunk_num_list
# # Compute statistics for reference (optional)
# mean_val = np.mean(data)
# median_val = np.median(data)

# # Set up bins so each integer has its own bin
# # +2 on max(data) because range() is exclusive at the upper boundary
# bins = range(min(data), max(data) + 2)

# plt.hist(data, bins=bins, align='left', alpha=0.8)
# plt.axvline(mean_val, linestyle='--', label=f"Mean = {mean_val:.2f}")
# # plt.axvline(median_val, linestyle=':', label=f"Median = {median_val:.2f}")
# plt.axvline(max(data), linestyle=':', label=f"Max = {max(data):.2f}")

# plt.title("Distribution of chunk num")
# plt.xlabel("Chunk number")
# plt.ylabel("Data number")
# plt.grid(True)
# plt.legend()
# plt.savefig(f"figs/{dataset_name}_500.png")  

print("Max doc length: ", max(doc_length_list))
print("Avg doc length: ", sum(doc_length_list) / len(doc_length_list))


data = doc_length_list
# Compute statistics for reference (optional)
# mean_val = np.mean(data)
# median_val = np.median(data)

# # Set up bins so each integer has its own bin
# # +2 on max(data) because range() is exclusive at the upper boundary
# bins = range(min(data), max(data) + 2)

# plt.hist(data, bins=bins, align='left', alpha=0.8)
# plt.axvline(mean_val, linestyle='--', label=f"Mean = {mean_val:.2f}")
# # plt.axvline(median_val, linestyle=':', label=f"Median = {median_val:.2f}")
# plt.axvline(max(data), linestyle=':', label=f"Max = {max(data):.2f}")

# plt.title("Distribution of Doc Length")
# plt.xlabel("Document token length")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.legend()

# Filter data to include only 0 <= x <= 500
filtered_data = [x for x in data if 0 <= x <= 500]

# Use Seaborn's whitegrid style for a clean look
sns.set(style="whitegrid")

# Construct bins of width 25 from 0 to 500 (inclusive)
# range(0, 501, 25) creates edges: 0, 25, 50, ..., 500
bins = range(0, 501, 25)

# Plot histogram:
#  - bins: controls our 25-unit intervals
#  - stat='density': normalizes the histogram (heights correspond to probability density)
#  - kde=False: turn off the kernel density curve
ax = sns.histplot(filtered_data, bins=bins, stat='density', kde=False)

# Labels and title
ax.set_title("Distribution of Doc Length")
ax.set_xlabel("Document token length")
ax.set_ylabel("Frequency")
plt.savefig(f"figs/{dataset_name}.png")  