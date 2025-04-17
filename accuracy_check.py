import json
import math 
from tqdm import tqdm
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

    context_len = []
    context_num = []
    exceed_chunk_num_index = []
    for i in tqdm(range(len(data))):

        if dataset_name == "hqa":

            context_num.append(len(data[i]['documents']))
            accumulated_len = 0
            chunk_num = 0
            exceed_num = 0
            for j in range(len(data[i]['documents'])):
                title = data[i]['documents'][j]['title']
                text = data[i]['documents'][j]['text']
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                accumulated_len += len(tem_id)
                chunk_num += math.ceil(len(tem_id) / 100)
                if(len(tem_id) > 175):
                    exceed_num += 1
                
                # if exceed_num >=3:
                #     exceed_chunk_num_index.append(i)
                #     break
            # if chunk_num > 20:
            #     exceed_chunk_num_index.append(i)
            # context_len.append(accumulated_len / len(data[i]['documents']))
            if accumulated_len / len(data[i]['documents']) > 150:
                exceed_chunk_num_index.append(i)

        elif dataset_name == "tqa":

            context_num.append(len(data[i]['documents']))
            accumulated_len = 0
            chunk_num = 0
            for j in range(len(data[i]['documents'])):
                title = data[i]['documents'][j]['title']
                text = data[i]['documents'][j]['text']
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                accumulated_len += len(tem_id)
                chunk_num += math.ceil(len(tem_id) / 100)
            if chunk_num > 20:
                exceed_chunk_num_index.append(i)
            context_len.append(accumulated_len / len(data[i]['documents']))

        elif dataset_name == "wiki":

            context_num.append(len(data[i]['context']))
            accumulated_len = 0
            chunk_num = 0
            for j in range(len(data[i]['context'])):
                title = data[i]['context'][j][0]
                text = " ".join(data[i]['context'][j][1])
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                accumulated_len += len(tem_id)
                chunk_num += math.ceil(len(tem_id) / 100)
            if chunk_num > 20:
                exceed_chunk_num_index.append(i)
            context_len.append(accumulated_len / len(data[i]['context']))

        elif dataset_name == "nq":

            context_num.append(len(data[i]['documents']))
            accumulated_len = 0
            chunk_num = 0
            for j in range(len(data[i]['documents'])):
                title = data[i]['documents'][j]['title']
                text = data[i]['documents'][j]['text']
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                accumulated_len += len(tem_id)
                chunk_num += math.ceil(len(tem_id) / 100)
            if chunk_num > 20:
                exceed_chunk_num_index.append(i)
            context_len.append(accumulated_len / len(data[i]['documents']))
        
        elif dataset_name == "train":

            context_num.append(len(data[i]['documents']))
            accumulated_len = 0
            chunk_num = 0
            for j in range(len(data[i]['documents'])):
                title = data[i]['documents'][j]['title']
                text = data[i]['documents'][j]['text']
                tem_id = tokenizer(f"Document [{j+1}](Title: {title}) {text}\n", add_special_tokens=False).input_ids
                accumulated_len += len(tem_id)
                chunk_num += math.ceil(len(tem_id) / 100)
            if chunk_num > 20:
                exceed_chunk_num_index.append(i)
            context_len.append(accumulated_len / len(data[i]['documents']))

    filtered_list = [x for x in list(range(len(data))) if x not in exceed_chunk_num_index]
    return filtered_list
    # return exceed_chunk_num_index

# data_path1="result/ratio_compress_qa_kvlink_multichunk20k/hqa2_full_ckpt1122_0.5752869682646861.jsonl"
# data_path2="result/ratio_compress_qa_multichunk20k/hqa2_full_ckpt1122_0.5786630654962863.jsonl"
# data_path3="result/kvlink_qa_pretrain20k/hqa2_full_ckpt1122_0.625118163403106.jsonl"
data_path3="result/block_qa/hqa2_full_ckpt1122_0.6062120189061445.jsonl"
data_path1="result/compress_chunk_qa_kvlink_nopadding_multichunk20k_epoch2/hqa2_full_ckpt1122_0.57474679270763.jsonl"
data_path2="result/compress_chunk_qa_nopadding_multichunk20k_epoch2/hqa2_full_ckpt1122_0.5705604321404456.jsonl"
# data_path3="result/kvlink_qa_pretrain20k/hqa2_full_ckpt1122_0.625118163403106.jsonl"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

exceed_index = do_stats("hqa", tokenizer)
with open(data_path1, 'r') as file:
    data1 = [json.loads(line) for line in file]

with open(data_path2, 'r') as file:
    data2 = [json.loads(line) for line in file]

with open(data_path3, 'r') as file:
    data3 = [json.loads(line) for line in file]

print(exceed_index[0], len(exceed_index))

total_score_data1 = 0
total_score_data2 = 0
total_score_data3 = 0

for idx in exceed_index:
    total_score_data1 += data1[idx]['Score']
    total_score_data2 += data2[idx]['Score']
    total_score_data3 += data3[idx]['Score']

print("With kvlink accuracy: ", total_score_data1 / len(exceed_index))
print("Without kvlink accuracy: ", total_score_data2 / len(exceed_index))
print("No compression accuracy: ", total_score_data3 / len(exceed_index))