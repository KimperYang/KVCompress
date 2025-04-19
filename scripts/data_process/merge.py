from datasets import DatasetDict, concatenate_datasets, load_from_disk

dd1 = load_from_disk("dataset_cache/processed/compress_qa")
dd2 = load_from_disk("dataset_cache/processed/hqa")

def drop_score(batch):
    batch["documents"] = [
        [{k: v for k, v in doc.items() if k != "score"} for doc in docs] 
        for docs in batch["documents"]
    ]
    return batch

dd1 = dd1.map(drop_score, batched=True, num_proc=8, load_from_cache_file=False)

merged = {}
for split in dd1.keys() | dd2.keys():                 # union of split names
    d1 = dd1.get(split)
    d2 = dd2.get(split)
    if d1 is not None and d2 is not None:
        drop = set(d1.column_names) - set(d2.column_names)
        d1 = d1.remove_columns(list(drop))
        merged[split] = concatenate_datasets([d1, d2])
    elif d1 is not None:
        merged[split] = d1
    else:
        merged[split] = d2

merged_dd = DatasetDict(merged)
merged_dd.save_to_disk("dataset_cache/processed/qa_aug")
