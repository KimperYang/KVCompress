dd1 = load_from_disk("dataset_cache/processed/compress_qa")
dd2 = load_from_disk("dataset_cache/processed/hqa")

from datasets import DatasetDict, concatenate_datasets

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
