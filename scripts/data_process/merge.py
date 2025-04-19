# from datasets import DatasetDict, concatenate_datasets, load_from_disk

# dd1 = load_from_disk("dataset_cache/processed/compress_qa")
# dd2 = load_from_disk("dataset_cache/processed/hqa")

# def drop_score(batch):
#     batch["documents"] = [
#         [{k: v for k, v in doc.items() if k != "score"} for doc in docs] 
#         for docs in batch["documents"]
#     ]
#     return batch

# dd1 = dd1.map(drop_score, batched=True, num_proc=8, load_from_cache_file=False)

# merged = {}
# for split in dd1.keys() | dd2.keys():                 # union of split names
#     d1 = dd1.get(split)
#     d2 = dd2.get(split)
#     if d1 is not None and d2 is not None:
#         drop = set(d1.column_names) - set(d2.column_names)
#         d1 = d1.remove_columns(list(drop))
#         merged[split] = concatenate_datasets([d1, d2])
#     elif d1 is not None:
#         merged[split] = d1
#     else:
#         merged[split] = d2

# merged_dd = DatasetDict(merged)
# merged_dd.save_to_disk("dataset_cache/processed/qa_aug")

#!/usr/bin/env python
# -------------------------------------------------------------
# Align the nested "documents" schema of hqa_dataset to qa_dataset
# -------------------------------------------------------------
from datasets import load_from_disk, Dataset, DatasetDict

# ─── 1. Paths ────────────────────────────────────────────────
qa_path  = "dataset_cache/processed/compress_qa"   # reference dataset (has the full schema)
hqa_path = "dataset_cache/processed/hqa"  # dataset that is missing some keys
out_path = "dataset_cache/processed/hqa"

# ─── 2. Load both datasets (works for Dataset or DatasetDict) ─
qa  = load_from_disk(qa_path)
hqa = load_from_disk(hqa_path)

# Helper: make {split_name: dataset} no matter the original type
def splits(dset_or_dict):
    if isinstance(dset_or_dict, Dataset):     # single split
        return {"__single__": dset_or_dict}
    return dset_or_dict                       # already a DatasetDict

fixed_splits = {}

# ─── 3. Align every split that exists in hqa ──────────────────
for split, hqa_split in splits(hqa).items():
    qa_split = splits(qa)[next(iter(splits(qa)))]  # any split in qa works

    # 3‑a. Which nested keys exist in qa["documents"]?
    doc_template = qa_split[0]["documents"][0]     # {'title':.., 'text':.., 'score':..}
    nested_keys  = list(doc_template.keys())

    # 3‑b. map() that adds the missing keys to each nested doc
    def add_missing_keys(batch, default_map):
        batch["documents"] = [
            [
                {k: doc.get(k, default_map[k]) for k in default_map}
                for doc in docs
            ]
            for docs in batch["documents"]
        ]
        return batch

    # Build a default value map with sensible dtypes
    default_map = {
        k: (0.0 if isinstance(v, (int, float)) else "")
        for k, v in doc_template.items()
    }

    hqa_fixed = hqa_split.map(
        add_missing_keys,
        batched=True,
        fn_kwargs={"default_map": default_map},
        load_from_cache_file=False,
        num_proc=8,            # speed‑up; adjust to your CPU
    )

    # 3‑c. Cast Arrow schema to be *exactly* identical to qa
    hqa_fixed = hqa_fixed.cast(qa_split.features)

    fixed_splits[split] = hqa_fixed

# ─── 4. Reassemble & save ─────────────────────────────────────
if "__single__" in fixed_splits:                # original was Dataset
    fixed_splits["__single__"].save_to_disk(out_path)
else:                                           # original was DatasetDict
    DatasetDict(fixed_splits).save_to_disk(out_path)

print("✅  hqa dataset rewritten with aligned schema ->", out_path)
