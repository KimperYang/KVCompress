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
out_path = "dataset_cache/processed/hqa_fix"

qa  = load_from_disk(qa_path)
hqa = load_from_disk(hqa_path)

# Helper to iterate through splits uniformly
def as_split_dict(ds):
    if isinstance(ds, Dataset):
        return {"__single__": ds}
    return ds

fixed = {}

for split, h in as_split_dict(hqa).items():
    q = as_split_dict(qa)[next(iter(as_split_dict(qa)))]  # any qa split

    # ── 2. Make top‑level columns identical ─────────────────
    cols_q = set(q.column_names)
    cols_h = set(h.column_names)

    # 2‑a Drop extra columns in hqa
    to_drop = list(cols_h - cols_q)
    if to_drop:
        h = h.remove_columns(to_drop)

    # 2‑b Add missing columns to hqa (string default "")
    to_add = list(cols_q - cols_h)
    for col in to_add:
        h = h.add_column(col, np.array([""] * len(h), dtype=object))

    # Now h.column_names == q.column_names (order may differ)
    # Re‑order to match qa exactly (important for cast)
    h = h.select(range(len(h)))           # materialise to allow reordering
    h = h.rename_columns({c: c for c in h.column_names})  # keeps original order
    h = h.with_format(None)               # reset any Arrow formatting

    # ── 3. Remove "score" inside each documents entry ───────
    def strip_score(batch):
        batch["documents"] = [
            [{k: v for k, v in doc.items() if k != "score"} for doc in docs]
            for docs in batch["documents"]
        ]
        return batch

    h = h.map(strip_score, batched=True, num_proc=8, load_from_cache_file=False)

    # ── 4. Cast to qa.features ───────────────────────────────
    h = h.cast(q.features)

    fixed[split] = h

# ─── 5. Save ────────────────────────────────────────────────
if "__single__" in fixed:
    fixed["__single__"].save_to_disk(out_path)      # Dataset
else:
    DatasetDict(fixed).save_to_disk(out_path)       # DatasetDict

print("✅  hqa dataset aligned and saved to", out_path)
