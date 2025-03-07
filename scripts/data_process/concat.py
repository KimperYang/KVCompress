from datasets import load_from_disk, concatenate_datasets, DatasetDict

# Load your datasets

dataset1 = load_from_disk("dataset_cache/processed/compress_qa")
print(dataset1)
# dataset2 = load_from_disk("/data/jingbo_yang/KVMemory/dataset_cache/processed/block_qa/qa_mem")



# # Merge the datasets

# merged_dataset = DatasetDict({"train": concatenate_datasets([dataset1["train"], dataset2["train"]]),
#                               "test": concatenate_datasets([dataset1["test"], dataset2["test"]])})
# shards = {'train': 128, 'test': 4}
# merged_dataset.save_to_disk("dataset_cache/processed/compress_qa", num_shards=shards, num_proc=128)