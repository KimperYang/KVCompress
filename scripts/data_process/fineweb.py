"""
download the fineweb pre-training corpus

```
python scripts/data_process/fineweb.py --num_samples=10000000 --validation_size=3000 --min_length=2048
```
"""

from typing import Any, Dict, List

from absl import app, flags
from datasets import load_dataset
from transformers import AutoTokenizer

FLAGS = flags.FLAGS

def set_args():
    flags.DEFINE_integer(
        "num_samples",
        default=10_000_000,
        help="number of samples to sample from the FineWeb.",
    )
    flags.DEFINE_integer(
        "validation_size",
        default=3_000,
        help="number of samples to sample from the FineWeb.",
    )
    flags.DEFINE_integer(
        "min_length",
        default=2048,
        help="Minimum length for FineWeb.",
    )



def main(argv):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    num_samples = FLAGS.num_samples
    min_length = FLAGS.min_length
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train"
    )

    random_seed = 42

    def tokenize_texts(examples: Dict[str, List[Any]]):
        token_counts = tokenizer(examples["text"], add_special_tokens= False)["input_ids"]
        examples["num_tokens"] = [len(x) for x in token_counts]
        return examples

    dataset_with_token_num = dataset.map(tokenize_texts, batched=True, num_proc=192)

    def filter_fn(examples: Dict[str, List[Any]]):
        token_counts = examples["num_tokens"]
        return [x > min_length for x in token_counts]
    
    filtered_dataset = dataset_with_token_num.filter(filter_fn, batched=True, num_proc=192)
    filtered_dataset = filtered_dataset.remove_columns("num_tokens")

    # text = dataset.shuffle(seed=random_seed).select(range(0, num_samples//2))
    # text_compress = dataset.shuffle(seed=random_seed).select(range(num_samples//2, num_samples))

    text = filtered_dataset.train_test_split(test_size=FLAGS.validation_size)
    # text_compress = text_compress.train_test_split(test_size=FLAGS.validation_size)

    print("text:", text)
    shards = {'train': 128, 'test': 4}
    text.save_to_disk(f"dataset_cache/processed/fineweb/text_min{min_length}", num_shards=shards, num_proc=128)

if __name__ == "__main__":
    set_args()
    app.run(main)

