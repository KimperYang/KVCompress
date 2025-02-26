"""
download the fineweb pre-training corpus

```
python scripts/data_process/fineweb.py --num_samples=10000000 --min_length_for_memory=2048 --validation_size=3000
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




def main(argv):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    num_samples = FLAGS.num_samples
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train"
    )

    random_seed = 42
    text = dataset.shuffle(seed=random_seed).select(range(0, num_samples//2))
    text_compress = dataset.shuffle(seed=random_seed).select(range(num_samples//2, num_samples))

    text = text.train_test_split(test_size=FLAGS.validation_size)
    text_compress = text_compress.train_test_split(test_size=FLAGS.validation_size)

    print("text:", text, "textcompress:", text_compress)
    shards = {'train': 128, 'test': 4}
    text.save_to_disk("dataset_cache/processed/fineweb/text", num_shards=shards, num_proc=128)
    text_compress.save_to_disk("dataset_cache/processed/fineweb/text_compress", num_shards=shards, num_proc=128)

if __name__ == "__main__":
    set_args()
    app.run(main)

