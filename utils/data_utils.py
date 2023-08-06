
from itertools import chain
from typing import List, Dict, Any, Mapping
import torch
import numpy as np
import logging
from datasets import load_dataset, load_from_disk

logger = logging.getLogger(__name__)


def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], return_tensors="pt")
   
# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def get_dataset(tokenizer, model_max_length, cache_dir):
    raw_dataset = load_dataset("wikipedia", "20220301.simple", split='train', cache_dir=cache_dir, keep_in_memory=False)
    logger.info("dataset has been loaded!")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=True,
        keep_in_memory=False,
        fn_kwargs={"tokenizer":tokenizer},
        desc="Running tokenizer on dataset",
    )
    grouped_datasets = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=8,
        load_from_cache_file=True,
        keep_in_memory=False,
        fn_kwargs={'block_size':model_max_length},
        desc=f"Grouping texts in chunks of {model_max_length}",
    )
    processed_dataset = grouped_datasets
    
    lm_datasets = processed_dataset.train_test_split(test_size = 0.01)
    # lm_datasets = lm_datasets['train']
    return lm_datasets



def fault_tolerance_data_collator(features: List) -> Dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.

    try:
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError: # quick fix by simply take the first example
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch
