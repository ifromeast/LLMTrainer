import os
import copy
from pathlib import Path
import logging
import datasets
from datasets import concatenate_datasets, load_dataset, Dataset

logger = logging.getLogger(__name__)

# PROMPT_DICT = {
#     "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
#     "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:\n"}
PROMPT_DICT = {
    "prompt_input": "{instruction}\n问:\n{input}\n答:\n",
    "prompt_no_input": "问:\n{instruction}\n答:\n"}

IGNORE_INDEX = -100


def generate_prompt(data_point):
    prompt_ = PROMPT_DICT['prompt_input'] if data_point["input"] else PROMPT_DICT['prompt_no_input']
    return prompt_.format_map(data_point)


def tokenize(tokenizer, prompt):
    result = tokenizer(prompt, truncation=True,  max_length=tokenizer.model_max_length, padding=False)
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
        "labels": copy.deepcopy(result["input_ids"])
    }

def generate_and_tokenize_prompt(data_point, tokenizer):
    tokenizer.add_eos_token = False
    prompt_no_resp = generate_prompt(data_point)
    tokenized_result = tokenize(tokenizer, prompt_no_resp)
    source_len = len(tokenized_result['input_ids'])

    tokenizer.add_eos_token = True
    prompt_with_response = prompt_no_resp + data_point["output"].strip()
    tokenized_with_response = tokenize(tokenizer, prompt_with_response)
    tokenized_with_response["labels"] = [IGNORE_INDEX] * source_len + tokenized_with_response["labels"][source_len:]
    return tokenized_with_response


def get_sft_dataset(data_args, tokenizer) -> "Dataset":
    dataset_path = Path(data_args.dataset_dir)
    data_files = [file.name for file in dataset_path.glob("*.json")]

    for idx, file in enumerate(data_files):
        data_file = os.path.join(dataset_path, file)
        filename = ''.join(file.split(".")[:-1])
        cache_path = os.path.join(data_args.data_cache_dir, filename)
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path, keep_in_memory=False)
            logger.info(f'training datasets-{filename} has been loaded from disk')
        except Exception:
            cache_dir = os.path.join(data_args.data_cache_dir, filename + "_json")
            os.makedirs(cache_dir, exist_ok=True)
            raw_dataset = load_dataset("json", data_files=data_file, cache_dir=cache_dir, keep_in_memory=False)
            logger.info(f"{file} has been loaded")
            tokenized_dataset = raw_dataset.map(
                generate_and_tokenize_prompt,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=raw_dataset.column_names['train'],
                load_from_cache_file=True,
                keep_in_memory=False,
                cache_file_names={k: os.path.join(cache_dir, 'tokenized.arrow') for k in raw_dataset},
                fn_kwargs={'tokenizer': tokenizer},
                desc="Running tokenizer on dataset"
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        if idx == 0:
            lm_datasets = processed_dataset['train']
        else:
            assert lm_datasets.features.type == processed_dataset["train"].features.type
            lm_datasets = concatenate_datasets([lm_datasets, processed_dataset["train"]])

    lm_datasets = lm_datasets.train_test_split(test_size=data_args.validation_split_percentage)

    train_dataset = lm_datasets['train']
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    logger.info(f"Num train_samples  {len(train_dataset)}")
    logger.info("training example:")
    logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
    return lm_datasets

