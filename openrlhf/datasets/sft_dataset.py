import os
from pathlib import Path
from typing import Optional, Dict
import json

import torch
from torch.utils.data import Dataset
import transformers
from openrlhf.datasets.utils import zero_pad_sequences, get_system_prompt


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset_dir, tokenizer, model_max_length, use_system=False):
        super(SupervisedDataset, self).__init__()
        self.data = []
        dataset_path = Path(dataset_dir)
        data_files = [file.name for file in dataset_path.glob("*.json")]
        for idx, file in enumerate(data_files):
            data_file = os.path.join(dataset_path, file)
            self.data += json.load(open(data_file, 'rb'))

        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

        self.system_tokens = [194]
        self.user_tokens = [195]
        self.assistant_tokens = [196]
        self.tool_tokens = [196]
        self.observation_tokens = [197]
        self.ignore_index = -100
        self.use_system = use_system

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        input_ids = []
        labels = []
        if self.use_system and example["conversations"][0]['role'] != 'special':
            if example["conversations"][0]['role'] == 'system':
                example["conversations"] = example["conversations"][1:]
            system_prompt = get_system_prompt()
            example["conversations"] = [{"role": "system", "content": system_prompt}] + example["conversations"]
        # elif not self.use_system and example["conversations"][0]['role'] == 'system':
        #     example["conversations"] = example["conversations"][1:]
        if example["conversations"][0]['role'] == 'special':
            example["conversations"] = example["conversations"][1:]

        for message in example["conversations"]:
            role = message["role"]
            value = message["content"]
            value_ids = self.tokenizer.encode(value)

            if role == "system":
                input_ids += self.system_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            elif role == "user":
                input_ids += self.user_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            elif role == "assistant" or role == "tool":
                input_ids += self.assistant_tokens + value_ids
                labels += [self.ignore_index] + value_ids
            elif role == "observation":
                input_ids += self.observation_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            else:
                raise TypeError(f"No role named {role}, please check it!")
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        input_ids = input_ids[: self.model_max_length]
        labels = labels[: self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return input_ids, labels, attention_mask, example

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

    def collate_fn(self, item_list):
        input_ids = []
        labels = []
        attention_masks = []
        infos = {"conversations": []}

        for input_id, label, attention_mask, example in item_list:
            input_ids.append(input_id)
            labels.append(label)
            attention_masks.append(attention_mask)
            infos["conversations"].append(example["conversations"])

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        labels = zero_pad_sequences(labels, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return input_ids, labels, attention_masks, infos


if __name__ == "__main__":
    import pdb

    model_name_or_path = "/data/share_user/zzd/ckpt/Baichuan2-13B-Base/"
    sft_dataset = "/data/share_user/zzd/data/rlhf_data/sft_data_v4/"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=4096,
    )

    dataset = SupervisedDataset(sft_dataset, tokenizer, 1024, use_system=True)

    pdb.set_trace()
    dataset.preprocessing(dataset.data[1])
