import os
import ast
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset
import transformers
from openrlhf.datasets.utils import exist_and_not_none, zero_pad_sequences
import pdb

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, dataset_dir, tokenizer, model_max_length, user_tokens=[195], assistant_tokens=[196],):
        super(SupervisedDataset, self).__init__()
        self.data = []
        dataset_path = Path(dataset_dir)
        data_files = [file.name for file in dataset_path.glob("*.json")]
        for idx, file in enumerate(data_files):
            data_file = os.path.join(dataset_path, file)
            self.data += json.load(open(data_file, 'rb'))

        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        input_ids = []
        labels = []

        for message in example["conversations"]:
            from_ = message["role"]
            value = message["content"]
            value_ids = self.tokenizer.encode(value)

            if from_ == "user":
                input_ids += self.user_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            else:
                input_ids += self.assistant_tokens + value_ids
                labels += [self.ignore_index] + value_ids
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


class RewardDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, dataset_dir, tokenizer, model_max_length, user_tokens=[195], assistant_tokens=[196],):
        super(RewardDataset, self).__init__()
        self.data = []
        dataset_path = Path(dataset_dir)
        data_files = [file.name for file in dataset_path.glob("*.json")]
        for idx, file in enumerate(data_files):
            data_file = os.path.join(dataset_path, file)
            self.data += json.load(open(data_file, 'rb'))

        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        history = example['history']
        history += [{"role": "user", "content": example['query']}]

        chosen_message = history + [{"role": "assistant", "content": example['chosen']}]
        rejected_message = history + [{"role": "assistant", "content": example['rejected']}]

        chosen_ids, chosen_mask = self.build_chat_input(chosen_message)
        rejected_ids, rejected_mask = self.build_chat_input(rejected_message)

        return chosen_ids, chosen_mask, rejected_ids, rejected_mask

    
    def build_chat_input(self, messages, max_new_tokens=2048):
        assert max_new_tokens <= self.model_max_length, "max_new_tokens must be less than model_max_length"

        def _parse_messages(messages, split_role="user"):
            system, rounds = "", []
            round = []
            for i, message in enumerate(messages):
                if message["role"] == "system":
                    assert i == 0
                    system = message["content"]
                    continue
                if message["role"] == split_role and round:
                    rounds.append(round)
                    round = []
                round.append(message)
            if round:
                rounds.append(round)
            return system, rounds

        max_input_tokens = self.model_max_length - max_new_tokens
        system, rounds = _parse_messages(messages, split_role="user")
        system_tokens = self.tokenizer.encode(system)
        max_history_tokens = max_input_tokens - len(system_tokens)

        history_tokens = []
        for round in rounds[::-1]:
            round_tokens = []
            for message in round:
                if message["role"] == "user":
                    round_tokens += self.user_tokens
                else:
                    round_tokens += self.assistant_tokens
                round_tokens.extend(self.tokenizer.encode(message["content"]))
            if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
                history_tokens = round_tokens + history_tokens  # concat left
                if len(history_tokens) < max_history_tokens:
                    continue
            break

        input_tokens = system_tokens + history_tokens
        if messages[-1]["role"] != "assistant":
            input_tokens.append(self.assistant_tokens[0])
        input_tokens = torch.LongTensor([input_tokens[-max_input_tokens:]])  # truncate left
        input_mask = torch.ones_like(input_tokens)
        return input_tokens, input_mask

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        for chosen_id, chosen_mask, reject_id, rejects_mask in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
        rejects_masks = zero_pad_sequences(rejects_masks)
        return chosen_ids, chosen_masks, reject_ids, rejects_masks


class PromptsDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, dataset_dir, tokenizer, model_max_length=4096, user_tokens=[195], assistant_tokens=[196],):
        super(PromptsDataset, self).__init__()
        self.data = []
        dataset_path = Path(dataset_dir)
        data_files = [file.name for file in dataset_path.glob("*.json")]
        for idx, file in enumerate(data_files):
            data_file = os.path.join(dataset_path, file)
            self.data += json.load(open(data_file, 'rb'))

        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        history = example['history']
        history += [{"role": "user", "content": example['query']}]

        context = ""
        for text in history:
            if text["role"] == "user":
                context += self.tokenizer.decode(self.user_tokens) + text['content']
            elif text["role"] == "assistant":
                context += self.tokenizer.decode(self.assistant_tokens) + text['content'] + self.tokenizer.eos_token
            else:
                raise ValueError(f"message role not supported yet: {text['role']}")

        context += self.tokenizer.decode(self.assistant_tokens)

        return context, example

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

    def collate_fn(self, item_list):
        context_list = []
        infos = {"conversations": []}
        for context, example in item_list:
            context_list.append(context)
            infos["conversations"].append(example["history"]+[{"role": "user", "content": example['query']}])
        return context_list, infos


if __name__ == "__main__":
    import argparse
    from openrlhf.utils import get_strategy
    import pdb
    model_name_or_path = "/data/share_user/zzd/ckpt/Baichuan2-13B-Base/"
    sft_dataset = "/data/share_user/zzd/data/rlhf_data/sft_data/"
    reward_dataset = "/data/share_user/zzd/data/rlhf_data/comparison_data/"
    prompt_dataset = "/data/share_user/zzd/data/rlhf_data/prompt_data/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--prompt_max_len", type=int, default=2048)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--inference_tp_size", type=int, default=1)
    parser.add_argument("--greedy_sampling", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--flash_attn", action="store_true", default=False)

    # batch inference
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()


    strategy = get_strategy(args)
    strategy.setup_distributed()


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=4096,
    )

    dataset = SupervisedDataset(sft_dataset, tokenizer, 1024)
    # dataset = RewardDataset(reward_dataset, tokenizer, 4096)
    # dataset = PromptsDataset(prompt_dataset, tokenizer, 4096)

    dataloader = strategy.setup_dataloader(dataset, 4, pin_memory=True, shuffle=True, collate_fn=dataset.collate_fn)
    
    pdb.set_trace()
    aa = next(iter(dataloader))
    dataset.preprocessing(dataset.data[1])

