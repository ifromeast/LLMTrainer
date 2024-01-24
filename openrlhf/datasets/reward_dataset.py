import os
from pathlib import Path
from typing import Dict
import json

import torch
from torch.utils.data import Dataset
import transformers
from openrlhf.datasets.utils import zero_pad_sequences, get_system_prompt


class RewardDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset_dir, tokenizer, model_max_length, use_system=False):
        super(RewardDataset, self).__init__()
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
        history = example['history']
        if self.use_system:
            if history and history[0]['role'] != 'system':
                system_prompt = get_system_prompt()
                history = [{"role": "system", "content": system_prompt}] + history

        if history and history[-1]["role"] == "tool":
            history += [{"role": "observation", "content": example['query']}]
        else:
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
        system_tokens = self.system_tokens + self.tokenizer.encode(system)
        max_history_tokens = max_input_tokens - len(system_tokens)

        history_tokens = []
        for round in rounds[::-1]:
            round_tokens = []
            for message in round:
                if message["role"] == "system":
                    round_tokens += self.system_tokens
                elif message["role"] == "user":
                    round_tokens += self.user_tokens
                elif message["role"] == "assistant" or message["role"] == "tool":
                    round_tokens += self.assistant_tokens
                elif message["role"] == "observation":
                    round_tokens += self.observation_tokens
                else:
                    raise TypeError(f"No role, please check it!")

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


if __name__ == "__main__":
    import pdb

    model_name_or_path = "/data/share_user/zzd/ckpt/Baichuan2-13B-Base/"
    reward_dataset = "/data/share_user/zzd/data/rlhf_data/comparison_data/"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=4096,
    )

    dataset = RewardDataset(reward_dataset, tokenizer, 4096)

    pdb.set_trace()
    dataset.preprocessing(dataset.data[1])

