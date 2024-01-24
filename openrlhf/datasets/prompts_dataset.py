import os
import ast
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset
import transformers
from openrlhf.datasets.utils import exist_and_not_none, zero_pad_sequences, get_system_prompt
import pdb


class PromptsDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, dataset_dir, tokenizer, model_max_length=4096, use_system=False):
        super(PromptsDataset, self).__init__()
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

        context = ""
        for text in history:
            if text["role"] == "system":
                context += self.tokenizer.decode(self.system_tokens) + text['content']
            elif text["role"] == "user":
                context += self.tokenizer.decode(self.user_tokens) + text['content']
            elif text["role"] == "assistant" or text["role"] == "tool":
                context += self.tokenizer.decode(self.assistant_tokens) + text['content'] + self.tokenizer.eos_token
            elif text["role"] == "observation":
                context += self.tokenizer.decode(self.observation_tokens) + text['content']
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
            infos["conversations"].append(example["history"] + [{"role": "user", "content": example['query']}])
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

