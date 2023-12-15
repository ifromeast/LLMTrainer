import os
from typing import Dict, Optional, Sequence
from pathlib import Path
import logging
import torch
from torch.utils.data import Dataset
import datasets
from datasets import concatenate_datasets, load_dataset, Dataset
import transformers
from conversation import get_conv_template

logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = -100


def preprocess(sources, tokenizer: transformers.PreTrainedTokenizer, ) -> Dict:
    conv = get_conv_template("baichuan2")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    source = sources["conversations"]
    if roles[source[0]["from"]] != conv.roles[0]:
        # Skip the first one if it is not from human
        source = source[1:]

    conv.messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{source}"
        conv.append_message(role, sentence["value"])
    conversation = conv.get_prompt()

    # Tokenize conversations
    input_ids = tokenizer(conversation, return_tensors="pt", padding="max_length",
                          max_length=tokenizer.model_max_length, truncation=True, ).input_ids[0]
    target = input_ids.clone()

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    total_len = int(target.ne(tokenizer.pad_token_id).sum())

    rounds = conversation.split(conv.sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_TOKEN_ID
    for i, rou in enumerate(rounds):
        if rou == "":
            break

        parts = rou.split(sep)
        tmp = ""
        if len(parts) != 2:
            tmp = parts[0]
            break
        parts[0] += sep
        round_len = len(tokenizer(rou).input_ids)
        instruction_len = len(tokenizer(parts[0]).input_ids) - 2
        target[cur_len:cur_len + instruction_len] = IGNORE_TOKEN_ID

        cur_len += round_len
    target[cur_len:] = IGNORE_TOKEN_ID

    if cur_len < tokenizer.model_max_length:
        if cur_len != total_len and (cur_len+len(tokenizer(tmp).input_ids)-1) != total_len:
            target[:] = IGNORE_TOKEN_ID
            print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}.")
            print(source)

    return dict(input_ids=input_ids, labels=target, attention_mask=input_ids.ne(tokenizer.pad_token_id), )

def get_multiturn_dataset(data_args, tokenizer) -> "Dataset":
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
                preprocess,
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


def generate_prompt_with_history(tokenizer, device, history, max_length=2048):
    text = input("USER: ")
    prompt = "The following is a conversation between a human and an AI assistant named Baize (named after a mythical creature in Chinese folklore). Baize is an open-source AI assistant developed by UCSD and Sun Yat-Sen University. The human and the AI assistant take turns chatting. Human statements start with [|Human|] and AI assistant statements start with [|AI|]. The AI assistant always provides responses in as much detail as possible, and in Markdown format. The AI assistant always declines to engage with topics, questions and instructions related to unethical, controversial, or sensitive issues. Complete the transcript in exactly that format.\n[|Human|]Hello!\n[|AI|]Hi!"
    history = ["\n[|Human|]{}\n[|AI|]{}".format(x[0], x[1]) for x in history]
    history.append("\n[|Human|]{}\n[|AI|]".format(text))
    history_text = ""
    flag = False
    for x in history[::-1]:
        if tokenizer(prompt + history_text + x, return_tensors="pt")["input_ids"].size(-1) <= max_length:
            history_text = x + history_text
            flag = True
        else:
            break
    if flag:
        inputs = tokenizer(prompt + history_text, return_tensors="pt")
        input_ids = inputs["input_ids"][:, -max_length:].to(device)
        torch.cuda.empty_cache()
        return input_ids, text
    else:
        return None


def sample_decode(
        input_ids: torch.Tensor,
        model: torch.nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        stop_words: list,
        max_length: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 25,
):
    generated_tokens = []
    past_key_values = None
    current_length = 1
    for i in range(max_length):
        with torch.no_grad():
            if past_key_values is None:
                outputs = model(input_ids)
            else:
                outputs = model(input_ids[:, -1:], past_key_values=past_key_values)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

        # apply temperature
        logits /= temperature

        probs = torch.softmax(logits, dim=-1)
        # apply top_p
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0

        # apply top_k
        # if top_k is not None:
        #    probs_sort1, _ = torch.topk(probs_sort, top_k)
        #    min_top_probs_sort = torch.min(probs_sort1, dim=-1, keepdim=True).values
        #    probs_sort = torch.where(probs_sort < min_top_probs_sort, torch.full_like(probs_sort, float(0.0)), probs_sort)

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        input_ids = torch.cat((input_ids, next_token), dim=-1)

        generated_tokens.append(next_token[0].item())
        text = tokenizer.decode(generated_tokens)

        yield text
        if any([x in text for x in stop_words]):
            return


def is_stop_word_or_prefix(s: str, stop_words: list) -> bool:
    for stop_word in stop_words:
        if s.endswith(stop_word):
            return True
        for i in range(1, len(stop_word)):
            if s.endswith(stop_word[:i]):
                return True
    return False

