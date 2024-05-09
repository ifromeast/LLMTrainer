# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import logging
import os
import pathlib
from pathlib import Path
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, BitsAndBytesConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType

from function_calling.utils import get_system_prompt


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    use_func: bool = False

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=4096, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},)
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(trainer.model.named_parameters(), bias)
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, use_system: bool):
        super(SupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_system = use_system

        rank0_print("Formatting inputs...")
        self.messages = [example["conversations"] for example in raw_data]

        self.roles = {
                      "system": "<|im_start|>system",
                      "user": "<|im_start|>user", 
                      "assistant": "<|im_start|>assistant",
                      "tool": "<|im_start|>tool",
                      "observation": "<|im_start|>observation",
                    }

        self.im_start = tokenizer('<|im_start|>').input_ids
        self.im_end = tokenizer('<|im_end|>').input_ids
        self.nl_tokens = tokenizer('\n').input_ids
        self._system = tokenizer('system').input_ids + self.nl_tokens
        self._user = tokenizer('user').input_ids + self.nl_tokens
        self._assistant = tokenizer('assistant').input_ids + self.nl_tokens
        self._tool = tokenizer('tool').input_ids + self.nl_tokens
        self._observation = tokenizer('observation').input_ids + self.nl_tokens
        self.roles = {
                      "user": "<|im_start|>user", 
                      "assistant": "<|im_start|>assistant",
                      "system": "<|im_start|>system",
                      "observation": "<|im_start|>observation",
                      "tool": "<|im_start|>tool",
                      }

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.preprocess(self.messages[i])
    
    def preprocess(self, example) -> Dict:
        """Preprocesses the data for supervised fine-tuning."""

        if self.use_system and example[0]['role'] != 'special':
            if example[0]['role'] == 'system':
                example = example[1:]
            # system_prompt = get_system_prompt()
            example = [{"role": "system", "content": get_system_prompt()}] + example
        if example[0]['role'] == 'special':
            example = example[1:]

        input_ids, targets = [], []
        for j, sentence in enumerate(example):
            role = self.roles[sentence["role"]]
            _input_id = self.tokenizer(role).input_ids + self.nl_tokens + self.tokenizer(sentence["content"]).input_ids + self.im_end + self.nl_tokens
            input_ids += _input_id
            if role == '<|im_start|>system':
                _target = self.im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + self.im_end + self.nl_tokens
            elif role == '<|im_start|>user':
                _target = self.im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + self.im_end + self.nl_tokens
            elif role == '<|im_start|>assistant':
                _target = self.im_start + [IGNORE_TOKEN_ID] * len(self.tokenizer(role).input_ids) + _input_id[len(self.tokenizer(role).input_ids)+1:-2] + self.im_end + self.nl_tokens
            elif role == '<|im_start|>observation':
                _target = self.im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + self.im_end + self.nl_tokens
            elif role == '<|im_start|>tool':
                _target = self.im_start + [IGNORE_TOKEN_ID] * len(self.tokenizer(role).input_ids) + _input_id[len(self.tokenizer(role).input_ids)+1:-2] + self.im_end + self.nl_tokens
            else:
                raise TypeError(f"No role named {role}, please check it!")
            targets += _target
        assert len(input_ids) == len(targets)
        if len(input_ids) < self.max_len:
            input_ids += [self.tokenizer.pad_token_id] * (self.max_len - len(input_ids))
            targets += [IGNORE_TOKEN_ID] * (self.max_len - len(targets))
        else:
            input_ids = input_ids[:self.max_len]
            targets = targets[:self.max_len]

        input_ids = torch.tensor(input_ids, dtype=torch.int)
        targets = torch.tensor(targets, dtype=torch.int)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(input_ids=input_ids, labels=targets, attention_mask=attention_mask)


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    rank0_print("Loading data...")

    train_data = []
    dataset_path = Path(data_args.data_path)
    data_files = [file.name for file in dataset_path.glob("*.json")]
    for idx, file in enumerate(data_files):
        data_file = os.path.join(dataset_path, file)
        train_data += json.load(open(data_file, 'rb'))

    train_dataset = SupervisedDataset(train_data, tokenizer=tokenizer, max_len=max_len, use_system=data_args.use_func)

    return dict(train_dataset=train_dataset, eval_dataset=None)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if (
        getattr(training_args, "deepspeed", None)
        and int(os.environ.get("WORLD_SIZE", 1)) == 1
    ):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning("FSDP or ZeRO3 is incompatible with QLoRA.")

    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Load model and tokenizer
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if training_args.use_lora and lora_args.q_lora
        else None,
        **model_load_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length)

    # Start trainer
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # `not training_args.use_lora` is a temporary workaround for the issue that there are problems with
    # loading the checkpoint when using LoRA with DeepSpeed.
    # Check this issue https://github.com/huggingface/peft/issues/746 for more information.
    if (
        list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        and not training_args.use_lora
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
