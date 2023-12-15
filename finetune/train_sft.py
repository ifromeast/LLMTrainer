import copy
from dataclasses import dataclass, field
import math
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import torch
from datasets import load_dataset, load_from_disk
import transformers
from transformers import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_sft_dataset

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model name or config path")},
    )


@dataclass
class DataArguments:
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": ("For debugging purposes or quicker training, truncate the number of "
                           " training examples to this value if set.")}, )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    preprocessing_num_workers: Optional[int] = field(
        default=32,
        metadata={"help": "The number of processes to use for the preprocessing."}, )
    data_cache_dir: Optional[str] = field(default="./", metadata={"help": "The datasets processed stored"})
    validation_split_percentage: Optional[float] = field(
        default=0.001,
        metadata={"help": "The percentage of the finetune set used as validation set"}, )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    flash_attn: Optional[bool] = field(default=False)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.flash_attn:
        from pretrain.flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
#         model_max_length=training_args.model_max_length,
        use_fast=False,
        trust_remote_code=True
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token

    train_data = get_sft_dataset(data_args, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        trust_remote_code=True,
#         low_cpu_mem_usage=True
    ).cuda()

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_data['train'] if training_args.do_train else None,
                      eval_dataset=train_data['train'] if training_args.do_eval else None,
                      data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True),
                      )

    # Training
    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            train_result = trainer.train(resume_from_checkpoint=True)
        else:
            train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    train()
