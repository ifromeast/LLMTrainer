import argparse
import logging
import pathlib
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import transformers
from transformers import Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_utils import get_dataset, fault_tolerance_data_collator

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for models."""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model & tokenizer name or config path")}, )


@dataclass
class DataArguments:
    """Arguments for datasets."""
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."} )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": ("For debugging purposes or quicker training, truncate the number of "
                           " training examples to this value if set.")}, )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    preprocessing_num_workers: Optional[int] = field(
        default=32,
        metadata={"help": "The number of processes to use for the preprocessing."},)
    data_cache_dir: Optional[str] = field(default="./tmp", metadata={"help": "The datasets processed stored"})
    validation_split_percentage: Optional[float] = field(
        default=0.01,
        metadata={"help": "The percentage of the finetune set used as validation set"},)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments for the training loop."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={'help': 'Maximum sequence length. Sequences will be right padded (and possibly truncated).', }, )
    flash_attn: Optional[bool] = field(default=False)

def main() -> None:
    """Main training routine."""
    parser = transformers.HfArgumentParser([TrainingArguments, ModelArguments, DataArguments])
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    if training_args.flash_attn:
        from flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True
    )

    tokenizer.add_eos_token = True
    train_dataset = get_dataset(training_args, data_args, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        trust_remote_code=True,
        # low_cpu_mem_usage=True
    ).cuda()

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params")

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator
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


if __name__ == '__main__':
    main()
