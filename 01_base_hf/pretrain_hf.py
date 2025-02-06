
import argparse
import logging
import math 
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import transformers
from transformers import Trainer
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from utils.data_utils import get_dataset, fault_tolerance_data_collator

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for models."""
    model_config_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model name or config path")}, )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The tokenizer name or path")}, )

@dataclass
class DataArguments:
    """Arguments for datasets."""
    datasets: List[str] = field(
        default=None,
        metadata={'help': 'Path to the local training data.'}, )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments for the training loop."""
    cache_dir: Optional[str] = field(default='/data/zzd/cache_dir')
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={'help': 'Maximum sequence length. Sequences will be right padded (and possibly truncated).',},)
    flash_attn : Optional[bool] = field(default=False)



def main() -> None:
    """Main training routine."""
    parser = transformers.HfArgumentParser([TrainingArguments, ModelArguments, DataArguments])
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    if training_args.flash_attn:
        from utils.flash_attn_patch import replace_llama_attn_with_flash_attn
        replace_llama_attn_with_flash_attn()


    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = get_dataset(tokenizer, training_args.model_max_length, training_args.cache_dir)

    config = AutoConfig.from_pretrained(model_args.model_config_path)
    model = AutoModelForCausalLM.from_config(config)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset['train'] if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
    )

     # Training
    if training_args.do_train:
        train_result = trainer.train()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model()



if __name__ == '__main__':
    main()

