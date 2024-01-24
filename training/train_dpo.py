import argparse
import math
import os
from copy import deepcopy
from datetime import datetime

from transformers.trainer import get_scheduler

from openrlhf.datasets.baichuan_dataset import RewardDataset
from openrlhf.models import Actor
from openrlhf.trainer import DPOTrainer
from openrlhf.utils import get_strategy, get_tokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model, load huggingface model/config
    from_config = bool(args.load_model or args.load_checkpoint)
    model = Actor(args.pretrain, from_config, use_flash_attention_2=args.flash_attn)

    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy)
    strategy.print(model)

    # load SFT model
    if args.load_model and not args.load_checkpoint:
        strategy.load_model(model, args.load_model, strict=True)
        strategy.print("Load model: ", args.load_model)
    ref_model = deepcopy(model)

    # lora
    if args.lora_rank > 0:
        model.lora_enable(args.lora_rank)

    # configure optimizer
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    # prepare for data and dataset
    train_dataset = RewardDataset(args.dataset, tokenizer, args.max_len)
    train_dataloader = strategy.setup_dataloader(train_dataset, args.micro_train_batch_size, True, True, train_dataset.collate_fn,)

    # scheduler
    num_update_steps_per_epoch = len(train_dataloader) * args.max_epochs // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        "cosine",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # strategy prepare
    ((model, optim, scheduler), ref_model) = strategy.prepare((model, optim, scheduler), ref_model)

    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)
        # strategy.load_checkpoint(args.save_path + '/rm_model.pt')

    os.makedirs(args.save_path, exist_ok=True)

    # batch_size here is expected to be C(k,2), k means # response of each prompt
    # be limited with the format of dataset 'Dahoas/rm-static', we'd better use batch_size as 1
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=None,
        scheduler=scheduler,
        max_norm=args.max_norm,
        beta=args.beta,
        max_epochs=args.max_epochs,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer.fit(use_lora=args.lora_rank)

    # save model checkpoint after fitting on only rank0
    strategy.save_model(model, args.save_path + "/dpo_model.pt", only_rank0=True)

    if args.save_hf_model:
        os.makedirs(args.save_path + "/dpo_hf", exist_ok=True)
        strategy.save_hf_format(model, tokenizer, args.save_path + "/dpo_hf",)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")
    parser.add_argument('--dataset', type=str, default='Anthropic/hh-rlhf')
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--micro_train_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_rank", type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--zpg", type=int, default=8, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--save_hf_model", action="store_true", default=False)

    # wandb pamameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_dpo")
    parser.add_argument("--wandb_run_name", type=str, default="dpo_%s" % datetime.now().strftime("%m%dT%H:%M"),)

    args = parser.parse_args()
    train(args)
