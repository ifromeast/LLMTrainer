import argparse
import os
import json
import resource
from contextlib import nullcontext
from functools import partial
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from attn import SUPPORT_XFORMERS, replace_xformers
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.data_utils import get_dataset, fault_tolerance_data_collator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import LlamaTokenizer

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device


def load_json(file_path: str):
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def get_model_numel(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f'{numel / B:.2f} B'
    elif numel >= M:
        return f'{numel / M:.2f} M'
    elif numel >= K:
        return f'{numel / K:.2f} K'
    else:
        return f'{numel}'


def tokenize_batch(batch, tokenizer: Optional[LlamaTokenizer] = None, max_length: int = 2048):
    texts = [sample['text'] for sample in batch]
    data = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
    data['labels'] = data['input_ids'].clone()
    return data


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def save(booster: Booster, model: nn.Module, optimizer: Optimizer, lr_scheduler: _LRScheduler, epoch: int, step: int,
         batch_size: int, coordinator: DistCoordinator, save_dir: str):
    save_dir = os.path.join(save_dir, f'epoch{epoch}-step{step}')
    os.makedirs(os.path.join(save_dir, 'model'), exist_ok=True)

    booster.save_model(model, os.path.join(save_dir, 'model'), shard=True)
    # TODO: sharded optimizer is not supported yet
    booster.save_optimizer(optimizer, os.path.join(save_dir, 'optimizer'), shard=False)
    booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, 'lr_scheduler'))
    running_states = {
        'epoch': epoch,
        'step': step,
        'sample_start_index': step * batch_size,
    }
    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, 'running_states.json'))


def load(booster: Booster, model: nn.Module, optimizer: Optimizer, lr_scheduler: _LRScheduler,
         load_dir: str) -> Tuple[int, int, int]:
    booster.load_model(model, os.path.join(load_dir, 'model'))
    booster.load_optimizer(optimizer, os.path.join(load_dir, 'optimizer'))
    booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, 'lr_scheduler'))
    running_states = load_json(os.path.join(load_dir, 'running_states.json'))
    return running_states['epoch'], running_states['step'], running_states['sample_start_index']


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/root/alpaca_test/LLMTrainer/config/config.json', help='Model configuration')
    parser.add_argument('-p',
                        '--plugin',
                        choices=['gemini', 'gemini_cuda', 'gemini_cpu', 'zero2', 'zero2_cpu'],
                        default='gemini',
                        help='Choose which plugin to use')
    parser.add_argument('-d',
                        '--dataset',
                        type=str,
                        default='togethercomputer/RedPajama-Data-1T-Sample',
                        help='Data set path')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Local batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('-w', '--weigth_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('-s', '--warmup_steps', type=int, default=2, help='Warmup steps')
    parser.add_argument('-g', '--grad_checkpoint', action='store_true', help='Use gradient checkpointing')
    parser.add_argument('-l', '--max_length', type=int, default=2048, help='Max sequence length')
    parser.add_argument('-x', '--mixed_precision', default='fp16', choices=['fp16', 'bf16'], help='Mixed precision')
    parser.add_argument('-i', '--save_interval', type=int, default=1000, help='Save interval')
    parser.add_argument('-o', '--save_dir', type=str, default='checkpoint', help='Checkpoint directory')
    parser.add_argument('-f', '--load', type=str, default=None, help='Load checkpoint')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('-t', '--tensorboard_dir', type=str, default='tb_logs', help='Tensorboard directory')
    parser.add_argument('-a', '--flash_attention', action='store_true', help='Use Flash Attention')
    args = parser.parse_args()

    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()

    # ==============================
    # Initialize Tensorboard
    # ==============================
    if coordinator.is_master():
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == 'gemini':
        plugin = GeminiPlugin(precision=args.mixed_precision,
                              placement_policy='auto',
                              initial_scale=2**16,
                              max_norm=args.grad_clip)
    elif args.plugin == 'gemini_cuda':
        plugin = GeminiPlugin(precision=args.mixed_precision,
                              placement_policy='cuda',
                              initial_scale=2**16,
                              max_norm=args.grad_clip)
    elif args.plugin == 'gemini_cpu':
        plugin = GeminiPlugin(precision=args.mixed_precision,
                              placement_policy='cpu',
                              initial_scale=2**16,
                              max_norm=args.grad_clip)
    elif args.plugin == 'zero2':
        plugin = LowLevelZeroPlugin(stage=2,
                                    precision=args.mixed_precision,
                                    initial_scale=2**16,
                                    max_norm=args.grad_clip)
    elif args.plugin == 'zero2_cpu':
        plugin = LowLevelZeroPlugin(stage=2,
                                    precision=args.mixed_precision,
                                    initial_scale=2**16,
                                    cpu_offload=True,
                                    max_norm=args.grad_clip)
    else:
        raise ValueError(f'Unknown plugin {args.plugin}')

    booster = Booster(plugin=plugin)

    # ==============================
    # Initialize Tokenizer, Dataset and Dataloader
    # ==============================
    tokenizer = AutoTokenizer.from_pretrained('/root/alpaca_test/LLMTrainer/ckpt/Llama-2-13b-hf')
    tokenizer.pad_token = tokenizer.unk_token

    train_ds = get_dataset(tokenizer, args.max_length, '/root/alpaca_test/cache_dir')['train']
    sampler = DistributedSampler(train_ds)
    dataloader = DataLoader(train_ds,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            collate_fn=fault_tolerance_data_collator,
                            drop_last=True,
                            num_workers=8,)

    # ==============================
    # Initialize Model, Optimizer and LR Scheduler
    # ==============================
    config = AutoConfig.from_pretrained(args.config)
    init_ctx = LazyInitContext(
        default_device=get_current_device()) if isinstance(plugin, GeminiPlugin) else nullcontext()

    # with init_ctx:
    model = AutoModelForCausalLM.from_config(config)

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    if args.flash_attention:
        assert SUPPORT_XFORMERS, 'Use flash attention while xfomers is not installed'
        replace_xformers(model)

    model_numel = get_model_numel(model)
    coordinator.print_on_master(f'Model params: {format_numel_str(model_numel)}')

    optimizer = HybridAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weigth_decay)
    lr_scheduler = CosineAnnealingWarmupLR(optimizer,
                                           total_steps=args.num_epochs * len(dataloader),
                                           warmup_steps=args.warmup_steps,
                                           eta_min=0.1 * args.lr)
    default_dtype = torch.float16 if args.mixed_precision == 'fp16' else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(model,
                                                                  optimizer,
                                                                  dataloader=dataloader,
                                                                  lr_scheduler=lr_scheduler)
    torch.set_default_dtype(torch.float)

    coordinator.print_on_master(f'Booster init max CUDA memory: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB')
    coordinator.print_on_master(f'Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB')

    # load checkpoint if specified
    start_epoch = 0
    start_step = 0
    if args.load is not None:
        coordinator.print_on_master('Loading checkpoint')
        start_epoch, start_step, sampler_start_idx = load(booster, model, optimizer, lr_scheduler, args.load)
        coordinator.print_on_master(f'Loaded checkpoint {args.load} at epoch {start_epoch} step {start_step}')

    num_steps_per_epoch = len(dataloader)
    for epoch in range(start_epoch, args.num_epochs):
        with tqdm(enumerate(dataloader), desc=f'Epoch {epoch}', disable=not coordinator.is_master(), total=num_steps_per_epoch, initial=start_step) as pbar:
            for step, batch in pbar:
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs[0]
                booster.backward(loss, optimizer)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                all_reduce_mean(loss)
                pbar.set_postfix({'loss': loss.item()})
                if coordinator.is_master():
                    writer.add_scalar('loss', loss.item(), epoch * num_steps_per_epoch + step)

                if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
                    coordinator.print_on_master(f'Saving checkpoint')
                    save(booster, model, optimizer, lr_scheduler, epoch, step + 1, args.batch_size, coordinator,
                         args.save_dir)
                    coordinator.print_on_master(f'Saved checkpoint at epoch {epoch} step {step + 1}')
        start_step = 0

    coordinator.print_on_master(f'Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB')


if __name__ == '__main__':
    main()