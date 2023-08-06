import torch
import lightning as L
import numpy as np
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from sophia import SophiaG
from functools import partial
torch.set_float32_matmul_precision('high')
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from model import Transformer, ModelArgs, TransformerBlock  # Assuming model.py is in the same directory
import os
import time
import math
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from preprocess_data import load_data
import argparse
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
dim = 2048
n_heads = 4
n_layers = 4
vocab_size = 32000
log_interval = 20
training_sample = 10000
num_epochs = 4

# Hyperparameters
learning_rate = 8e-4
batch_size = 32
weight_decay = 1e-1
beta1 = 0.965
beta2 = 0.99
grad_clip = 1.0
rho = 0.1

# Load the dataset
train_data, valid_data, test_data, train_len, validation_len = load_data(batch_size, training_sample)

def main(args):
    path = f"{args.optimizer}_out_5/"
    path_loss = f"{args.optimizer}_out_5/loss/"
    path_training = f"{args.optimizer}_out_5/training/"
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path_loss):
        os.mkdir(path_loss)
    if not os.path.exists(path_training):
        os.mkdir(path_training)
    m = Model(path_training)
    local_rank, world_size = m.setup_model_parallel()

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, activation_checkpointing=TransformerBlock)

    fabric = L.Fabric(accelerator="cuda", devices=args.num_nodes, precision="bf16-mixed", strategy=strategy)
    #fabric = L.Fabric(accelerator="cuda", devices=2, strategy=strategy)
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)


    model_args = fabric.to_device(ModelArgs(dim=dim, n_layers=n_layers, n_heads=n_heads, vocab_size=vocab_size, max_batch_size=batch_size))  # Update these parameters as needed
    model = fabric.to_device(Transformer(model_args))
    #print(model.parameters())
    for name, param in model.named_parameters():
        param.requires_grad = True

    # initialize a model/ start from checkpoint
    load_file_path = args.load_iter_path
    if load_file_path == "" or load_file_path=="pretrain":
        print("initial weight!!!")
        #model.apply(model._init_weights)
        model = fabric.setup_module(model)
        if args.optimizer == "sophia":
            optimizer = SophiaG(model.parameters(), lr=learning_rate, betas=(beta1, beta2), rho = 0.01, weight_decay=weight_decay)
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

        optimizer = fabric.setup_optimizers(optimizer)

        # Create the learning rate scheduler.

    elif not os.path.isfile(load_file_path):
        raise Exception(f'{load_file_path} is not a valid file path')
    else:
        model, optimizer, iter = m.load_model_checkpoint(fabric, model, load_file_path, args.optimizer)
        m.file_cnt = iter

    # save initial weights
    m.save_model_checkpoint(fabric, model, optimizer)

    total_steps = train_len * (num_epochs - m.iter_num + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=3e-4)

    filename = f"{path_loss}"+str(m.file_cnt)+".txt"
    stat_f = open(filename, "w", buffering=1)

    # save parameters to file
    stat_f.write('Parameters:\n')
    stat_f.write(f'learning_rate = {learning_rate}\n')
    stat_f.write(f'dim = {dim}\n')
    stat_f.write(f'n_heads = {n_heads}\n')
    stat_f.write(f'n_layers = {n_layers}\n')
    stat_f.write(f'vocab_size = {vocab_size}\n')
    #stat_f.write(f'epochs = {epochs}\n')
    stat_f.write(f'log_interval = {log_interval}\n')
    stat_f.write(f'batch_size = {batch_size}\n')
    stat_f.write(f'weight_decay = {weight_decay}\n')
    stat_f.write(f'beta1 = {beta1}\n')
    stat_f.write(f'beta2 = {beta2}\n')
    stat_f.write(f'grad_clip = {grad_clip}\n\n')

    print("Start training!")
    if args.optimizer == "sophia":
        m.train(fabric, model, optimizer, scheduler, train_data, valid_data, stat_f, True)
    elif args.optimizer == "adamw":
        m.train(fabric, model, optimizer, scheduler, train_data, valid_data, stat_f, False)

    stat_f.close()
    m.file_cnt += 1
    print("Training Complete")


class Model():
    def __init__(self, path_training):
        self.file_cnt = 1
        self.iter_num = 0
        self.path_training = path_training

    def setup_model_parallel(self):
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", -1))

        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(world_size)
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(1)
        return local_rank, world_size

    def save_model_checkpoint(self, fabric, model, optimizer):
        file_path = os.path.join(self.path_training, f"iter-{self.iter_num:06d}-ckpt.pth")
        
        if isinstance(fabric.strategy, FSDPStrategy):
            save_policy = FullStateDictConfig(offload_to_cpu=(fabric.world_size > 1), rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = model._forward_module.state_dict()
        else:
            state_dict = model.state_dict()
        
        if fabric.global_rank == 0:
            torch.save(state_dict, file_path)
        fabric.barrier()

    def load_model_checkpoint(self, fabric, model, file_path, optim="adamw"):
        index = file_path.find('iter-')
        self.iter_num = int(file_path[index+5:index+11])+1
        checkpoint = fabric.load(file_path)
        model.load_state_dict(checkpoint["model"])
        model = fabric.setup_module(model)
        if optim == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
        elif optim == "sophia":
            optimizer = SophiaG(model.parameters(), lr=learning_rate, betas=(beta1, beta2), rho = 0.01, weight_decay=weight_decay)
        else:
            raise Exception(f'{optim} is not a valid optimizer name, please try adamw or sophia.')

        optimizer = fabric.setup_optimizers(optimizer)

        return model, optimizer, self.iter_num
        

    @torch.no_grad()
    def validate(self, fabric, model, validation_dataset):
        model.eval()
        losses = 0
        with torch.no_grad():
            for i, batch in enumerate(validation_dataset):
                input_ids, labels = fabric.to_device(batch)
                logits = model(input_ids, 0)

                # Calculate the loss
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)

                losses += loss.item()
        out = losses / validation_len
        model.train()
        return out

    def train(self,
        fabric,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler, 
        train_data,
        val_data,
        stat_f,
        is_sophia=False):
        k = 10

        model.train()
        # run epoches
        for epoch in range(self.iter_num, num_epochs):            
          
            t0 = time.time()
            total_iter_loss = 0
            # run batches
            for i, batch in enumerate(train_data):
                input_ids, labels = fabric.to_device(batch)

                logits = model(input_ids, 0)

                # Calculate the loss
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)

                fabric.backward(loss)

                # Gradient clipping
                if grad_clip != 0.0:
                    fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_iter_loss += loss.item()

                # print and save log every log_interval iter
                if i % log_interval == 0:
                    elapsed = time.time() - t0
                    log = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.8f} | {:5.4f} s/batch  | loss {:5.4f} | ppl {:8.2f}\n'.format(\
                    self.iter_num, i+1, train_len , optimizer.param_groups[0]["lr"], elapsed / (i+1), total_iter_loss/(i+1), math.exp(total_iter_loss/(i+1)))
                    print(log)
                    print(log, file=stat_f)

                if is_sophia and i % k == k - 1:
                    logits = model(input_ids, 0)
                    samp_dist = torch.distributions.Categorical(logits=logits)
                    y_sample = samp_dist.sample()
                    loss_sampled = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=0)
                    loss_sampled.backward()
                    optimizer.update_hessian()
                    optimizer.zero_grad(set_to_none=True)
            
            epoch_loss = total_iter_loss / train_len

            dt = time.time() - t0
            
            self.iter_num += 1

            # evaluate the loss on train/val sets at end of epoch and write checkpoints
            fabric.print(f"Saving checkpoint to {self.path_training}")
            self.save_model_checkpoint(fabric, model, optimizer)
            val_loss = self.validate(fabric, model, val_data)
            log = '| end of epoch {:3d} | time elapsed {:3f}s | \
            valid loss {:5.2f} | valid ppl {:8.2f}\n | epoch loss {:5.4f} | epoch loss ppl {:5.2f}'.format(self.iter_num, dt, val_loss, \
            np.exp(val_loss), epoch_loss, math.exp(epoch_loss))
            print(log, file=stat_f)
            fabric.print(log)

            
            t0 = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--load_iter_path', type=str, default="")
    parser.add_argument('-o','--optimizer', type=str, default="adamw")
    parser.add_argument('-n','--num_nodes', type=int, default=2)


    args = parser.parse_args()
    main(args)