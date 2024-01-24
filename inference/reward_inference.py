import argparse
import json
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from openrlhf.datasets.baichuan_dataset import SupervisedDataset
from openrlhf.models import RewardModel
from openrlhf.utils import get_tokenizer


def batch_rm_inference(args):
    accelerator = Accelerator()

    accelerator.print("Loading tokenizer ...")
    tokenizer = get_tokenizer(args.pretrain, None, "left")

    accelerator.print("Loading dataset ...")
    dataset = SupervisedDataset(args.dataset, tokenizer, args.max_len)
    dataloader = DataLoader(dataset,
                            batch_size=args.micro_batch_size,
                            drop_last=True,
                            collate_fn=dataset.collate_fn,
                            )

    accelerator.print(f"Loading model ...")
    model = RewardModel(args.pretrain, from_config=False, use_flash_attention_2=args.flash_attn, )
    model.load_state_dict(torch.load(args.load_model), strict=True)
    model.eval().half().cuda()

    model, dataloader = accelerator.prepare(model, dataloader)

    accelerator.print("Start running ...")
    output_dataset = []
    for input_ids, _, attention_masks, example in tqdm(dataloader):
        with torch.no_grad():
            input_ids = input_ids.squeeze(1).to('cuda')
            attention_masks = attention_masks.squeeze(1).to('cuda')
            rewards = model(input_ids, attention_masks)

        for conv, reward in zip(example["conversations"], rewards):
            output_dataset.append({"conversations": conv, "reward": reward.item()})

    accelerator.print("Saving results ...")
    with open(args.output_path, "w", encoding="utf-8") as dump_f:
        json.dump(output_dataset, dump_f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--prompt_max_len", type=int, default=3000)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--flash_attn", action="store_true", default=False)

    # batch inference
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    args = parser.parse_args()
    batch_rm_inference(args)

