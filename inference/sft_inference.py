import argparse
import json
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from openrlhf.datasets.baichuan_dataset import PromptsDataset
from openrlhf.models import Actor
from openrlhf.utils import get_tokenizer


def batch_rm_inference(args):
    accelerator = Accelerator()

    accelerator.print("Loading tokenizer ...")
    tokenizer = get_tokenizer(args.actor_model, None, "left")

    # tokenizer
    def tokenize_fn(texts):
        batch = tokenizer(
            texts,
            return_tensors="pt",
            max_length=args.prompt_max_len,
            padding=True,
            truncation=True,
        )
        return {k: v.to(torch.cuda.current_device()) for k, v in batch.items()}

    accelerator.print("Loading dataset ...")
    dataset = PromptsDataset(args.prompt_dataset, tokenizer, args.max_len)
    dataloader = DataLoader(dataset,
                            batch_size=args.micro_batch_size,
                            drop_last=True,
                            collate_fn=dataset.collate_fn,
                            )

    accelerator.print(f"Loading model ...")
    model = Actor(args.actor_model, from_config=False, use_flash_attention_2=args.flash_attn, )
    model.eval().half().cuda()

    model, dataloader = accelerator.prepare(model, dataloader)

    accelerator.print("Start running ...")
    N = args.best_of_n
    output_dataset = []
    for prompts, example in tqdm(dataloader):
        inputs = tokenize_fn(prompts)
        for _ in range(N):
            outputs = model.model(**inputs, **model.model.generation_config)
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for prompt, output in zip(example["conversations"], outputs):
            output = output[len(prompt):]
            output_dataset.append({"history": prompt, "output": output})

    accelerator.print("Saving results ...")
    with open(args.output_path, "w", encoding="utf-8") as dump_f:
        json.dump(output_dataset, dump_f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_model", type=str, default=None)

    parser.add_argument("--prompt_max_len", type=int, default=3000)
    parser.add_argument("--max_len", type=int, default=4096)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--best_of_n", type=int, default=1)

    # batch inference
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--prompt_dataset", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    args = parser.parse_args()
    batch_rm_inference(args)

