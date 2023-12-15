import os
import torch
import argparse
import platform
import subprocess
from colorama import Fore, Style
from tempfile import NamedTemporaryFile
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='', help='base model to use')
    return parser.parse_args()


def init_model(args):
    print("init model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map='auto'
    )
    model.generation_config = GenerationConfig.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, trust_remote_code=True)
    model.eval().to(device)
    return model, tokenizer


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(Fore.YELLOW + Style.BRIGHT + "欢迎使用携程问道大模型，输入进行对话，vim 多行输入，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。")
    return []


def vim_input():
    with NamedTemporaryFile() as tempfile:
        tempfile.close()
        subprocess.call(['vim', '+star', tempfile.name])
        text = open(tempfile.name).read()
    return text


def main(stream=True):
    args = set_args()
    model, tokenizer = init_model(args)
    messages = clear_screen()
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\nUSER：" + Style.NORMAL)
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        if prompt.strip() == 'vim':
            prompt = vim_input()
            print(prompt)
        print(Fore.CYAN + Style.BRIGHT + "\nTripGPT：" + Style.NORMAL, end='')
        if prompt.strip() == "stream":
            stream = not stream
            print(Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"), end='')
            continue
        messages.append({"role": "user", "content": prompt})
        if stream:
            position = 0
            try:
                for response in model.chat(tokenizer, messages, stream=True):
                    print(response[position:], end='', flush=True)
                    position = len(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            except KeyboardInterrupt:
                pass
            print()
        else:
            response = model.chat(tokenizer, messages)
            print(response)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})
    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()
