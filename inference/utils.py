import torch
from typing import List
from queue import Queue
from threading import Thread
from typing import List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


class TextIterStreamer:
    def __init__(self, tokenizer, skip_prompt=False, skip_special_tokens=False):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.tokens = []
        self.text_queue = Queue()
        self.next_tokens_are_prompt = True

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
        else:
            if len(value.shape) > 1:
                value = value[0]
            self.tokens.extend(value.tolist())
            self.text_queue.put(
                self.tokenizer.decode(self.tokens, skip_special_tokens=self.skip_special_tokens))

    def end(self):
        self.text_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get()
        if value is None:
            raise StopIteration()
        else:
            return value
    
    def length(self):
        return len(self.tokens)


class DataTool:
    """Dataset for supervised fine-tuning."""
    def __init__(self, tokenizer, model_max_length=2048, user_tokens=[195], assistant_tokens=[196],):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

        self.system_tokens = [194]
        self.user_tokens = [195]
        self.assistant_tokens = [196]
        self.tool_tokens = [196]
        self.observation_tokens = [197]
        self.ignore_index = -100

    def preprocessing(self, messages):
        input_ids = []
        labels = []

        for message in messages:
            role = message["role"]
            value = message["content"]
            value_ids = self.tokenizer.encode(value)

            if role == "system":
                input_ids += self.system_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            elif role == "user":
                input_ids += self.user_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            elif role == "assistant" or role == "tool":
                input_ids += self.assistant_tokens + value_ids
                labels += [self.ignore_index] + value_ids
            elif role == "observation":
                input_ids += self.observation_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(value_ids)
            else:
                raise TypeError(f"No role named {role}, please check it!")

        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        input_ids = input_ids[: self.model_max_length]
        labels = labels[: self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.model_max_length - len(input_ids))
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return input_ids, labels, attention_mask, messages

    def build_chat_input(self, model, tokenizer, messages: List[dict], max_new_tokens: int=0):
        def _parse_messages(messages, split_role=["user","observation"]):
            system, rounds = "", []
            round = []
            for i, message in enumerate(messages):
                if message["role"] == "system":
                    assert i == 0
                    system = message["content"]
                    continue
                if message["role"] in split_role and round:
                    rounds.append(round)
                    round = []
                round.append(message)
            if round:
                rounds.append(round)
            return system, rounds

        max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
        max_input_tokens = model.config.model_max_length - max_new_tokens
        system, rounds = _parse_messages(messages, split_role=["user","observation"])
        system_tokens = self.system_tokens + tokenizer.encode(system) ####
        max_history_tokens = max_input_tokens - len(system_tokens)

        history_tokens = []
        for round in rounds[::-1]:
            round_tokens = []
            for message in round:
                if message["role"] == "system":
                    round_tokens += self.system_tokens
                elif message["role"] == "user":
                    round_tokens += self.user_tokens
                elif message["role"] == "assistant" or message["role"] == "tool":
                    round_tokens += self.assistant_tokens
                elif message["role"] == "observation":
                    round_tokens += self.observation_tokens
                else:
                    raise TypeError(f"No role, please check it!")

                round_tokens.extend(tokenizer.encode(message["content"]))
            if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
                history_tokens = round_tokens + history_tokens  # concat left
                if len(history_tokens) < max_history_tokens:
                    continue
            break

        input_tokens = system_tokens + history_tokens
        if messages[-1]["role"] != "assistant":
            input_tokens.append(model.generation_config.assistant_token_id)
        input_tokens = input_tokens[-max_input_tokens:]  # truncate left
        return torch.LongTensor([input_tokens]).to(model.device)



    def chat(self, model, input_ids, stream=False, device=None):
        generation_config = model.generation_config
        # input_ids = self.build_chat_input(model, self.tokenizer, messages, generation_config.max_new_tokens)
        # input_ids = input_ids.unsqueeze(0).to(device)
        if stream:
            streamer = TextIterStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            Thread(target=model.generate, kwargs=dict(
                inputs=input_ids, streamer=streamer,
                generation_config=generation_config,
            )).start()
            return streamer
        else:
            outputs = model.generate(input_ids, generation_config=generation_config)
            response = self.tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
            return response


class InferModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        print("init infer model ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map='auto'
        ).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self.datatool = DataTool(self.tokenizer, self.model.generation_config.max_new_tokens)

    def chat(self, messages, stream=False):
        input_ids = self.datatool.build_chat_input(self.model, 
                                                   self.tokenizer, 
                                                   messages, 
                                                   self.model.generation_config.max_new_tokens)
        return self.datatool.chat(self.model, input_ids, stream, self.model.device)
