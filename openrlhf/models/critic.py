from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM


class Critic(nn.Module):
    """
    Critic model base class.

    Args:
        model (nn.Module): Critic model.
        value_head (nn.Module): Value head to get value.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self, pretrain_or_model, from_config=False, normalize_reward=True, use_flash_attention_2=False
    ) -> None:
        super().__init__()
        if isinstance(pretrain_or_model, str):
            if from_config:
                config = AutoConfig.from_pretrained(
                    pretrain_or_model,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    use_flash_attention_2=use_flash_attention_2,
                )
                self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).model
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrain_or_model,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    use_flash_attention_2=use_flash_attention_2,
                ).model
        else:
            self.model = pretrain_or_model

        self.value_head = nn.Linear(self.model.config.hidden_size, 1)

        # mean std
        self.normalize_reward = normalize_reward
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(sequences, attention_mask=attention_mask)
        last_hidden_states = outputs["last_hidden_state"]

        values = self.value_head(last_hidden_states).squeeze(-1)[:, :-1]
        num_actions = action_mask.size(1)

        # normalize reward
        if self.normalize_reward:
            values = (values - self.mean) / self.std
        return values[:, -num_actions:]

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def lora_enable(self, lora_rank=0, lora_train_bias="none"):
        if lora_rank > 0:
            lora_config = LoraConfig(
                inference_mode=False,
                r=lora_rank,
                lora_alpha=16,
                lora_dropout=0.05,
                bias=lora_train_bias,
            )
            self.model = get_peft_model(self.model, lora_config)
