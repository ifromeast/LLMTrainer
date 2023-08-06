
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
# from Sophia.decoupled_sophia.decoupled_sophia import DecoupledSophia, HutchinsonEstimator
from sophia import SophiaG, DecoupledSophia
from transformers import AutoTokenizer
from utils.data_utils import get_dataset, fault_tolerance_data_collator


tokenizer = AutoTokenizer.from_pretrained('/root/alpaca_test/LLMTrainer/ckpt/Llama-2-13b-hf')
train_ds = get_dataset(tokenizer, model_max_length=2048, cache_dir='/root/alpaca_test/cache_dir')['train']

# Initialize the GPT-2 model and tokenizer
config = AutoConfig.from_pretrained('/root/alpaca_test/LLMTrainer/config/config.json')
model = AutoModelForCausalLM.from_config(config)


# Initialize the DecoupledSophia optimizer
optimizer = SophiaG(model.parameters(), lr=2e-4, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
# DecoupledSophia(model.parameters(), lr=1e-3, betas=(0.9, 0.999), rho=0.04, weight_decay=1e-1, k=10, estimator="Hutchinson")

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=1,
    fp16=True,
    per_device_train_batch_size=2,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1,
    logging_strategy="steps",
    logging_steps=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=2,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=fault_tolerance_data_collator,
    train_dataset=train_ds,
    optimizers=(optimizer, None),
)

# Train the model
trainer.train()

# Evaluate the model
# eval_results = trainer.evaluate()
# print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss']))}")