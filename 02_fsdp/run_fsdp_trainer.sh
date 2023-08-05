
torchrun --nnodes 1 --nproc_per_node 8 ../01_base_hf/pretrain_hf.py \
    --model_config_path /root/alpaca_test/LLMTrainer/config/config.json \
    --tokenizer_name_or_path /root/alpaca_test/LLMTrainer/ckpt/Llama-2-13b-hf \
    --per_device_train_batch_size 12 \
    --do_train \
    --seed 1234 \
    --fp16 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 1 \
    --save_steps 100 \
    --gradient_accumulation_steps 8 \
    --model_max_length 2048 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_config './fsdp.json' \
    --output_dir './fsdp_logs' \
    --overwrite_output_dir \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False