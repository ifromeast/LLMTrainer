#!/bin/bash
export OMP_NUM_THREADS=2
export PYTHONPATH=$PYTHONPATH:`pwd`

DATA_PATH="/data/share_user/zzd/data/rlhf_data/sft_data_v7/"
OUTPUT_PATH="/data/share_user/zzd/ckpt/sft_qwen/0407-f3-lora-hf"
MODEL_PATH="/data/share_user/zzd/ckpt/qwen/Qwen1.5-72B-Chat"

deepspeed training/finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --use_func True \
    --bf16 True \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 4096 \
    --use_lora True \
    --gradient_checkpointing \
    --deepspeed training/config/ds_zero3_no_offload.json
