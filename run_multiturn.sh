PYTHONPATH=$PYTHONPATH:/data/path/triptrainer
output=/data/path
mkdir -p ${output}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 torchrun --nproc_per_node=8 --master_port=50125 \
    train_multiturn_no_cache.py \
    --model_name_or_path /model/path\
    --dataset_dir /data/path \
    --do_train \
    --bf16 \
    --output_dir ${output} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --model_max_length 4096 \
    --save_strategy "steps" \
    --save_steps 40 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.03 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to 'tensorboard' \
    --deepspeed '/config/ds_zero3_no_offload.json' \
    --logging_dir ${output}/logs \
    --gradient_checkpointing True > ${output}/logs.txt