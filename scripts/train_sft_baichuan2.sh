export PATH=$HOME/.local/bin/:$PATH
export PYTHONPATH=$PYTHONPATH:`pwd`

deepspeed training/train_sft.py \
    --max_len 4096 \
    --dataset /sft_data_path/ \
    --train_batch_size 128 \
    --micro_train_batch_size 1 \
    --pretrain /ckpt/Baichuan2-13B-Base/ \
    --save_path /ckpt/rlhf_baichuan2_path/{date} \
    --use_func \
    --zero_stage 2 \
    --max_epochs 3 \
    --bf16 \
    --learning_rate 5e-6 \
    --gradient_checkpointing \
    --save_hf_model \
    --use_wandb 20f72f2655fa46883391bc52a5b6f13bec145818


