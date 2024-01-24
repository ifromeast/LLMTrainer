export PATH=$HOME/.local/bin/:$PATH
export PYTHONPATH=$PYTHONPATH:`pwd`

deepspeed training/train_rm.py \
     --pretrain /ckpt/Baichuan2-13B-Chat/ \
     --save_path /ckpt/rlhf_baichuan2_path/{date} \
     --dataset /comparison_data_path/ \
     --train_batch_size 128 \
     --micro_train_batch_size 1 \
     --bf16 \
     --max_epochs 4 \
     --max_len 4096 \
     --zero_stage 3 \
     --learning_rate 9e-6 \
     --gradient_checkpointing \
     --use_wandb 20f72f2655fa46883391bc52a5b6f13bec145818
