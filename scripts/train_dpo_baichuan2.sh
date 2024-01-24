export PATH=$HOME/.local/bin/:$PATH
export PYTHONPATH=$PYTHONPATH:`pwd`

deepspeed training/train_dpo.py \
     --pretrain /ckpt/Baichuan2-13B-Base/ \
     --save_path /ckpt/rlhf_baichuan2/1024/ \
     --dataset /comparison_data_path/ \
     --load_model //ckpt/rlhf_baichuan2_path/sft_model.pt \
     --train_batch_size 128 \
     --micro_train_batch_size 1 \
     --bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --beta 0.1 \
     --learning_rate 5e-7 \
     --gradient_checkpointing \
     --save_hf_model \
     --use_wandb 20f72f2655fa46883391bc52a5b6f13bec145818
     

