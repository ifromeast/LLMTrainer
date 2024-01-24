set -x
export PATH=$HOME/.local/bin/:$PATH

ray job submit --address="http://10.59.144.213:8037" \
    --runtime-env-json='{"working_dir": "."}' \
    -- python3 training/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 1 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 1 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --pretrain /ckpt/Baichuan2-13B-Base/ \
    --critic_pretrain /ckpt/Baichuan2-13B-Chat/ \
    --reward_model_path /ckpt/rlhf_baichuan2_path/{date}/rm_model.pt \
    --sft_model_path /ckpt/rlhf_baichuan2_path/{date}/sft_model.pt \
    --save_path /ckpt/rlhf_baichuan2_path/{date} \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --num_episodes 1 \
    --prompt_max_len 2048 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --inference_tp_size 1 \
    --init_kl_coef 0.01 \
    --prompt_data /prompt_data_path/ \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing \
    --save_hf_model \
    --use_wandb 20f72f2655fa46883391bc52a5b6f13bec145818