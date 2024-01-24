
RLHF solution for Baichuan 2

```
# run sft
sh scripts/train_sft_baichuan2.sh

# run rm
sh scripts/train_rm_baichuan2.sh


# run ppo by ray
# launch ray
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8 --dashboard-host 10.59.144.213 --dashboard-port 8090

# train ray PPO model (8 gpus)
sh scripts/train_ppo_baichuan2.sh

```

