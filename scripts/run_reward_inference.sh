export PATH=$HOME/.local/bin/:$PATH
export PYTHONPATH=$PYTHONPATH:`pwd`

accelerate launch --num_processes=8 inference/reward_inference.py \
    --pretrain /ckpt/Baichuan2-13B-Chat/ \
    --load_model /ckpt/rlhf_baichuan2/1124/rm_model.pt \
    --dataset /sft_data_path/ \
    --output_path /output.json \
    --micro_batch_size 4