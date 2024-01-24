export PATH=$HOME/.local/bin/:$PATH
export PYTHONPATH=$PYTHONPATH:`pwd`
export CUDA_VISIBLE_DEVICES=0,1

streamlit run inference/web_demo3.py --server.port 8022 -- \
     --actor_path /ckpt/rlhf_baichuan2_path/{date}
