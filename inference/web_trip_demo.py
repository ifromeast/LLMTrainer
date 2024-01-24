import os
import re
import copy
import time
import json
import torch
import argparse
from enum import Enum
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from openrlhf.models import RewardModel
from utils import DataTool
from datetime import date
from function_calling.tool_registry import get_tools, dispatch_tool, get_rag
from openrlhf.datasets.utils import get_system_prompt

class Engine(str, Enum):
    google, rag = '🎨 Google', '🎯 RAG'

tools = list(get_tools().keys())
system_prompt = get_system_prompt()

def extract_code(text: str) -> str:
    pattern = r'```([^\n]*)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1][1]

def tool_call(*args, **kwargs) -> dict:
    print("=== Tool call:")
    print(args)
    print(kwargs)
    return kwargs


today = str(date.today())
st.set_page_config(page_title="WenDao")
st.title("携程问道大模型")
tab = st.radio('Engine', [e.value for e in Engine], horizontal=True, label_visibility='hidden',)

actor_device = torch.device("cuda:0")

def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_path', type=str, default='', help='actor model to use')
    return parser.parse_args()
args = set_args()

@st.cache_resource
def init_model():
    print("init actor model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.actor_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map='auto'
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(args.actor_path)
    tokenizer = AutoTokenizer.from_pretrained(args.actor_path, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = 'left'

    return model, tokenizer

model, tokenizer = init_model()
datatool = DataTool(tokenizer, model.generation_config.max_new_tokens)


def goodcase_status():
    st.session_state['key'] = "goodcase"
    st.session_state['n_good'] += 1

def badcase_status():
    st.session_state['key'] = "badcase"
    st.session_state['n_bad'] += 1

def clear_chat_history():
    del st.session_state.messages
    st.session_state['length'] = 0
    torch.cuda.empty_cache()

def proc():
    st.session_state.text = st.session_state.text_key

good_list, bad_list, pair_list = [],[],[]
dir_path = f"/data/share_user/zzd/data/rlhf_data/log_data_v3/{today}"
if not os.path.exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def init_chat_history():
    with st.chat_message("assistant", avatar='🐬'):
        st.markdown("您好，我是携程问道大模型，很高兴为您服务🥰")

    print(st.session_state.key)
    if st.session_state.key == "goodcase":
        good_list.append(st.session_state.messages)
        st.session_state.key = "none"
        with open(dir_path+f"/goodcase_{st.session_state.n_good}.json", "w", encoding="utf-8") as dump_f:
            json.dump(good_list, dump_f, ensure_ascii=False)
        clear_chat_history()

    elif st.session_state.key == "badcase":
        bad_list.append(st.session_state.messages)
        st.session_state.key = "none"
        with open(dir_path+f"/badcase_{st.session_state.n_bad}.json", "w", encoding="utf-8") as dump_f:
            json.dump(bad_list, dump_f, ensure_ascii=False)
        clear_chat_history()

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            if message["role"] == "user":
                avatar = '🙋'  
            elif message["role"] == "assistant":
                avatar = '🐬'
            elif message["role"] == "tool":
                avatar = '🛠️'
            elif message["role"] == "observation":
                avatar = '🦉'
            elif message["role"] == "system":
                avatar = '🧭'
                continue

            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if st.session_state.messages[-1]['role'] == "assistant":
            output_text = copy.deepcopy(st.session_state.messages[-1]['content'])
            tool, *out_text = output_text.strip().split('\n')
            if tool in tools:
                st.session_state.messages[-1]['role'] = "tool"
                out_text = '\n'.join(out_text)
                code = extract_code(out_text)
                args = eval(code, {'tool_call': tool_call}, {})
                if tool == 'get_search' and tab == Engine.rag:
                    observation, rag_flag = get_rag(args['query'], tokenizer)
                else:
                    observation = dispatch_tool(tool, args)
                    
                st.session_state.messages.append({'role':'observation','content':observation})
                with st.chat_message("observation", avatar='🦉'):
                    placeholder = st.empty()
                    placeholder.markdown(observation)

                with st.chat_message("assistant", avatar='🐬'):
                    placeholder = st.empty()
                    generation_config = model.generation_config
                    with torch.no_grad():
                        input_ids = datatool.build_chat_input(model, tokenizer, st.session_state.messages, generation_config.max_new_tokens)
                        for response in datatool.chat(model, input_ids, stream=True, device=model.device):
                            placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.session_state.messages = [{"role": "system", "content": system_prompt}]


    return st.session_state.messages


def main():
    if 'key' not in st.session_state:
        st.session_state['key'] = 'value'
    if 'n_good' not in st.session_state:
        st.session_state['n_good'] = 0
    if 'n_bad' not in st.session_state:
        st.session_state['n_bad'] = 0
    if 'n_pair' not in st.session_state:
        st.session_state['n_pair'] = 0
    if 'pair_flag' not in st.session_state:
        st.session_state['pair_flag'] = 'state0'
    if 'mess' not in st.session_state:
        st.session_state['mess'] = {}
    if 'length' not in st.session_state:
        st.session_state['length'] = 0
    if 'perf' not in st.session_state:
        st.session_state['perf'] = 0.0
    if 'rag_flag' not in st.session_state:
        st.session_state['rag_flag'] = True

    messages = init_chat_history()

    def copy_and_move():
        idx = st.session_state.loc
        if st.session_state.messages:
            st.session_state.text = st.session_state.messages[-idx]["content"]

    def finish_and_update():
        idx = st.session_state.loc
        if st.session_state.messages:
            if idx == 1 and st.session_state.messages[-1]["role"] in ["assistant", "tool"]:
                mess = {}
                if st.session_state.pair_flag == 'state0':
                    st.session_state['n_pair'] += 1
                    history = st.session_state.messages[:-2]
                    query = st.session_state.messages[-2]['content']
                    chosen = st.session_state.text
                    rejected = st.session_state.messages[-1]["content"]
                    st.session_state.mess = {'history':history, 'query':query, 'chosen':chosen, 'rejected':rejected}
                    pair_list = [st.session_state.mess]
                    st.session_state['pair_flag'] = 'state1'
                    with open(dir_path+f"/pairwise_{st.session_state.n_pair}.json", "w", encoding="utf-8") as dump_f:
                        json.dump(pair_list, dump_f, ensure_ascii=False)
                    
                elif st.session_state.pair_flag == 'state1':
                    chosen = st.session_state.text
                    st.session_state.mess['chosen'] = chosen
                    pair_list = [st.session_state.mess]
                    with open(dir_path+f"/pairwise_{st.session_state.n_pair}.json", "w", encoding="utf-8") as dump_f:
                        json.dump(pair_list, dump_f, ensure_ascii=False)

                st.session_state.messages[-1]["content"] = st.session_state.text
            else:
                st.session_state.messages[-idx]["content"] = st.session_state.text

    with st.sidebar:
        if 'text' not in st.session_state:
            st.session_state['text'] = ''
        if 'loc' not in st.session_state:
            st.session_state['loc'] = 1
        col1, col2 = st.columns([1,1])
        with col1:
            st.button("转移🧲", on_click=copy_and_move)
        with col2:
            st.button("应用🛰️", on_click=finish_and_update)
        text_box = st.text_area(
            label="文本修改框",
            key='text_key',
            height=600,
            value=st.session_state.text,
            on_change=proc, 
        )
        loc = st.number_input('索引位置', min_value=1, max_value=20, step=1, label_visibility='hidden')
        st.session_state.loc = loc

        col3, col4 = st.columns([1,1])
        with col3:
            if st.button('删除指定位置'):
                del st.session_state.messages[st.session_state.loc]
        with col4:
            with open("/data/share_user/zzd/data/rlhf_data/tmp_data/search_data.txt", "r") as file:
                btn = st.download_button(label="下载检索数据", data=file, file_name="search_data.txt")

        my_bar = st.progress(st.session_state.length/model.config.model_max_length, 
                             text=f"当前token总数为: {st.session_state.length}/{model.config.model_max_length}")
        st.info(f'{round(st.session_state.perf,3)} tokens/s', icon="⚡")
        

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        st.session_state['pair_flag'] = 'state0'
        with st.chat_message("user", avatar='🙋'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)

        retry = 0
        while messages[-1]['role'] == 'user' or st.session_state.rag_flag:
            output_text = ''
            with st.chat_message("assistant", avatar='🐬'):
                placeholder = st.empty()
                generation_config = model.generation_config
                input_ids = datatool.build_chat_input(model, tokenizer, messages, generation_config.max_new_tokens)
                start_time = time.time()
                for response in datatool.chat(model, input_ids, stream=True, device=model.device):
                    placeholder.markdown(response)
                    output_text = response
                    out_len = len(tokenizer.encode(response))
                    total_len = out_len + input_ids.shape[1]
                    st.session_state.length = total_len
                    my_bar.progress(st.session_state.length/model.config.model_max_length, text=f"当前token总数为: {st.session_state.length}/{model.config.model_max_length}")
                
                st.session_state.perf = out_len / (time.time() - start_time)

            print(output_text)
            tool, *out_text = output_text.strip().split('\n')
            if tool in tools:
                messages.append({"role": "tool", "content": response})
                out_text = '\n'.join(out_text)
                code = extract_code(out_text)
                args = eval(code, {'tool_call': tool_call}, {})
                if tool == 'get_search' and tab == Engine.rag:
                    observation, rag_flag = get_rag(args['query'],tokenizer)
                    st.session_state.rag_flag = rag_flag
                    
                    if not rag_flag:
                        with st.chat_message("observation", avatar='🦉'):
                            placeholder = st.empty()
                            rag_messages = [{'role':'user','content':observation}]
                            rag_input_ids = datatool.build_chat_input(model, tokenizer, rag_messages, generation_config.max_new_tokens)
                            for resp in datatool.chat(model, rag_input_ids, stream=True, device=model.device):
                                placeholder.markdown(resp)
                                observation = resp
                    else:
                        with st.chat_message("observation", avatar='🦉'):
                            placeholder = st.empty()
                            placeholder.markdown(observation)
                                
                    messages.append({'role':'observation','content':observation})
                elif (tool == 'get_search' and tab == Engine.google) or tool == 'get_weather':
                    observation = dispatch_tool(tool, args)
                    st.session_state.rag_flag = True
                    messages.append({'role':'observation','content':observation})
                    with st.chat_message("observation", avatar='🦉'):
                        placeholder = st.empty()
                        placeholder.markdown(observation)    
                else:
                    st.session_state.rag_flag = False  
            else:
                messages.append({"role": "assistant", "content": response})
                st.session_state.rag_flag = False 
            print(json.dumps(messages, ensure_ascii=False), flush=True)
            

    col1, col2, col3 = st.columns([8,1,1])
    with col1:
        st.button("清空对话", on_click=clear_chat_history)
    with col2:
        st.button("👍", key="good case", type="primary", on_click=goodcase_status)
    with col3:
        st.button("👎", key="bad case", type="primary", on_click=badcase_status)



if __name__ == "__main__":
    main()