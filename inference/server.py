

from typing import Optional, List
import uvicorn
from fastapi import FastAPI, Request, Body
import json
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from utils import InferModel
from openrlhf.datasets.utils import get_system_prompt

app = FastAPI(
    title="对话系统",
    description="""Chatbot""",
    version="0.1.0",)

system_prompt = get_system_prompt()
model_path = '/ckpt/rlhf_baichuan2_path/{date}/sft_hf'
infer_model = InferModel(model_path)


def get_infosec(query, biz="ai_userqa_text", url=""):
    data={
      "appId": "100048217",
      "bizName": biz,
      "ruleParams": "{'uid':'test666', 'ip':'196.168.10.125'}",
      "contentType": "tripshoot",
      "contentText": query
    }
    resp = requests.post(url, json=data)
    return json.loads(resp.text)


def init_messages():
    messages = [{"role": "system", "content": system_prompt}]
    return messages

messages = init_messages()

@app.post("/chat")
def gen_chat(query: str = Body(1, title='对话前文', embed=True), restart: str = Body(1, title='是否重启', embed=True)):
    print(restart)
    global messages
    if restart == 'true':
        messages = init_messages()
    
    messages += [{'role':'user', 'content':query}]
    sec_q_res = get_infosec(query)

    if sec_q_res['keywordResult']['enResult'] == 'PASS':
        response = infer_model.chat(messages, stream=False)

        sec_a_res = get_infosec(response, biz='ai_useraiqa_text')
        if sec_a_res['keywordResult']['enResult'] == 'NOT_PASS':
            response = "抱歉，这个问题我不清楚。如果你有更多关于旅游相关问题，欢迎向我提问。"
        
    else:
        response = "抱歉，该问题我无法回答。如果你有更多关于旅游相关问题，欢迎向我提问。"
    messages += [{'role':'assistant', 'content':response}]
    
    print(query)
    print(response)
    print(messages)
    return {'response': response}




if __name__ == "__main__":
    uvicorn.run("main:app", host="10.59.144.213", port=8036)