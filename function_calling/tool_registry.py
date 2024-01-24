from copy import deepcopy
import inspect
import json
import datetime
import pytz
import requests
from pprint import pformat
import traceback
from types import GenericAlias
from typing import get_origin, Annotated
from function_calling.search_engine import GoogleSerperAPIWrapper 

_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = {}


def register_tool(func: callable):
    tool_name = func.__name__
    tool_description = inspect.getdoc(func).strip()
    python_params = inspect.signature(func).parameters
    tool_params = []
    for name, param in python_params.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter `{name}` missing type annotation")
        if get_origin(annotation) != Annotated:
            raise TypeError(f"Annotation type for `{name}` must be typing.Annotated")

        typ, (description, required) = annotation.__origin__, annotation.__metadata__
        typ: str = str(typ) if isinstance(typ, GenericAlias) else typ.__name__
        if not isinstance(description, str):
            raise TypeError(f"Description for `{name}` must be a string")
        if not isinstance(required, bool):
            raise TypeError(f"Required for `{name}` must be a bool")

        tool_params.append({
            "name": name,
            "description": description,
            "type": typ,
            "required": required
        })
    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "params": tool_params
    }

    print("[registered tool] " + pformat(tool_def))
    _TOOL_HOOKS[tool_name] = func
    _TOOL_DESCRIPTIONS[tool_name] = tool_def

    return func


def dispatch_tool(tool_name: str, tool_params: dict) -> str:
    if tool_name not in _TOOL_HOOKS:
        return f"Tool `{tool_name}` not found. Please use a provided tool."
    tool_call = _TOOL_HOOKS[tool_name]
    try:
        ret = tool_call(**tool_params)
    except:
        ret = traceback.format_exc()
    return str(ret)


def get_tools() -> dict:
    return deepcopy(_TOOL_DESCRIPTIONS)

weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
# Tool Definitions
@register_tool
def get_weather(location: Annotated[str, 'The location of a city or place to be queried', True],) -> str:
    """
    Get the weather of today and next 3 days for `location`
    """

    if not isinstance(location, str):
        raise TypeError("Location name must be a string")

    key_selection = {
        "forecast": ["province", "city", "temperature", "humidity", "weather", "winddirection", "windpower", "reporttime"],
    }
    eastern = pytz.timezone('Asia/Shanghai')
    current_time = datetime.datetime.now(eastern)
    weekday_num = current_time.weekday()
    weekday = weekdays[weekday_num]

    try:
        addr_resp = requests.get(
            f"https://restapi.amap.com/v3/geocode/geo?address={location}&key=a6bef5e3db6f4c4dd5f289c0f005af67")
        adcode = addr_resp.json()['geocodes'][0]['adcode']
        resp = requests.get(
            f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&extensions=all&key=a6bef5e3db6f4c4dd5f289c0f005af67")
        resp = resp.json()

        ret = resp['forecasts'][0]['casts']
    except:
        import traceback
        ret = "Error encountered while fetching weather data!\n" + traceback.format_exc()
    return f'API查询到的今天及未来3天{location}的天气预报情况的信息如下：' + '\n'.join([str(item) for item in ret]) + \
           f'\n当前的时间是：{current_time.strftime("%Y-%m-%d %H:%M")} ({weekday})'


search = GoogleSerperAPIWrapper()

@register_tool
def get_search(query: Annotated[str, 'The query to search by Google', True]) -> str:
    """
    Get the information about `query` by search engine
    """
    if not isinstance(query, str):
        raise TypeError("Query name must be a string")

    res_arr = []
    ret = search.run(query)
    for item in ret:
        res_arr.append(item['content'].replace("...",""))
    res_str = "\n\n".join(res_arr)

    return f'API检索到的关于{query}的信息如下:\n\n' + res_str


@register_tool
def get_reservation(requirement: Annotated[str, 'The information for booking hotels, flights or train tickets', True]) -> str:
    """
    Get the information about `requirement` to book hotels, flights or train tickets
    """
    if not isinstance(requirement, str):
        raise TypeError("requirement name must be a string")

    return f'已经为您查到的关于{requirement}的信息，希望对您有所帮助！' 


poi_keywors_list = ["价格","门票","地址","位置","优惠","政策","票价","开放时间"]
data_path = "/data/share_user/zzd/data/rlhf_data/tmp_data/"
def get_retrieval(query, tokenizer,
                  url="http://general.retrieval.ctripcorp.com/general_retrieval", 
                  districts="阳朔", req_id=1, topk=5, source="lvp"):
    flag = False
    for keyword in poi_keywors_list:
        if keyword in query:
            flag = True
            break
    if flag:
        topk = 2
        source="poipolicy"
    
    data={
        "reqId": req_id,
        "query": query,
        # "districts": "上海",
        # "pioids":["9262827"],
        "source": source,
        # "source": "lvp",
        # "publish_time_min": "2020-01-01",
        "return_fields": ["title", "content", "publish_time"],
        "topK": 10
    }
    
    eastern = pytz.timezone('Asia/Shanghai')
    current_time = datetime.datetime.now(eastern)
    weekday_num = current_time.weekday()
    weekday = weekdays[weekday_num]
    
    resp = requests.post(url, json=data)
    response = json.loads(resp.text)['results']
    text_out = '\n'.join([str(item['source']) for item in response[:topk]])
    text_len = len(tokenizer.encode(text_out))
    if text_len < 200:
        return f'API检索到的关于{query}的信息如下:\n' + text_out, True
    
    with open(data_path+"search_data.txt", "w") as file:
        file.write(text_out)
    return f'API查询到的关于{query}的可能相关的信息如下：' + text_out + \
           f'\n当前的时间是：{current_time.strftime("%Y-%m-%d %H:%M")} ({weekday})', False

def get_rag(query, tokenizer) -> str:
    results = get_retrieval(query, tokenizer)
    return results


if __name__ == "__main__":
    print(get_tools())
    # print(dispatch_tool("get_weather", {"location": "芜湖方特梦幻王国"}))
    # print(dispatch_tool("get_search", {"query": "贡嘎雪山门票"}))
    print(get_rag("鲁迅故里门票价格"))
