# -*- coding: utf-8 -*-
# @Time    : 2022/4/14 16:45
# @Author  : Damon Li
# @Email   : bingzhenli@hotmail.com
# @File    : sinohealth_rasa_robot_server.py
# @Project : task-oriented-dialogue-system-for-smart-follow-up


import json
import requests
import jwt

from sanic import Sanic
import sanic.response as sanic_response
from sanic.request import Request
from sanic_cors import CORS

from collections import defaultdict
from typing import Any, Callable
from functools import wraps
import pandas as pd

# 加载问诊类
from actions.robot_20220322.robot_20220306_cpu import Robot

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--core_service_url", type=str, required=True,
                    help="RASA核心服务的URL地址")
parser.add_argument("--host", type=str, required=True,
                    help="指定本机服务URL")
parser.add_argument("--port", type=int, default=8686,
                    help="指定本机服务端口")
args = vars(parser.parse_args())

DEFAULT_RESPONSE_TIMEOUT = 60 * 60 # 1h

# 放在actions目录下面，这里要加main，不然在开启actions服务的时候也会执行这里的程序入口
# 一般user没有权限，仅有admin没有username也是可以访问
payload = {"user": {"username": "damon", "role": "admin"}}  # 可以访问
signed = jwt.encode(payload, "sinohealth_rasa_robot", algorithm="HS256")
headers = {
    'Authorization': 'Bearer {0}'.format(signed),
    'Content-Type': 'text/plain'
}

# 本机服务地址："http://localhost:5005"
# gpu私有云服务地址："http://192.168.60.10:25071"
baseUrl = args["core_service_url"] 

robot = Robot()

# 读取图谱数据，为了查询症状对应的snomed_id
df = pd.read_csv("./data/graph_20211207.csv")
df = df[(df['relation'] == 'has symptom_main') | (df['relation'] == 'has symptom')]
df = pd.DataFrame({'symptoms': df['e_name'], 'snomed_id': df['e_code']})

class ChatbotObj(object):
    def __init__(self):
        self.action = defaultdict(str)  # chatbot的动作
        self.messages = defaultdict(str)  # rasa回应的信息
        self.user_input_chief_complaint = defaultdict(list)  # 用户主诉
        self.symptom_list = defaultdict(list)  # 用户症状列表
        self.diagnosis_process_status = defaultdict(str)  # 用户问诊流程状态
        self.diagnosis_disease = defaultdict(str)  # 诊断疾病名称

# 美化终端输出
def colorstr(*input, color="blue", bold=False, underline=False):
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    prefix = colors[color]
    if bold:
        prefix += colors["bold"]
    if underline:
        prefix += colors["underline"]

    return "{}{}{}".format(prefix, input[0], colors["end"])

# 根据conversation_id，获取对应的实例化的chatbot对象
def get_chatterbot_instance(chatterbot: ChatbotObj) -> Callable[[Callable], Callable[..., Any]]:
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args: Any, **kwargs: Any) -> Any:
            return f(chatterbot=chatterbot, *args, **kwargs)
        return decorated
    return decorator

# 创建APP
def create_app(response_timeout: int = DEFAULT_RESPONSE_TIMEOUT) -> Sanic:
    app = Sanic("sinohealth_rasa_robot_service")
    CORS(app)
    app.config.RESPONSE_TIMEOUT = response_timeout
    chatterbot_instance = ChatbotObj()

    def predict_and_execute_next_action(predict_url, execute_url, chatterbot: ChatbotObj, conversation_id: str):
        # 对当前用户的输入预测的下一步action
        response_json = json.loads(
            requests.request("POST", predict_url, headers=headers).text.encode('utf8')
        )

        # 取出置信度最高的下一步动作
        chatterbot.action[conversation_id] = response_json["scores"][0]["action"]

        print(
            colorstr("[BACKEND] robot predicted next action   - {0}", color="green").format(
                colorstr(chatterbot.action[conversation_id], color="bright_green")
            )
        )

        # 执行预测动作
        data = json.dumps(
            {"name": chatterbot.action[conversation_id]},
            ensure_ascii=False
        ).encode(encoding="utf-8")
        response_json = json.loads(
            requests.request("POST", execute_url, headers=headers, data=data).text.encode('utf8')
        )

        # 执行动作后，解析robot返回的数据
        chatterbot.messages[conversation_id] = response_json["messages"]
        if chatterbot.messages[conversation_id]:
            _printout = "\n"
            for _id, message in enumerate(chatterbot.messages[conversation_id]):
                _title_list = []
                if "buttons" in message:
                    for button in message["buttons"]:
                        _title_list.append(button["title"])

                _title_str = ""
                if len(_title_list) == 2:
                    _title_str = "(请回答'{0}'或'{1}')".format(_title_list[0], _title_list[1])
                elif len(_title_list) > 2:
                    _title_str = "(可选: {0})".format(", ".join(_title_list))

                _printout += colorstr("Robot -> {0}{1}", color="red", bold=True).format(message["text"],
                                                                                        _title_str)
                if _id != len(chatterbot.messages[conversation_id]) - 1:
                    _printout += "\n"
            print(_printout)
        else:
            chatterbot.messages[conversation_id] = []

        return chatterbot

    @app.post("/conversations/<conversation_id:path>")
    @get_chatterbot_instance(chatterbot_instance)  # 根据conversation_id，获取对应的实例化的chatbot对象
    async def rasa_robot(request: Request, conversation_id: str, chatterbot: ChatbotObj = None):
        # 如果当前chatterbot对象的动作不存在，则设置为"action_listen"
        if not chatterbot.action[conversation_id]:
            chatterbot.action[conversation_id] = "action_listen"  # default="action_listen", 动作初始化为等待输入

        # 如果当前chatterbot对象的信息不存在，则初始化为None
        if not chatterbot.messages[conversation_id]:
            chatterbot.messages[conversation_id] = None  # default=None

        # 上行的json本体，并获取text字段数据
        req_body = request.json
        text = req_body["text"]

        # 需用到的各种RASA的http接口
        intent_cls_url = "{0}/model/parse".format(baseUrl)  # 预测当前用户输入的意图
        messages_url = "{0}/conversations/{1}/messages".format(baseUrl, conversation_id)  # 发送消息
        predict_url = "{0}/conversations/{1}/predict".format(baseUrl, conversation_id)  # 预测下一步动作
        execute_url = "{0}/conversations/{1}/execute".format(baseUrl, conversation_id)  # 执行动作
        inject_intent_url = "{0}/conversations/{1}/trigger_intent".format(baseUrl, conversation_id)  # 注入新的意图
        retrieve_tracker_url = "{0}/conversations/{1}/tracker".format(baseUrl, conversation_id)  # 获取tracker信息

        # 打印chatterbot对上一次动作
        print(
            colorstr("[BACKEND] last action                   - {0}", color="green").format(
                colorstr(chatterbot.action[conversation_id], color="bright_green")
            )
        )

        # 打印用户当前输入
        print(
            colorstr("[BACKEND] current user input            - {0}", color="green").format(
                colorstr(text, color="bright_green")
            )
        )

        if chatterbot.action[conversation_id] in ["action_listen", "action_default_fallback", "action_restart"]:
            # 如果当前chatterbot是在监听状态，则将用户的text信息post到rasa core服务端
            data = json.dumps(
                {"text": text, "sender": "user"},  # 一定要"user"才行
                ensure_ascii=False
            ).encode(encoding="utf-8")
            requests.request("POST", messages_url, headers=headers, data=data)

        # 当用户在问诊结束后输错意图时（暂时只允许“预约挂号”和“推荐用药”两个后续意图），则重新激活
        # utter_ask_registration_or_drug_recommendation动作，并用该变量记录服务端返回的json信息
        wrong_input_response_json = None  #

        _diagnosis_choice = None  #

        # 如果当前chatterbot的动作是唤起问诊服务的动作"action_arouse_diagnosis_api"
        if chatterbot.action[conversation_id] == "action_arouse_diagnosis_api":
            # 如果上一次返回的messages中包含buttons，判断是否已经进入问诊流程（这里逻辑可以优化）
            if chatterbot.messages[conversation_id] and "buttons" in chatterbot.messages[conversation_id][0]:
                # 将用户上行的文本text构建成json参数
                data = json.dumps(
                    {"text": text},
                    ensure_ascii=False
                ).encode(encoding="utf-8")
                # 将json参数上传intent_cls_url进行意图分类
                response_raw = requests.request("POST", intent_cls_url, headers=headers, data=data)
                # 解析返回的数据，得到意图分类结果
                response_json = json.loads(response_raw.text.encode('utf8'))

                if chatterbot.messages[conversation_id][0]['text'] == '请选择':
                    payload = None
                    # 问诊结束后给出选项按钮，包含【”请选择“文本+”预约挂号“选项+”推荐用药“选项】
                    if response_json["intent"]["name"] == "need_registration":
                        # 构建预约挂号的http请求数据体
                        payload = json.dumps({
                            "name": "need_registration",
                            "entities": {
                                "is_press_registration_button": 'true'
                            }
                        })
                    elif response_json["intent"]["name"] == "need_drug_recommendation":
                        # 构建推荐用药的http请求数据体
                        payload = json.dumps({
                            "name": "need_drug_recommendation",
                            "entities": {
                                "is_press_drug_recommendation_button": 'true'
                            }
                        })
                    else:
                        # 当用户在问诊结束后输错意图时（暂时只允许“预约挂号”和“推荐用药”两个后续意图），
                        # 则重新激活utter_ask_registration_or_drug_recommendation动作
                        data = json.dumps(
                            {"name": "utter_ask_registration_or_drug_recommendation"},
                            ensure_ascii=False
                        ).encode(encoding="utf-8")
                        wrong_input_response_json = json.loads(
                            requests.request("POST", execute_url, headers=headers, data=data).text.encode('utf8')
                        )

                    # 注入意图并携带实体到rasa服务端
                    requests.request("POST", inject_intent_url, headers=headers, data=payload)

                else:
                    # 如果用户输入的意图是肯定的意图"affirm"，则意味着当前的症状选择为阳性，否则为阴性
                    if response_json["intent"]["name"] == "affirm":
                        _diagnosis_choice = "positive"
                        # 构建http请求数据体
                        payload = json.dumps({
                            "name": "affirm",
                            "entities": {
                                "diagnosis_choice": _diagnosis_choice
                            }
                        })

                        # 获取tracker信息
                        response_json = json.loads(
                            requests.request("GET", retrieve_tracker_url, headers=headers).text.encode('utf8')
                        )

                        # 一定在注入新的“affirm”之前，获取text加入症状列表
                        _symptom = robot.id2cname[response_json['slots']['suggestion_node']]

                        _snomedid_df = df[df['symptoms'] == _symptom]['snomed_id']
                        if len(_snomedid_df) == 0:
                            _snomed_id = ""
                        else:
                            _snomed_id = _snomedid_df.iloc[0]

                        chatterbot.symptom_list[conversation_id].append({"standard_terminology": _symptom,
                                                                         "snomed_id": _snomed_id})

                    else:
                        _diagnosis_choice = "negative"
                        # 构建http请求数据体
                        payload = json.dumps({
                            "name": "deny",
                            "entities": {
                                "diagnosis_choice": _diagnosis_choice
                            }
                        })

                    print(
                        colorstr("[BACKEND] symptom positive or negative  - {0}", color="green").format(
                            colorstr(_diagnosis_choice, color="bright_green")
                        )
                    )

                    # 注入"affirm"意图并携带"diagnosis_choice"实体到rasa服务端
                    requests.request("POST", inject_intent_url, headers=headers, data=payload)

        # 如果用户没有输入错误的意图，再继续执行下面一系列操作
        cant_detect_symptom_message = None
        if not wrong_input_response_json:
            chatterbot = predict_and_execute_next_action(predict_url, execute_url, chatterbot, conversation_id)

            # 重置当前action为action_listen
            if chatterbot.action[conversation_id] not in ["action_arouse_diagnosis_api", "action_positive_or_negative"]:
                chatterbot.action[conversation_id] = "action_listen"
                data = json.dumps(
                    {"name": chatterbot.action[conversation_id]},
                    ensure_ascii=False
                ).encode(encoding="utf-8")
                requests.request("POST", execute_url, headers=headers, data=data).text.encode('utf8')

            elif chatterbot.action[conversation_id] == "action_positive_or_negative":
                # 如果当前的动作是action_positive_or_negative时，执行多一次预测与执行操作，在actions.py代码可以看到，
                # "执行多一次预测与执行操作"意味着继续followup（调起）action_arouse_diagnosis_api动作
                chatterbot = predict_and_execute_next_action(predict_url, execute_url, chatterbot, conversation_id)

            if chatterbot.action[conversation_id] == "action_arouse_diagnosis_api":
                if "buttons" not in chatterbot.messages[conversation_id][0]:
                    cant_detect_symptom_message = chatterbot.messages[conversation_id]
                    predict_and_execute_next_action(predict_url, execute_url, chatterbot, conversation_id)

            # 获取tracker信息
            response_json = json.loads(
                requests.request("GET", retrieve_tracker_url, headers=headers).text.encode('utf8')
            )

            # 通过tracker里面的‘is_start_diagnosis_query’词槽来判断是否开启问诊流程
            if response_json['slots']['is_start_diagnosis_query']:
                # 如果上一个状态是off，当准备开启新的问诊对话时，先做初始化
                if chatterbot.diagnosis_process_status[conversation_id] == 'off':
                    chatterbot.diagnosis_disease[conversation_id] = ""
                    chatterbot.symptom_list[conversation_id] = []
                    chatterbot.user_input_chief_complaint[conversation_id] = []

                # 更新诊断流程状态
                chatterbot.diagnosis_process_status[conversation_id] = 'on-going'

                # 保存用户主诉症状以及snomedid
                if not chatterbot.user_input_chief_complaint[conversation_id]:
                    main_symptom = response_json['slots']['symptom']
                    main_symptom_snomed_id_df = df[df['symptoms'] == main_symptom]['snomed_id']
                    if len(main_symptom_snomed_id_df) == 0:
                        main_symptom_snomed_id = ""
                    else:
                        main_symptom_snomed_id = main_symptom_snomed_id_df.iloc[0]
                    chatterbot.user_input_chief_complaint[conversation_id] = {"standard_terminology": main_symptom,
                                                                              "snomed_id": main_symptom_snomed_id}
                    chatterbot.symptom_list[conversation_id].append({"standard_terminology": main_symptom,
                                                                     "snomed_id": main_symptom_snomed_id})
            else:
                if chatterbot.diagnosis_process_status[conversation_id] == 'on-going':
                    # 如果上一个状态是on-going，但下一个状态是off，说明诊断流程结束，获取疾病名称
                    chatterbot.diagnosis_disease[conversation_id] = response_json['slots']['disease']

                # 重置保存列表，等待下次问诊流程使用
                chatterbot.diagnosis_process_status[conversation_id] = 'off'

        # 如果用户输入错误意图，意味着chatterbot.messages[conversation_id]是空列表，且wrong_input_response_json不为None
        if not chatterbot.messages[conversation_id] and wrong_input_response_json:
            # 则将服务端应对用户错误意图的反馈信息赋值到chatterbot.messages[conversation_id]
            chatterbot.messages[conversation_id] = wrong_input_response_json["messages"]

        # 用户输入的主诉症状，chatbot无法识别处理
        if not chatterbot.messages[conversation_id] and cant_detect_symptom_message:
            chatterbot.messages[conversation_id] = cant_detect_symptom_message

        output_dict = chatterbot.messages[conversation_id][0]
        if len(chatterbot.messages[conversation_id]) != 1:
            _text_list = []
            for message in chatterbot.messages[conversation_id]:
                _text_list.append(message['text'])
            output_dict['text'] = _text_list
        output_dict.update({'chief_complaint': chatterbot.user_input_chief_complaint[conversation_id]})
        output_dict.update({'symptoms': chatterbot.symptom_list[conversation_id]})
        output_dict.update({'disease': chatterbot.diagnosis_disease[conversation_id]})
        output_dict.update({'diagnosis_process_status': chatterbot.diagnosis_process_status[conversation_id]})

        return sanic_response.json(output_dict)

    return app


if __name__ == "__main__":
    app = create_app()

    # gpu私有云：host="0.0.0.0", port=8686
    # 本机：host="localhost", port=8686
    app.run(host=args["host"], port=args["port"])  # 本地机器
