# -*- coding: utf-8 -*-
# @Time    : 2022/4/14 17:02
# @Author  : Damon Li
# @Email   : bingzhenli@hotmail.com
# @File    : customer_test.py
# @Project : task-oriented-dialogue-system-for-smart-follow-up


import jwt
import json
import requests
import secrets

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--request_url", type=str, required=True,
                    help="""medical chatbot url address. If run in localhost, it should break\
                    'http://localhost:8686/conversations'""")
args = vars(parser.parse_args())


# 本地："http://localhost:8686/conversations" 
# gpu私有云："http://192.168.60.10:25650/conversations"
url = args["request_url"]

# 多轮调用
def chatbot(conversation_id, text):
    try:
        post_data = json.dumps(
            {"text": text},
            ensure_ascii=False
        ).encode(encoding="utf-8")

        response = requests.request("POST",
                                    "{0}/{1}".format(url, conversation_id),
                                    data=post_data)
        return response

    except Exception as e:
        print("[ERROR] text -> {}\n error -> {}".format(text, e))


if __name__ == '__main__':
    sender_id = secrets.token_urlsafe(16) # 随机生成会话id

    while True:
        user_input = input("User({0}) -> ".format(sender_id))
        res = chatbot(sender_id, user_input)
        res_parse = json.loads(res.text.encode('utf8'))
        if res_parse:
            print("Robot -> {0}".format(res_parse))

        if user_input == 'Q' or user_input == 'q':
            break
