# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, FollowupAction
from collections import defaultdict
from py2neo import Graph

from .robot_20220322.robot_20220306_cpu import Robot

# url = "http://192.168.16.164:31146/api/model/diagnose/diagnosenet/predict"
# headers = {
#     'Content-Type': 'application/json',
#     'token': 'IHTFUSDEUFSLKO89fdhf)FH*WFBfdk'
# }

local_neo4jdb = Graph(host="localhost",  # neo4j 搭载服务器的ip地址，ifconfig可获取到
                      port=7687,  # neo4j 服务器监听的端口号
                      user="neo4j",
                      password="123456")

robot = Robot()

class ActionPositiveOrNegative(Action):
    def name(self) -> Text:
        return "action_positive_or_negative"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any],
            ) -> List[Dict[Text, Any]]:

        _is_enough = tracker.slots["is_enough"]
        _is_break_loop = tracker.slots["is_break_loop"]

        if tracker.slots["diagnosis_choice"] == 'positive':
            robot.input_sid((tracker.slots["suggestion_node"]), tracker.sender_id)
            _is_enough = robot.is_enough(tracker.sender_id)
            _is_break_loop = True
        else:
            robot.update_wrong_sid([tracker.slots["suggestion_node"]], tracker.sender_id)
            _is_enough = robot.is_enough(tracker.sender_id)
            if _is_enough:
                _is_break_loop = True

        return [SlotSet("is_enough", _is_enough),
                SlotSet("is_break_loop", _is_break_loop),
                FollowupAction("action_arouse_diagnosis_api")]

class ActionArouseDiagnosisApi(Action):

    def name(self) -> Text:
        return "action_arouse_diagnosis_api"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
            ) -> List[Dict[Text, Any]]:

        # current_place = next(tracker.get_latest_entity_values("place"), None)
        # text_date = tracker.get_slot("date") or "今天"

        _is_enough = tracker.slots["is_enough"]
        _is_break_loop = tracker.slots["is_break_loop"]
        _is_start_diagnosis_query = tracker.slots["is_start_diagnosis_query"]
        _is_reg_chief_complaint = tracker.slots["is_reg_chief_complaint"]

        # 调用robot进行多轮次诊断疾病
        if not tracker.slots["is_start_diagnosis_query"]:
            print("[INFO] 当前意图：{}".format(tracker.get_intent_of_latest_message()))
            print("[INFO] 当前词槽：{}".format(tracker.current_slot_values()))

            symptom_set = set()
            for entity_info in tracker.latest_message["entities"]:
                if entity_info["entity"] == "symptom":
                    symptom_set.add(entity_info["value"])
            print("[INFO] 当前主诉症状：{}".format(symptom_set))

            if not symptom_set:
                # 如果检测不出症状实体
                dispatcher.utter_message(
                    text="不好意思，我无法识别当前描述的症状，麻烦重新描述一遍！"
                )
            else:
                try:
                    robot.reset(tracker.sender_id)
                    for symptom in symptom_set:
                        robot.input_sid((robot.cname2id[symptom]), tracker.sender_id)  # there is no relations
                    tracker.slots["suggestions"], _ = robot.suggestion(tracker.sender_id, num=10)  # 10 symptoms loop
                    _is_start_diagnosis_query = True
                    _is_reg_chief_complaint = True
                except Exception as e:
                    print("[ERROR] {}".format(e))
                    dispatcher.utter_message(
                        text="不好意思，我无法识别当前描述的症状，麻烦重新描述一遍！"
                    )

        if _is_reg_chief_complaint:
            if not _is_enough:
                if _is_break_loop:
                    suggestions, _ = robot.suggestion(tracker.sender_id, num=10)  # 10 symptoms loop
                    tracker.slots["suggestions"] = suggestions

                # print("[INFO] 当前所有suggestions结点：", tracker.slots["suggestions"])
                suggestion_node = tracker.slots["suggestions"].pop(0)
                question = "{0}？".format(robot.id2cname[suggestion_node])

                # ask question
                if suggestion_node in robot.wrong_set[tracker.sender_id]:
                    patient_choice = 'positive'
                    print('[INFO] has answered before, answer is: {0}'.format(patient_choice))
                    return [SlotSet("is_enough", _is_enough),
                            SlotSet("is_break_loop", _is_break_loop),
                            SlotSet("is_start_diagnosis_query", _is_start_diagnosis_query),
                            SlotSet("is_reg_chief_complaint", _is_reg_chief_complaint),
                            SlotSet("suggestions", tracker.slots["suggestions"]),
                            SlotSet("diagnosis_choice", patient_choice)]

                elif suggestion_node in robot.selected_set[tracker.sender_id]:
                    patient_choice = 'negative'
                    print('[INFO] has answered before, answer is: {0}'.format(patient_choice))
                    return [SlotSet("is_enough", _is_enough),
                            SlotSet("is_break_loop", _is_break_loop),
                            SlotSet("is_start_diagnosis_query", _is_start_diagnosis_query),
                            SlotSet("is_reg_chief_complaint", _is_reg_chief_complaint),
                            SlotSet("suggestions", tracker.slots["suggestions"]),
                            SlotSet("diagnosis_choice", patient_choice)]

                else:
                    print('[INFO] asking this node:{} - {}'.format(suggestion_node, robot.id2cname[suggestion_node]))
                    dispatcher.utter_message(
                        text=question,
                        buttons=[
                            {"title": "是", "payload": '/affirm{"diagnosis_choice":"positive"}'},
                            {"title": "否", "payload": '/deny{"diagnosis_choice":"negative"}'}
                        ],
                    )

                    return [SlotSet("is_enough", _is_enough),
                            SlotSet("is_break_loop", _is_break_loop),
                            SlotSet("is_start_diagnosis_query", _is_start_diagnosis_query),
                            SlotSet("is_reg_chief_complaint", _is_reg_chief_complaint),
                            SlotSet("suggestions", tracker.slots["suggestions"]),
                            SlotSet("suggestion_node", suggestion_node)]

            else:
                # dispatcher.utter_message(
                #     text="请选择",
                #     buttons=[
                #         {'title': '预约挂号',
                #          'payload': '/need_registration{"is_press_registration_button":true}'},
                #         {'title': "推荐用药",
                #          'payload': '/need_drug_recommendation{"is_press_drug_recommendation_button":true}'}
                #     ]
                # )
                dispatcher.utter_message(
                    response="utter_ask_registration_or_drug_recommendation"
                )

                # e.g. disease_pred = (['心肌梗死', '原发性高血压', '高血压性心肌病', '劳力性心绞痛', '非ST段抬高心肌梗死',
                # '妊娠期高血压', '心绞痛', '咳嗽变异性哮喘', '夜间哮喘', '肥厚型心肌病'],
                # (0.27688052633435456, 0.24572468816045054, 0.11915590058399547, 0.09788156999945019,
                # 0.07893838304624383, 0.058386391286157795, 0.03823498828103523, 0.009455128682949596,
                # 0.006946625154820114, 0.006694486892791386))
                disease_pred = robot.inference_disease_with_probs(tracker.sender_id, num=10)
                print("[INFO] 疾病诊断结果：{0}".format(disease_pred))

                department_recommend = robot.recommend_departments(tracker.sender_id)
                print("[INFO] 推荐多个科室：{0}".format(department_recommend))

                robot.reset(tracker.sender_id)
                _is_start_diagnosis_query = False
                _is_enough = False
                _is_break_loop = False
                _is_reg_chief_complaint = False

                return [SlotSet("is_enough", _is_enough),
                        SlotSet("is_break_loop", _is_break_loop),
                        SlotSet("is_start_diagnosis_query", _is_start_diagnosis_query),
                        SlotSet("is_reg_chief_complaint", _is_reg_chief_complaint),
                        SlotSet("disease", disease_pred[0][0]),
                        SlotSet("department", department_recommend[0]),
                        SlotSet("suggestions", []),
                        SlotSet("suggestion_node", None),
                        SlotSet("diagnosis_choice", None)]

class ActionCheckDiseaseOrDepSlot(Action):
    def name(self) -> Text:
        return "action_check_disease_or_dep_slot"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
            ) -> List[Dict[Text, Any]]:

        print("[INFO] 当前意图：{}".format(tracker.get_intent_of_latest_message()))
        print("[INFO] 当前词槽：{}".format(tracker.current_slot_values()))

        res = defaultdict(str)
        for idx, entity_info in enumerate(tracker.latest_message["entities"]):
            res[entity_info["entity"]] = entity_info["value"]

        # 刚做完问诊流程来到”预约挂号”和“推荐用药”选项处，可以判断“is_press_registration_button”实体是否在最新的消息里面
        # 如果存在，则直接跳到推荐科室以及询问是否需要医生信息；否则导向用户重新询问疾病名称，科室名称或从头输入症状
        if "is_press_registration_button" in res:
            dispatcher.utter_message(text="经过诊断后推荐您挂{0}".format(tracker.slots["department"]))
            dispatcher.utter_message(
                response="utter_ask_check_doctors",
                department=tracker.slots["department"]
            )
            return []
            # return [SlotSet("is_press_registration_button", True),
            #         FollowupAction("action_response_for_selecting_registration")]

        if not res["department"] and not res["disease"]:
            dispatcher.utter_message(text="好的，您可以直接回复所患疾病名称或科室名称进行挂号，或者可以回复不适的症状")
            return []

        elif not res["department"] and res["disease"]:
            department_recommend = local_neo4jdb.run(
                """
                MATCH(m:Disease)-[:belong_to]->(n:Department) WHERE m.name="{0}" RETURN n.name AS Department
                """.format(res["disease"])
            ).data()
            if department_recommend:
                res["department"] = department_recommend[0]["Department"]
                print("[INFO] {0}(疾病)匹配到{1}(科室)".format(res["disease"], res["department"]))
                dispatcher.utter_message(text="推荐您挂{0}".format(res["department"]))
                dispatcher.utter_message(
                    response="utter_ask_check_doctors",
                    department=res["department"]
                )
                return [SlotSet("disease", res["disease"]),
                        SlotSet("department", res["department"])]
            else:
                # # 判断当前实体是否是标准症状术语，因为就可能是症状的实体预测成成疾病实体
                # if res["disease"] in standard_symptoms:
                dispatcher.utter_message(text="无法识别该疾病名称，请重新再说一遍")
            return [SlotSet("disease", res["disease"])]

        elif res["department"]:
            # TODO: return doctors info from database
            # dispatcher.utter_message(image="http://localhost:8000/img/doctor_info.jpg")
            # TODO: add doctor names list onto buttons
            dispatcher.utter_message(
                text="已为您提供{0}相关医生信息，请选择挂号".format(res["department"]),
                buttons=[
                    {"title": "李医生", "payload": '/select_doctor{"doctor":"李医生"}'},
                    {"title": "黄医生", "payload": '/select_doctor{"doctor":"黄医生"}'},
                    {"title": "陈医生", "payload": '/select_doctor{"doctor":"陈医生"}'},
                    {"title": "梁医生", "payload": '/select_doctor{"doctor":"梁医生"}'},
                    {"title": "曾医生", "payload": '/select_doctor{"doctor":"曾医生"}'},
                    {"title": "蒋医生", "payload": '/select_doctor{"doctor":"蒋医生"}'},
                ],
            )

            return [SlotSet("department", res["department"])]

class ActionArouseDrugRecommendationApi(Action):
    def name(self) -> Text:
        return "action_arouse_drug_recommendation_api"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
            ) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="TODO -> 调用用药推荐接口")

        return []

class ActionCheckDoctorsInfo(Action):
    def name(self) -> Text:
        return "action_check_doctors_info"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
            ) -> List[Dict[Text, Any]]:
        if tracker.slots["department"]:
            # TODO: return doctors info from database
            # dispatcher.utter_message(image="http://localhost:8000/img/doctors_info.jpg")
            # TODO: add doctor names list onto buttons
            dispatcher.utter_message(
                text="已为您提供{0}相关医生信息，请选择挂号".format(tracker.slots["department"]),
                buttons=[
                    {"title": "李医生", "payload": '/select_doctor{"doctor":"李医生"}'},
                    {"title": "黄医生", "payload": '/select_doctor{"doctor":"黄医生"}'},
                    {"title": "陈医生", "payload": '/select_doctor{"doctor":"陈医生"}'},
                    {"title": "梁医生", "payload": '/select_doctor{"doctor":"梁医生"}'},
                    {"title": "曾医生", "payload": '/select_doctor{"doctor":"曾医生"}'},
                    {"title": "蒋医生", "payload": '/select_doctor{"doctor":"蒋医生"}'},
                ],
            )
        return []

class ActionStartUpRegisterService(Action):
    def name(self) -> Text:
        return "action_start_up_register_service"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
            ) -> List[Dict[Text, Any]]:

        # TODO: start up registration service

        if tracker.get_slot("doctor"):
            dispatcher.utter_message(text="已成功为您预约{0}".format(tracker.get_slot("doctor")))
        else:
            dispatcher.utter_message(text="请选择医生")

        return []
