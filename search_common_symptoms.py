# -*- coding: utf-8 -*-
# @Time    : 2022/3/7 13:45
# @Author  : Damon Li
# @Email   : bingzhenli@hotmail.com
# @File    : search_common_symptoms.py
# @Project : task-oriented-dialogue-system-for-smart-follow-up

import collections
import random

from tqdm import tqdm
import pickle
from py2neo import Graph, Node
import pandas as pd
import os
from general import search_sim_terms
import random
import yaml

local_neo4jdb = Graph(host="localhost",  # neo4j 搭载服务器的ip地址，ifconfig可获取到
                      port=7687,  # neo4j 服务器监听的端口号
                      user="neo4j",
                      password="123456"
                      )

# 重新构建disease_diagnosis意图样本数据
df = pd.read_csv("data/症状相关疾病数量.csv")
# RELATIONSHIPS_SAVED_PATH = "data/relationships.pkl"
# with open(RELATIONSHIPS_SAVED_PATH, 'rb') as file:
#     rel_data_dict = pickle.load(file)
with open("nlu/data/nlu_intent_disease_diagnosis.yml", "w", encoding="utf-8") as file_saved:
    disease_diagnosis_intent_samples = """version: "3.0"

nlu:
- intent: disease_diagnosis
  examples: |
    - 不舒服
    - 我不舒服
    - 我有点不舒服
    - 我不太舒服
    - 我身体不舒服
    - 觉得很不舒服，很不自在，有点难受
"""
    for symptom, colloquial_desc_list in df.get(["symptoms", "colloquial_desc"]).values:
        disease_diagnosis_intent_samples += "    - [{0}](symptom)\n".format(symptom)
        pretext = random.choices(["我", "我感到", "我觉得", "觉得", "感觉"], k=1)[0]
        posttext = random.choices(["，该怎么办", "，该怎么处理", "，该怎么弄", "，该如何是好"], k=1)[0]
        disease_diagnosis_intent_samples += "    - {0}[{1}](symptom){2}\n".format(pretext, symptom, posttext)
        for desc in colloquial_desc_list.strip("[]").replace("'", "").replace(" ", "").split(","):
            disease_diagnosis_intent_samples += """    - [{0}]{{"entity":"symptom","value":"{1}"}}\n""".format(desc, symptom)
            pretext = random.choices(["我", "我感到", "我觉得", "觉得", "感觉"], k=1)[0]
            posttext = random.choices(["，该怎么办", "，该怎么处理", "，该怎么弄", "，该如何是好"], k=1)[0]
            disease_diagnosis_intent_samples += """    - {0}[{1}]{{"entity":"symptom","value":"{2}"}}{3}\n""".format(pretext, desc, symptom, posttext)
    file_saved.write(disease_diagnosis_intent_samples)


# 创建need_registration意图数据集
# （1）通过科室名称创建语料，例如“我需要挂[呼吸内科]的号“
# （2）通过疾病名称创建语料，例如“我感冒了，帮我挂号”，”我感冒了，挂哪个科室的号“，或者直接“感冒”
dep = []
with open("nlu/data/nlu_intent_need_registration.yml", "w", encoding="utf-8") as file_saved:
    prefix = """version: "3.0"

nlu:
- intent: need_registration
  examples: |
    - 我想挂号
    - 挂号
    - 我需要挂号
    - 帮我挂号
"""
    with open("nlu/data/departments.txt", "r", encoding="utf-8") as file:
        while True:
            line = file.readline()
            if not line:
                break
            pretext = random.choices(["我想挂", "我需要挂", "帮我挂", ""], k=1)[0]
            posttext = random.choices(["的号", ""], k=1)[0]
            prefix += "    - {0}[{1}](department){2}\n".format(pretext, line.strip("\n"), posttext)

    with open("data/relationships.pkl", 'rb') as file:
        rel = pickle.load(file)

    diseases_from_yining = set()
    for d, _ in rel["Disease-has_main_symptom-Symptom"]:
        diseases_from_yining.add(d)

    for disease in diseases_from_yining:
            pretext = random.choices(["我得了", ""], k=1)[0]
            posttext = random.choices(["，怎么挂号", "，挂哪个科室的号", ""], k=1)[0]
            prefix += "    - {0}[{1}](disease){2}\n".format(pretext, disease, posttext)
            posttext = random.choices(["相关科室的号", ""], k=1)[0]
            prefix += "    - 帮我挂[{0}](disease){1}\n".format(disease, posttext)

    # with open("nlu/data/diseases.txt", "r", encoding="utf-8") as file:
    #     while True:
    #         line = file.readline()
    #         if not line:
    #             break
    #         pretext = random.choices(["我得了", ""], k=1)[0]
    #         posttext = random.choices(["，怎么挂号", "，挂哪个科室的号", ""], k=1)[0]
    #         prefix += "    - {0}[{1}](disease){2}\n".format(pretext, line.strip("\n"), posttext)
    #         posttext = random.choices(["相关科室的号", ""], k=1)[0]
    #         prefix += "    - 帮我挂[{0}](disease){1}\n".format(line.strip("\n"), posttext)

    file_saved.write(prefix)


with open("data/symptoms.pkl", "rb") as file:
    symptoms_set = pickle.load(file)

if not os.path.exists("data/symptom_disease_stats.pkl"):
    symptom_disease_stats = collections.defaultdict(int)
    for symptom in tqdm(symptoms_set, desc="[INFO] 正在统计每个症状的关联疾病数"):
        symptom_disease_stats[symptom] = local_neo4jdb.run(
            """
            MATCH (m:Disease)-[:has_main_symptom | has_symptom]->(:Symptom{name:"%s"}) 
            RETURN COUNT(DISTINCT m) AS related_diseases_count
            """ % symptom
        ).data()[0]["related_diseases_count"]

    with open("data/symptom_disease_stats.pkl", "wb") as file:
        pickle.dump(symptom_disease_stats, file)
else:
    with open("data/symptom_disease_stats.pkl", "rb") as file:
        symptom_disease_stats = pickle.load(file)

df = pd.DataFrame(data=[symptom_disease_stats.keys(), symptom_disease_stats.values()]).T
df.columns = ["symptoms", "related_diseases_number"]
df = df[df["related_diseases_number"] > 10]
df = df.sort_values(by="related_diseases_number", ascending=False)

df_omaha = pd.read_csv("data/omaha术语.csv")
df_omaha_symptom = df_omaha[df_omaha["semanticTag"] == "T001"]

df["most_sim_words"] = None
# df["most_sim_score"] = None
df["colloquial_desc"] = None
for i, symptom in tqdm(enumerate(df["symptoms"]), desc="[INFO] 在OMAHA术语中搜索最相近的症状列表"):
    related_term_list = df_omaha_symptom[df_omaha_symptom["term"].str.contains(symptom)]["term"].to_list()
    sim_terms_dict = search_sim_terms(symptom, related_term_list)
    if sim_terms_dict:
        most_sim_term, score = sorted(sim_terms_dict.items(), key=lambda x: x[1], reverse=True)[0]  # 降序
        if score == 1:
            df["most_sim_words"].iloc[i] = most_sim_term
            tmp = df_omaha_symptom[df_omaha_symptom["term"] == most_sim_term]["sim_words"].values[0].replace("\\N", "")
            if tmp:
                df["colloquial_desc"].iloc[i] = tmp.split("##")

# df保存图数据库中能对应上OMAHA术语表的症状及对应的口语化描述列表
df = df.dropna()
if not os.path.exists("data/症状相关疾病数量.csv"):
    df.to_csv("症状相关疾病数量.csv", index=False, encoding="utf-8-sig")

# 根据df创建nlu诊断意图标注数据
disease_diagnosis_intent_samples = ""
prefix = "    - "
for symptom, colloquial_desc_list in df.get(["symptoms", "colloquial_desc"]).values:
    for desc in colloquial_desc_list.strip("[]").replace("'", "").replace(" ", "").split(","):
        disease_diagnosis_intent_samples += "{0}[{1}]{{'entity':'symptom', 'value':'{2}'}}\n".format(prefix, desc,
                                                                                                     symptom)

# 查找neo4j数据库中疾病、症状和科室的实体名称，并做去重处理，保存在txt文件作为RASA的分词器自定义词典
terms_dict = collections.defaultdict(list)
res = local_neo4jdb.run(
    """MATCH (n) WHERE n:Disease OR n:Symptom OR n:Department RETURN DISTINCT n.name AS DiseSympDepTerms"""
).data()
for term in res:
    terms_dict["DiseSympDepTerms"].append(term.get("DiseSympDepTerms"))

with open("nlu/jieba_selfdefined_terms.txt", "w", encoding="utf-8") as file:
    for term in terms_dict["DiseSympDepTerms"]:
        file.write("{0}\n".format(term))

# 构建nlu.yml中的synonym样例
nlu_synonym_examples = ""
prefix = """- synonym: 疲劳
  examples: |"""
df = pd.read_csv("data/症状相关疾病数量.csv")
for ind, row in df.iterrows():
    prefix = """- synonym: {0}
  examples: |\n""".format(row["symptoms"])
    synonyms = row["colloquial_desc"].strip("[]").replace("'", "").replace(" ", "").split(",")
    for synonym in synonyms:
        prefix += "    - {0}\n".format(synonym)
    nlu_synonym_examples += prefix + "\n"

with open("synonyms_for_nluyml.txt", "w", encoding="utf-8") as file:
    file.write(nlu_synonym_examples)


# 构建nlu.yml中的lookup表样例，分别构建症状，疾病和科室的lookup列表
local_neo4jdb = Graph(host="localhost",  # neo4j 搭载服务器的ip地址，ifconfig可获取到
                      port=7687,  # neo4j 服务器监听的端口号
                      user="neo4j",
                      password="123456"
                      )
disease = local_neo4jdb.run(
    """MATCH (n:Disease) RETURN COLLECT(DISTINCT n.name) AS Disease"""
).data()[0]["Disease"]

symptom = local_neo4jdb.run(
    """MATCH (n:Symptom) RETURN COLLECT(DISTINCT n.name) AS Symptom"""
).data()[0]["Symptom"]

department = local_neo4jdb.run(
    """MATCH (n:Department) RETURN COLLECT(DISTINCT n.name) AS Department"""
).data()[0]["Department"]

for dise in disease:
    if dise in symptom:
        local_neo4jdb.run("""MATCH (n:Disease {{name:"{0}"}}) DETACH DELETE n""".format(dise))
        disease.remove(dise)

with open("nlu/data/nlu_lookup_table_diseases.yml", "w", encoding="utf-8") as file:
        prefix = """version: "3.0"
nlu:
  - lookup: disease
    examples: |
"""
        for term in disease:
            prefix += "      - {0}\n".format(term)
        file.write(prefix)

with open("nlu/data/nlu_lookup_table_symptoms.yml", "w", encoding="utf-8") as file:
        prefix = """version: "3.0"
nlu:
  - lookup: symptom
    examples: |
"""
        for term in symptom:
            prefix += "      - {0}\n".format(term)
        file.write(prefix)

with open("nlu/data/nlu_lookup_table_departments.yml", "w", encoding="utf-8") as file:
        prefix = """version: "3.0"
nlu:
  - lookup: department
    examples: |
"""
        for term in department:
            prefix += "      - {0}\n".format(term)
        file.write(prefix)

# 删除症状lookup表中，某个症状的分词都属于症状表中，比如“头痛头晕”，jieba分词为“头痛”，“头晕”都属于标准症状术语，此时应该删除“头痛头晕”，降低术语粒度
with open("nlu/data/nlu_lookup_table_symptoms.yml", "r", encoding="utf-8") as file:
    standard_symptoms = yaml.load(file.read(), Loader=yaml.FullLoader)
    standard_symptoms = standard_symptoms["nlu"][0]["examples"].strip().replace("- ", "").split("\n")
import jieba
for sym in standard_symptoms:
    notall = False
    tmp = list(jieba.cut(sym))
    if len(tmp) == 1:
        continue
    for c in tmp:
        if c not in standard_symptoms:
            notall = True
            break
    if not notall:
        print(sym)
        # 删除数据库中的该症状以及相关关系
        local_neo4jdb.run("""MATCH (n:Symptom {{name:"{0}"}}) DETACH DELETE n""".format(sym))
        # 手动删除症状lookup表和jieba词典
        pass






# import pkuseg
# # 细领域分词（如果用户明确分词领域，推荐使用细领域模型分词）
# seg = pkuseg.pkuseg(model_name='medicine')  # 程序会自动下载所对应的细领域模型
# text = seg.cut('比赛整理近万条真实语境下疫情相关的肺炎、支原体肺炎、支气管炎、上呼吸道感染、肺结核、哮喘、胸膜炎、肺气肿、感冒、咳血等患者提问句对，要求选手通过自然语言处理技术识别相似的患者问题。')              # 进行分词
# print(text)
