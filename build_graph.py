# Date    : 2024/6/2 21:28
# File    : build_graph.py
# Desc    : 
# Author  : Damon
# E-mail  : bingzhenli@hotmail.com


import os
import json
import pickle
import collections
from tqdm import tqdm

import pandas as pd
from py2neo import Graph, Node
from utils import colorstr


# AI开放平台知识图谱对应MedicalKG的关系类型映射表
REL_MAP = {"examination suggest": "need_check",
           "has symptom": "has_symptom",
           "has symptom_main": "has_main_symptom",
           "medication suggest": "use_drug"}

# 关系类型的中文名称
REL_CHN_MAP = {"need_check": "诊断检查",
               "has_symptom": "有症状",
               "has_main_symptom": "有主症",
               "use_drug": "可用药",
               "belong_to": "所属科室"}

DISEASES_ENTITY_LIST_PTH = "data/diseases.pkl"
SYMPTOMS_ENTITY_LIST_PTH = "data/symptoms.pkl"
EXAMINATIONS_ENTITY_LIST_PTH = "data/examinations.pkl"
DRUGS_ENTITY_LIST_PTH = "data/drugs.pkl"
DEPARTMENTS_ENTITY_LIST_PTH = "data/departments.pkl"
RELATIONSHIPS_SAVED_PATH = "data/relationships.pkl"


class MedicalGraph:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.data_path = os.path.join(cur_dir, 'data/medical.json')
        self.g = Graph(
            host="localhost",  # neo4j 搭载服务器的ip地址，ifconfig可获取到
            port=7687,  # neo4j 服务器监听的端口号
            user="medicalkg")  # 数据库user name，如果没有更改过，应该是neo4j
            # password="123456")

    '''读取文件'''
    def read_nodes(self):
        # 共７类节点
        drugs = [] # 药品
        foods = [] #　食物
        checks = [] # 检查
        departments = [] #科室
        producers = [] #药品大类
        diseases = [] #疾病
        symptoms = []#症状

        disease_infos = []#疾病信息

        # 构建节点实体关系
        rels_department = [] #　科室－科室关系
        rels_noteat = [] # 疾病－忌吃食物关系
        rels_doeat = [] # 疾病－宜吃食物关系
        rels_recommandeat = [] # 疾病－推荐吃食物关系
        rels_commonddrug = [] # 疾病－通用药品关系
        rels_recommanddrug = [] # 疾病－热门药品关系
        rels_check = [] # 疾病－检查关系
        rels_drug_producer = [] # 厂商－药物关系

        rels_symptom = [] #疾病症状关系
        rels_acompany = [] # 疾病并发关系
        rels_category = [] #　疾病与科室之间的关系


        count = 0
        for data in open(self.data_path, encoding='utf-8'):
            disease_dict = {}
            count += 1
            print(count)
            data_json = json.loads(data)
            disease = data_json['name']
            disease_dict['name'] = disease
            diseases.append(disease)
            disease_dict['desc'] = ''
            disease_dict['prevent'] = ''
            disease_dict['cause'] = ''
            disease_dict['easy_get'] = ''
            disease_dict['cure_department'] = ''
            disease_dict['cure_way'] = ''
            disease_dict['cure_lasttime'] = ''
            disease_dict['symptom'] = ''
            disease_dict['cured_prob'] = ''

            if 'symptom' in data_json:
                symptoms += data_json['symptom']
                for symptom in data_json['symptom']:
                    rels_symptom.append([disease, symptom])

            if 'acompany' in data_json:
                for acompany in data_json['acompany']:
                    rels_acompany.append([disease, acompany])

            if 'desc' in data_json:
                disease_dict['desc'] = data_json['desc']

            if 'prevent' in data_json:
                disease_dict['prevent'] = data_json['prevent']

            if 'cause' in data_json:
                disease_dict['cause'] = data_json['cause']

            if 'get_prob' in data_json:
                disease_dict['get_prob'] = data_json['get_prob']

            if 'easy_get' in data_json:
                disease_dict['easy_get'] = data_json['easy_get']

            if 'cure_department' in data_json:
                cure_department = data_json['cure_department']
                if len(cure_department) == 1:
                     rels_category.append([disease, cure_department[0]])
                if len(cure_department) == 2:
                    big = cure_department[0]
                    small = cure_department[1]
                    rels_department.append([small, big])
                    rels_category.append([disease, small])

                disease_dict['cure_department'] = cure_department
                departments += cure_department

            if 'cure_way' in data_json:
                disease_dict['cure_way'] = data_json['cure_way']

            if  'cure_lasttime' in data_json:
                disease_dict['cure_lasttime'] = data_json['cure_lasttime']

            if 'cured_prob' in data_json:
                disease_dict['cured_prob'] = data_json['cured_prob']

            if 'common_drug' in data_json:
                common_drug = data_json['common_drug']
                for drug in common_drug:
                    rels_commonddrug.append([disease, drug])
                drugs += common_drug

            if 'recommand_drug' in data_json:
                recommand_drug = data_json['recommand_drug']
                drugs += recommand_drug
                for drug in recommand_drug:
                    rels_recommanddrug.append([disease, drug])

            if 'not_eat' in data_json:
                not_eat = data_json['not_eat']
                for _not in not_eat:
                    rels_noteat.append([disease, _not])

                foods += not_eat
                do_eat = data_json['do_eat']
                for _do in do_eat:
                    rels_doeat.append([disease, _do])

                foods += do_eat
                recommand_eat = data_json['recommand_eat']

                for _recommand in recommand_eat:
                    rels_recommandeat.append([disease, _recommand])
                foods += recommand_eat

            if 'check' in data_json:
                check = data_json['check']
                for _check in check:
                    rels_check.append([disease, _check])
                checks += check
            if 'drug_detail' in data_json:
                drug_detail = data_json['drug_detail']
                producer = [i.split('(')[0] for i in drug_detail]
                rels_drug_producer += [[i.split('(')[0], i.split('(')[-1].replace(')', '')] for i in drug_detail]
                producers += producer
            disease_infos.append(disease_dict)
        return set(drugs), set(foods), set(checks), set(departments), set(producers), set(symptoms), set(diseases), disease_infos,\
               rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commonddrug, rels_drug_producer, rels_recommanddrug,\
               rels_symptom, rels_acompany, rels_category

    '''建立节点'''
    def create_node(self, label, nodes):
        count = 0
        for node_name in nodes:
            node = Node(label, name=node_name)
            self.g.create(node)
            count += 1
            print(count, len(nodes))
        return

    '''创建知识图谱中心疾病的节点'''
    def create_diseases_nodes(self, disease_infos):
        count = 0
        for disease_dict in disease_infos:
            node = Node("Disease", name=disease_dict['name'], desc=disease_dict['desc'],
                        prevent=disease_dict['prevent'] ,cause=disease_dict['cause'],
                        easy_get=disease_dict['easy_get'],cure_lasttime=disease_dict['cure_lasttime'],
                        cure_department=disease_dict['cure_department']
                        ,cure_way=disease_dict['cure_way'] , cured_prob=disease_dict['cured_prob'])
            self.g.create(node)
            count += 1
            print(count)
        return

    '''创建知识图谱实体节点类型schema'''
    def create_graphnodes(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos,rels_check, rels_recommandeat, \
        rels_noteat, rels_doeat, rels_department, rels_commonddrug, rels_drug_producer, rels_recommanddrug,rels_symptom, \
        rels_acompany, rels_category = self.read_nodes()
        self.create_diseases_nodes(disease_infos)
        self.create_node('Drug', Drugs)
        print(len(Drugs))
        self.create_node('Food', Foods)
        print(len(Foods))
        self.create_node('Check', Checks)
        print(len(Checks))
        self.create_node('Department', Departments)
        print(len(Departments))
        self.create_node('Producer', Producers)
        print(len(Producers))
        self.create_node('Symptom', Symptoms)
        return


    '''创建实体关系边'''
    def create_graphrels(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commonddrug, rels_drug_producer, rels_recommanddrug,rels_symptom, rels_acompany, rels_category = self.read_nodes()
        self.create_relationship('Disease', 'Food', rels_recommandeat, 'recommand_eat', '推荐食谱')
        self.create_relationship('Disease', 'Food', rels_noteat, 'no_eat', '忌吃')
        self.create_relationship('Disease', 'Food', rels_doeat, 'do_eat', '宜吃')
        self.create_relationship('Department', 'Department', rels_department, 'belongs_to', '属于')
        self.create_relationship('Disease', 'Drug', rels_commonddrug, 'common_drug', '常用药品')
        self.create_relationship('Producer', 'Drug', rels_drug_producer, 'drugs_of', '生产药品')
        self.create_relationship('Disease', 'Drug', rels_recommanddrug, 'recommand_drug', '好评药品')
        self.create_relationship('Disease', 'Check', rels_check, 'need_check', '诊断检查')
        self.create_relationship('Disease', 'Symptom', rels_symptom, 'has_symptom', '症状')
        self.create_relationship('Disease', 'Disease', rels_acompany, 'acompany_with', '并发症')
        self.create_relationship('Disease', 'Department', rels_category, 'belongs_to', '所属科室')

    '''创建实体关联边'''
    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        count = 0
        # 去重处理
        set_edges = []
        for edge in edges:
            set_edges.append('###'.join(edge))
        all = len(set(set_edges))
        for edge in set(set_edges):
            edge = edge.split('###')
            p = edge[0]
            q = edge[1]
            query = "match(p:%s),(q:%s) where p.name='%s'and q.name='%s' create (p)-[rel:%s{name:'%s'}]->(q)" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.g.run(query)
                count += 1
                print(rel_type, count, all)
            except Exception as e:
                print(e)
        return

    '''导出数据'''
    def export_data(self):
        Drugs, Foods, Checks, Departments, Producers, Symptoms, Diseases, disease_infos, rels_check, rels_recommandeat, rels_noteat, rels_doeat, rels_department, rels_commonddrug, rels_drug_producer, rels_recommanddrug, rels_symptom, rels_acompany, rels_category = self.read_nodes()
        f_drug = open('drug.txt', 'w+')
        f_food = open('food.txt', 'w+')
        f_check = open('check.txt', 'w+')
        f_department = open('department.txt', 'w+')
        f_producer = open('producer.txt', 'w+')
        f_symptom = open('symptoms.txt', 'w+')
        f_disease = open('disease.txt', 'w+')

        f_drug.write('\n'.join(list(Drugs)))
        f_food.write('\n'.join(list(Foods)))
        f_check.write('\n'.join(list(Checks)))
        f_department.write('\n'.join(list(Departments)))
        f_producer.write('\n'.join(list(Producers)))
        f_symptom.write('\n'.join(list(Symptoms)))
        f_disease.write('\n'.join(list(Diseases)))

        f_drug.close()
        f_food.close()
        f_check.close()
        f_department.close()
        f_producer.close()
        f_symptom.close()
        f_disease.close()

        return


class MedicalKG:

    def __init__(self, ):
        self.local_neo4jdb = Graph(
            host="localhost",  # neo4j 搭载服务器的ip地址，ifconfig可获取到
            port=7687,  # neo4j 服务器监听的端口号
            user="neo4j",
            password="123456"
        )

    def create_nodes(self, label, nodes):
        for node_name in tqdm(
                nodes,
                desc="[INFO] creating {0} nodes: ".format(colorstr(label, color="red", bold=True))
        ):
            node = Node(label, name=node_name)
            self.local_neo4jdb.create(node)

    def create_relationship(self, start_node, end_node, edges, rel_type, rel_name):
        # 去重处理
        edges_set = set()
        for edge in edges:
            edges_set.add('###'.join(edge))

        for edge in tqdm(
                edges_set,
                desc="[INFO] creating {0}({1}) relationship: ".format(
                    colorstr(rel_name, color="red", bold=True),
                    colorstr(rel_type, color="red", bold=True)
                )
        ):
            edge = edge.split('###')
            p, q = edge[0], edge[1]
            query = """MATCH (p:%s), (q:%s) WHERE p.name="%s" AND q.name="%s" CREATE (p)-[REL:%s{name:"%s"}]->(q)""" % (
                start_node, end_node, p, q, rel_type, rel_name)
            try:
                self.local_neo4jdb.run(query)
            except Exception as e:
                print(e)

    def create_graph(self, data_pth_list=None):
        diseases_set, symptoms_seth, exam_set, drug_set, department_set = set(), set(), set(), set(), set()
        rel_data_dict = collections.defaultdict(list)  # 存储所有关系列表数据

        if not os.path.exists(DISEASES_ENTITY_LIST_PTH) or not os.path.exists(SYMPTOMS_ENTITY_LIST_PTH) \
                or not os.path.exists(EXAMINATIONS_ENTITY_LIST_PTH) or not os.path.exists(DRUGS_ENTITY_LIST_PTH) \
                or not os.path.exists(DEPARTMENTS_ENTITY_LIST_PTH) or not os.path.exists(RELATIONSHIPS_SAVED_PATH):
            self.knowledge_base = Graph(
                "http://192.168.2.20:7474/browser/", user="neo4j", password="neo4j"
            )  # 知识库

            # 如果上述文件任何一个不存在，重新运行下面代码生成pkl
            for pth in data_pth_list:
                if not os.path.exists(pth):
                    raise IOError("File: '{}' does not exist".format(pth))

                suffix = pth.split(".")[-1]  # 文件路径后缀
                if suffix == "xlsx":
                    data = pd.read_excel(pth)

                    # d_name和e_name列表都包含疾病和症状，只是d_name列表大多数是疾病，e_name大多数是症状
                    ename_ecode_pd = data[(data.relation == "has symptom") | (data.relation == "has symptom_main")]\
                        .get(["e_name", "e_code"])\
                        .dropna()\
                        .astype(str)\
                        .drop_duplicates()
                    ename_ecode_pd.rename(columns={"e_name": "name", "e_code": "code"}, inplace=True)

                    dname_dcode_pd = data\
                        .get(["d_name", "d_code"])\
                        .dropna()\
                        .astype(str)\
                        .drop_duplicates()
                    dname_dcode_pd.rename(columns={"d_name": "name", "d_code": "code"}, inplace=True)

                    name_code_pd = pd.concat([ename_ecode_pd, dname_dcode_pd]).drop_duplicates()

                    diseases_set, symptoms_set = set(), set()
                    for name, code in tqdm(name_code_pd.values, desc="[INFO] 正在查询知识库确认实体属于疾病/症状: "):
                        if "|" not in code:
                            query_data = self.knowledge_base.run(
                                """
                                MATCH (n) WHERE n.conceptId = '{0}' WITH COLLECT(DISTINCT LABELS(n)) AS tmp 
                                RETURN [ANY(x IN tmp WHERE 'DISORDER' IN x), ANY(x IN tmp WHERE 'FINDING' IN x)] 
                                AS isDiseOrSymp
                                """.format(code)
                            ).data()[0]["isDiseOrSymp"]
                            if query_data[0]:
                                # 有DISORDER的属于疾病
                                diseases_set.add(name)
                            elif query_data[1]:
                                # 有FINDING的属于疾病
                                symptoms_set.add(name)
                            else:
                                # 既没有DISORDER，也没有FINDING的属于疾病
                                diseases_set.add(name)

                    # 检查项目列表（已去重）
                    exam_set = set(data[(data.relation == "examination suggest")]["e_name"])

                    # 药品列表（已去重）
                    drug_set = set(data[(data.relation == "medication suggest")]["e_name"])

                    # 保存数据data/graph_20211201.xlsx中的关系列表到变量rel_data_dict
                    for rel in set(data["relation"]):
                        _rel_data = data[data["relation"] == rel]

                        # e.g.: _key_name = 'Disease-has_main_symptom-Symptom'
                        if rel == "has symptom_main" or rel == "has symptom":
                            _key_name = "{0}-{1}-{2}".format("Disease", REL_MAP[rel], "Symptom")
                        elif rel == "medication suggest":
                            _key_name = "{0}-{1}-{2}".format("Disease", REL_MAP[rel], "Drug")
                        else:  # rel == "examination suggest"
                            _key_name = "{0}-{1}-{2}".format("Disease", REL_MAP[rel], "Examination")

                        rel_data_dict[_key_name] += [_ for _ in zip(_rel_data["d_name"], _rel_data["e_name"])]

                elif suffix == "json":
                    for _data_dict in open(pth, encoding="utf-8"):
                        _data_json = json.loads(_data_dict)
                        diseases_set.add(_data_json["name"])

                        if "symptom" in _data_json:
                            for sym in _data_json["symptom"]:
                                symptoms_set.add(sym)
                                rel_data_dict["Disease-has_symptom-Symptom"].append((_data_json["name"], sym))

                        if "recommand_drug" in _data_json:
                            for drug in _data_json['recommand_drug']:
                                drug_set.add(drug)
                                rel_data_dict["Disease-use_drug-Drug"].append((_data_json["name"], drug))

                        if "check" in _data_json:
                            for exam in _data_json['check']:
                                exam_set.add(exam)
                                rel_data_dict["Disease-need_check-Examination"].append((_data_json["name"], exam))

                        if "cure_department" in _data_json:
                            _small_department_category = _data_json['cure_department'][-1] # 科室细分类，不管大类，方便查询医生挂号
                            department_set.add(_small_department_category)
                            rel_data_dict["Disease-belong_to-Department"].append((_data_json["name"],
                                                                                  _small_department_category))

            # 保存疾病实体
            with open(DISEASES_ENTITY_LIST_PTH, 'wb') as file:
                pickle.dump(diseases_set, file)

            # 保存症状实体
            with open(SYMPTOMS_ENTITY_LIST_PTH, 'wb') as file:
                pickle.dump(symptoms_set, file)

            # 保存检查实体
            with open(EXAMINATIONS_ENTITY_LIST_PTH, 'wb') as file:
                pickle.dump(exam_set, file)

            # 保存药物实体
            with open(DRUGS_ENTITY_LIST_PTH, 'wb') as file:
                pickle.dump(drug_set, file)

            # 保存部门实体
            with open(DEPARTMENTS_ENTITY_LIST_PTH, 'wb') as file:
                pickle.dump(department_set, file)

            # 保存所有关系
            with open(RELATIONSHIPS_SAVED_PATH, 'wb') as file:
                pickle.dump(rel_data_dict, file)

        else:
            # 读取疾病实体
            with open(DISEASES_ENTITY_LIST_PTH, 'rb') as file:
                diseases_set = pickle.load(file)

            # 读取症状实体
            with open(SYMPTOMS_ENTITY_LIST_PTH, 'rb') as file:
                symptoms_set = pickle.load(file)

            # 读取检查实体
            with open(EXAMINATIONS_ENTITY_LIST_PTH, 'rb') as file:
                exam_set = pickle.load(file)

            # 读取药物实体
            with open(DRUGS_ENTITY_LIST_PTH, 'rb') as file:
                drug_set = pickle.load(file)

            # 读取部门实体
            with open(DEPARTMENTS_ENTITY_LIST_PTH, 'rb') as file:
                department_set = pickle.load(file)

            # 读取所有关系
            with open(RELATIONSHIPS_SAVED_PATH, 'rb') as file:
                rel_data_dict = pickle.load(file)

        print("[INFO] contains {0} diseases, {1} symptoms, {2} examinations, {3} drugs, {4} departments entities"
              .format(colorstr(len(diseases_set), bold=True),
                      colorstr(len(symptoms_set), bold=True),
                      colorstr(len(exam_set), bold=True),
                      colorstr(len(drug_set), bold=True),
                      colorstr(len(department_set), bold=True)))

        # 创建图谱实体节点
        self.create_nodes("Disease", diseases_set)
        self.create_nodes("Symptom", symptoms_set)
        self.create_nodes("Examination", exam_set)
        self.create_nodes("Drug", drug_set)
        self.create_nodes("Department", department_set)

        # 创建实体关系边
        for triple_name, head_tail_list in rel_data_dict.items():
            # head_name: 'Disease', rel_eng_name: 'has_symptom', tail_name: 'Symptom'
            head_name, rel_eng_name, tail_name = triple_name.split("-")
            self.create_relationship(head_name, tail_name, head_tail_list, rel_eng_name, REL_CHN_MAP[rel_eng_name])


def test_build_medical_graph():
    handler = MedicalGraph()
    handler.create_graphnodes()
    handler.create_graphrels()
    # handler.export_data()


def test_merge_graph():
    mkg = MedicalKG()
    mkg.create_graph(["data/graph_20211201.xlsx",
                      "data/medical.json"])
