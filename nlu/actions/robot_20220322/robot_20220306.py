import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import models
import os
from torch.autograd import Variable
import pickle
import pandas as pd
from collections import Counter

# to do
os.environ['CUDA_VISIBLE_DEVICES']='1'

class DecisionMakingNet(nn.Module):
    '''
    初始化网络
    '''
    def __init__(self, input_dimension):
        super(DecisionMakingNet, self).__init__()
        self.linear1 = nn.Linear(input_dimension, 256)
        self.linear2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    '''
    前向计算过程
    '''
    def forward(self, x):
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x

        
'''
建议网络,建议应该询问的下一条症状
'''      
class SuggestionNet(nn.Module):
    '''
    初始化网络
    '''
    def __init__(self, input_dimension, output_dimension):
        super(SuggestionNet, self).__init__()
        self.linear1 = nn.Linear(input_dimension, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, output_dimension)
        self.sigmoid = nn.Sigmoid()

    '''
    前向计算过程
    '''
    def forward(self, x):
        x = F.relu(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x
        
'''
诊断网络,用于问诊过程中和最后动态做出诊断的
'''  
class Diagnose_net():
    
    def __init__(self, g='graph_20211207.csv', contribution_dict_cn='sub_tree_contribution_cn.pkl'):
        with open(contribution_dict_cn, 'rb') as f:
            self.contribution_dict_cn = pickle.load(f)
        
        self.df = pd.read_csv(g)
        self.dfs = self.df[(self.df['relation'] == 'has symptom_main') | (self.df['relation'] == 'has symptom')]
        
        self.dentity2dname = {str(k).strip(): str(v).strip() for k, v in list(zip(list(self.dfs['did']), list(self.dfs['d_name'])))}
        self.eentity2ename = {str(k).strip(): str(v).strip() for k, v in list(zip(list(self.dfs['eid']), list(self.dfs['e_name'])))}

        self.dname2dentity = {v: k for k, v in self.dentity2dname.items()}
        self.ename2eentity = {v: k for k, v in self.eentity2ename.items()}

        self.ds_pair = list(zip(list(self.dfs['d_name']), list(self.dfs['e_name']), list(self.dfs['relation'])))

        self.d2s = {}
        self.s2d = {}
        self.s2d_w = {}
        for i1, i2, i3 in self.ds_pair:
            i1 = i1.strip()
            i2 = i2.strip()

            if i1 not in self.d2s:
                self.d2s[i1] = set()
            self.d2s[i1].add(i2)

            if i2 not in self.s2d:
                self.s2d[i2] = set()
            self.s2d[i2].add(i1)

            if i2 not in self.s2d_w:
                self.s2d_w[i2] = dict()

            self.weight_set = 3
            self.weight_normal = 1
            self.weight_default = 1
            if i3.strip() == 'has symptom':
                self.s2d_w[i2][i1] = self.weight_normal
            elif i3.strip() == 'has symptom_main':
                self.s2d_w[i2][i1] = self.weight_set
            else:
                self.s2d_w[i2][i1] = self.weight_default


        self.d2s_c = {d: len(sset) for d, sset in self.d2s.items()}
        self.s2d_c = {s: len(dset) for s, dset in self.s2d.items()}

        self.d2did = {}
        for d, _ in self.d2s.items():
            if d not in self.d2did:
                self.d2did[d] = len(self.d2did)

        self.did2d = {v: k for k, v in self.d2did.items()}

        self.s2sid = {}
        for s, _ in self.s2d.items():
            if s not in self.s2sid:
                self.s2sid[s] = len(self.s2sid)

        self.sid2s = {v: k for k, v in self.s2sid.items()}

        self.d_info = np.zeros(len(self.d2s))
        for d, c in self.d2s_c.items():
            did = self.d2did[d]
            self.d_info[did] = 1 / c
            
        f = open('../../data/dname_values_new.pkl', 'rb')
        #dname_values_50 = pickle.load(f)
        dname_values_mean = pickle.load(f)
        f.close()
        '''
        self.d_freq_50 = np.zeros(len(self.d2s))
        for d, c in self.d2s_c.items():
            did = self.d2did[d]
            self.d_freq_50[did] = dname_values_50[d]
        '''
            
        self.d_freq_mean = np.zeros(len(self.d2s))
        for d, c in self.d2s_c.items():
            did = self.d2did[d]
            self.d_freq_mean[did] = dname_values_mean[d]
            
    def predict(self, eentity_list):
        symptom_list_original = [self.eentity2ename[eentity] for eentity in eentity_list]
#         symptom_list_original = ['胸闷', '心悸', '晚咳']
        #symptom_list_extension = []
        #for so in symptom_list_original:
        #    if so in self.contribution_dict_cn:
        #        symptom_list_extension.extend(self.contribution_dict_cn[so])
        #    else:
        #        symptom_list_extension.extend([so])
        symptom_list_extension = symptom_list_original
        symptom_list_extension = [_item for _item in symptom_list_extension if _item in self.ename2eentity]
        symptom_list = list(set(symptom_list_extension))

        disease_weight = np.zeros(len(self.d2s))

        for symptom in symptom_list:

            s_info = np.zeros(len(self.d2s))
            
#             if symptom in self.s2d 
#             for d in self.s2d[symptom]：
                
            dids = [self.d2did[d] for d in self.s2d[symptom] if symptom in self.s2d]
            for did in dids:
                s_info[did] = 1 / self.s2d_c[symptom]

            w_info = np.zeros(len(self.d2s))
            dws = self.s2d_w[symptom]
            for d, w in dws.items():
                did = self.d2did[d]
                w_info[did] = w

            disease_weight += (s_info * self.d_freq_mean * self.d_info * w_info)


        disease_weight_normalized = (disease_weight ** 2) / np.sum(disease_weight ** 2)

        indexs = [i for i in range(len(disease_weight_normalized))]
        disease_weight_index = list(zip(indexs, disease_weight_normalized))

        disease_weight_index.sort(key=lambda k: k[1])
        disease_weight_index.reverse()
        
        disease_pred = [(self.dname2dentity[self.did2d[d]], p) for d, p in disease_weight_index]
        return disease_pred
        
'''
机器医生类
'''
class Robot(object):
    def __init__(self):
        super(Robot, self).__init__()
        
        # to do
        # ./benchmarks/DS_DATA_20181221/
        # ./res/kg_test_transe_20181221/
        # ./sid2idx.txt
        # ./decisionMakingNet.pkl
        # ./plan_b_2.pkl
        print('start to load data')
        self.entity2id = {}
        with open('../../data/benchmarks/DS_DATA_20211201/entity2id.txt', 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                arr = line.split('\t')
                if len(arr) == 2:
                    entity_name = arr[0].strip()
                    entity_id = int(arr[1].strip())
                    if entity_name not in self.entity2id:
                        self.entity2id[entity_name] = entity_id
        
        self.relation2id = {}
        with open('../../data/benchmarks/DS_DATA_20211201/relation2id.txt', 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                arr = line.split('\t')
                if len(arr) == 2:
                    relation_name = arr[0].strip()
                    relation_id = int(arr[1].strip())
                    if relation_name not in self.relation2id:
                        self.relation2id[relation_name] = relation_id
        
        self.input_dimension = len(self.entity2id)

        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        with open('../../data/benchmarks/DS_DATA_20211201/triplets.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
            
        # tree is a data structure that mapping from tail to heads
        # it expresses the how many heads point to a specific tail
        self.tree = {}
        for line in lines:
            arr = line.strip().split(',')
            h = arr[0].strip()
            t = arr[1].strip()
            r = arr[2].strip()
            
            #if t.endswith('Condition'):
            hid = self.entity2id[h]
            tid = self.entity2id[t]
            rid = self.relation2id[r]
            
            if hid not in self.tree:
                self.tree[hid] = set()
            self.tree[hid].add((tid, rid))
            
        self.conditions = []
        for line in lines:
            arr = line.strip().split(',')
            h = arr[0].strip()
            t = arr[1].strip()
            self.conditions.append(self.entity2id[h])
                
        self.conditions = list(set(self.conditions))

        self.all_disease = self.conditions.copy()
        self.n_disease = len(self.all_disease)

        self.d2s = {}
        for disease in self.all_disease:
            if disease not in self.d2s:
                self.d2s[disease] = []
            for symptom, _ in self.tree[disease]:
                self.d2s[disease].append(symptom)
                
        self.d2s = {d: list(set(sl)) for d, sl in self.d2s.items()}
        
        #症状对疾病列表
        self.s2d = {}
        for d, slist in self.d2s.items():
            for s in slist:
                if s not in self.s2d:
                    self.s2d[s] = set()
                self.s2d[s].add(d)
        '''
        self.did2dindex = {}
        with open('did2dindex.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                arr = line.strip().split(':')
                did = int(arr[0])
                dindex = int(arr[1])
                if did not in self.did2dindex:
                    self.did2dindex[did] = int(dindex)        
        self.dindex2did = {v: k for k, v in self.did2dindex.items()}
        '''
        
        self.suggestion_id2index = {}
        with open('../../data/suggestion_id2index.txt', 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                arr = line.split(',')
                self.suggestion_id2index[int(arr[0].strip())] = int(arr[1].strip())
        self.index2suggestion_id = {v: k for k,v in self.suggestion_id2index.items()}
        
        self.output_dimension = len(self.index2suggestion_id)
        self.slist = list(set([s for d, sl in self.d2s.items() for s in sl]))
        self.multi_symptom_disease = [did for did in self.all_disease if len(self.d2s[did]) > 1]
        
        #科室
        df_department = pd.read_csv('../../data/汇知医学知识图谱_疾病科室_20210520.txt')
        df_department = df_department[(df_department['entityTag'] == '疾病') & (df_department['property'] == '科室')]
        df_department_new = pd.read_excel('df_unmatched_diseases_lqx&qyh_220321.xlsx')
        disease2department = list(zip(list(df_department['entity']), list(df_department['value'])))
        disease2department.extend(list(zip(list(df_department_new['diseases']), list(df_department_new['疾病科室']))))
        self.disease2department = {k: v for k, v in disease2department}
        
        print('finish loading data')
        print('start to load models')
        
        with open('../../data/model_n_matrix.pkl', 'rb') as f:
            self.model_n_matrix = pickle.load(f)
            self.model_n_matrix = torch.tensor(self.model_n_matrix).cuda()
            self.model_n_matrix.requires_grad = False
        print('finish loading translation model')
        #self.mutant = Mutant(self.con.testModel, self.tree, self.input_dimension)
        self.embedding_size = 150
        
        '''
        self.con = config.Config()
        # specify the path of data set 
        self.con.set_in_path("/root/workspace_dl/openKE_pytorch_new_0521/OpenKE/benchmarks/DS_DATA_20190521/") 
        self.con.set_work_threads(4)
        self.con.set_dimension(100)
        self.con.set_result_dir('/root/workspace_dl/openKE_pytorch_new_0521/OpenKE/result')
        self.con.init()
        self.con.set_test_model(models.TransE)
        print('finish loading translation model')
        self.mutant = Mutant(self.con.testModel, self.tree, self.input_dimension)
        '''
        self.decisionMakingNet = DecisionMakingNet(self.embedding_size).cuda()
        self.decisionMakingNet.load_state_dict(torch.load('decisionMakingNet_mutant_20220305.pkl'))
        self.decisionMakingNet.eval()
        print('finish loading decisionMakingNet')
        
        '''
        self.decisionMakingNet = DecisionMakingNet(self.embedding_size).cuda()
        self.decisionMakingNet.load_state_dict(torch.load('decisionMakingNet_mutant.pkl'))
        self.decisionMakingNet.eval()
        print('finish loading decisionMakingNet')
        '''
        
        self.suggestionNet = SuggestionNet(self.embedding_size, self.output_dimension).cuda()
        self.suggestionNet.load_state_dict(torch.load('suggestion_net_mutant_20220305.pkl'))
        self.suggestionNet.eval()
        print('finish loading suggestionNet')
        
        '''
        self.suggestionNet = SuggestionNet(self.embedding_size, self.output_dimension).cuda()
        self.suggestionNet.load_state_dict(torch.load('suggestion_net_mutant.pkl'))
        self.suggestionNet.eval()
        print('finish loading suggestionNet')
        '''
        
        self.diagnose_net = Diagnose_net()
        print('finish loading diagnoseNet')
        
        '''
        self.diagnoseNet = DiagnoseNet(self.n_disease, self.embedding_size).cuda()
        self.diagnoseNet.load_state_dict(torch.load('Diagnose_net_0521_with_mutant.pkl'))
        self.diagnoseNet.eval()
        print('finish loading diagnoseNet')
        '''
        
        '''
        self.name2entityid = {int(e.split('_')[0]): eid for e, eid in self.entity2id.items()}
        self.entityid2name = {v: k for k, v in self.name2entityid.items()}
        '''
        self.cname2id = {cname: self.entity2id[entity] for cname, entity in self.diagnose_net.ename2eentity.items()}
        self.id2cname = {v: k for k, v in self.cname2id.items()}
        '''
        self.id2cname = {}
        with open('id2cname.txt', 'r', encoding='utf8') as f:
            for line in f:
                if len(line.strip()) != 0:
                    arr = line.split(':')
                    if len(arr) == 2:
                        self.id2cname[int(arr[0].strip())] = arr[1].strip()
        self.cname2id = {v: k for k, v in self.id2cname.items()}
        '''
        '''
        self.single_s_d = [d for d, slist in self.d2s.items() if len(slist) == 1]
        self.single_s = {slist[0]: d for d, slist in self.d2s.items() if len(slist) == 1}
        '''
        
        #used for interaction
        self.suggestion_list = []
        self.suggestion_prob_list = []
        
        #for filtering the suggestion nodes
        '''
        self.node_set = {'Condition',
             'Condition_group',
             'Symptom',
             'Symptom_group',
             'Symptom_sub'}
        '''
        
        ##########################################################################################################
        '''
        #全量数据集的d和对应的sympton列表
        self.d2s = {}

        # train2id 所有关系为1的数据
        with open('./benchmarks/DS_DATA_20181221/train2id.txt') as f:
            for line in f:
                arr = line.strip().split('\t')
                if len(arr) == 3:
                    if arr[2].strip() == '1':
                        s = arr[0].strip()
                        d = arr[1].strip()
                        if d not in self.d2s:
                            self.d2s[d] = []
                        self.d2s[d].append(s)
                        
        # test2id 所有关系为1的数据
        with open('./benchmarks/DS_DATA_20181221/test2id.txt') as f:
            for line in f:
                arr = line.strip().split('\t')
                if len(arr) == 3:
                    if arr[2].strip() == '1':
                        s = arr[0].strip()
                        d = arr[1].strip()
                        if d not in self.d2s:
                            self.d2s[d] = []
                        self.d2s[d].append(s)

        slist = list(set([s for d, sl in self.d2s.items() for s in sl]))

        ## symptom id要转换成action维度(929)
        with open('sid2idx.txt', 'r', encoding='utf8') as f:
            self.sid2idx = {}
            for line in f:
                if len(line.strip()) != 0:
                    arr = line.strip().split(':')
                    self.sid2idx[arr[0].strip()] = int(arr[1].strip())
        self.idx2sid = {v: k for k, v in self.sid2idx.items()}
        self.symptom_size = len(slist)
        #print(symptom_size)
        #dids = list(set([d for d, _ in self.d2s.items()]))
        
        #症状对疾病列表
        self.s2d = {}
        for d, slist in self.d2s.items():
            for s in slist:
                if s not in self.s2d:
                    self.s2d[s] = set()
                self.s2d[s].add(d)
        #疾病数量
        self.disease_size = len(self.d2s)
        #疾病id与index的互相转换
        self.did2idx = {}
        dl = [d for d, _ in self.d2s.items()]
        dl.sort(key=lambda k: k)
        for did in dl:
            if did not in self.did2idx:
                self.did2idx[did] = len(self.did2idx)
        self.idx2did = {v: k for k, v in self.did2idx.items()}
    
        self.single_s_d = [d for d, slist in self.d2s.items() if len(slist) == 1]
        self.single_s = {slist[0]: d for d, slist in self.d2s.items() if len(slist) == 1}
        
        #新旧模型融合需要用到
        self.entity2id = {}
        with open('./benchmarks/DS_DATA_20190322/entity2id.txt') as f:
            for line in f:
                arr = line.strip().split('\t')
                if len(arr) == 2:
                    entity = arr[0].strip()
                    id = arr[1].strip()
                    if entity not in self.entity2id:
                        self.entity2id[entity] = id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.input_dimension = len(self.id2entity)
        
        self.con = config.Config()
        # specify the path of data set 
        self.con.set_in_path("./benchmarks/DS_DATA_20190322/") 
        self.con.set_work_threads(4)
        self.con.set_dimension(100)
        self.con.set_result_dir('./result')
        self.con.init()
        self.con.set_test_model(models.TransE)
        
        self.decisionMakingNet = DecisionMakingNet(100).cuda()
        self.decisionMakingNet.load_state_dict(torch.load('decisionMakingNet_mutant.pkl'))
        self.decisionMakingNet.eval()
        
        # 创建inference tool
        self.inference_tool = Inference_tool(self.d2s)
        
        self.suggestionNet = SuggestionNet(100, self.output_dimension).cuda()
        self.suggestionNet.load_state_dict(torch.load('suggestion_net_mutant.pkl'))
        self.suggestionNet.eval()
        #self.suggestionNet = SuggestionNet(self.symptom_size, 2 * self.symptom_size + self.disease_size).cuda()
        #self.suggestionNet.load_state_dict(torch.load('eval_network_policygradient_v10_2019_03_01.pkl'))
        
        
        
        #sid2sname
        with open('sid2sname.txt', 'r', encoding='utf8') as f:
            self.sid2sname = {}
            for line in f:
                if len(line.strip()) != 0:
                    arr = line.strip().split(':')
                    self.sid2sname[arr[0].strip()] = arr[1].strip()
        #sname2sid            
        self.sname2sid = {v: k for k, v in self.sid2sname.items()}
        
        #did2dname
        with open('did2dname.txt', 'r', encoding='utf8') as f:
            self.did2dname = {}
            for line in f:
                if len(line.strip()) != 0:
                    arr = line.strip().split(':')
                    self.did2dname[arr[0].strip()] = arr[1].strip()
        #dname2did            
        self.dname2did = {v: k for k, v in self.did2dname.items()}
        '''
        
    '''
    重新初始化,为下一个interaction过程做准备,在开始新的一个interaction前调用
    '''
    def reset(self):
        #print('---------restart----------')
        
        self.done = False
        self.start2count = False
        self.observation = None
        #self.single_symptom = False
        self.selected_index = [] #已经选择并且是positive的症状index
        self.selected_set = set([eid for eid in self.selected_index])
        
        self.wrong_index = []
        self.wrong_set = set(self.wrong_index)
        
        self.total_reward = 0
        
        self.max_try_times = 5
        self.try_times = 0
        self.more = 2
        self._i = 0
        
        #used for interaction
        self.suggestion_list = []
        self.suggestion_prob_list = []
        
        #self.current_candidate_disease = None
    
    '''
    用于计算轮数,限定轮数长度
    假如轮数足够可以通过调用本类的is_enough()函数来查看是否应该进入疾病推断环节
    '''
    def increase_times(self):
        self.try_times += 1
        if self.start2count:
            self._i += 1
        if self._i >= self.more:
            self.done = True
            return
        if (self.try_times >= self.max_try_times) or self.is_done(self.selected_index):
            self.start2count = True
    
    '''
    假如patient对于医生提出的症状给出正反馈,则调用
    '''
    def input_sid(self, node):
        # 第一次输入症状
        #if self.observation == None:
        #    if node[0] in self.single_s:
        #        # 正常问问题,假如回答false则继续问,如果回答true则肯定不是单症状疾病
        #        self.single_symptom = True
        #if self.current_candidate_disease is None:
        #    self.current_candidate_disease = self.s2d[sid].copy()
        #else:
        #    self.current_candidate_disease = self.current_candidate_disease & self.s2d[sid]
        
        self.observation = node
        self.selected_index.append((self.observation))
        self.selected_set = set([eid for eid in self.selected_index])
        self.total_reward += 1
        self.increase_times()
    
    '''
    批量输入多个正反馈症状
    '''
    def batch_input_sid(self, nodes):
        for node in nodes:
            self.input_sid(node)
    
    '''
    假如patient对于医生提出的症状给出负反馈,则调用
    一旦输入,则往后都不会在给出想用的症状suggestion
    '''
    def update_wrong_sid(self, nodes):
        for node in nodes:
            self.increase_times()
            self.total_reward -= 1
            if node not in self.wrong_index:
                self.wrong_index.append(node)
        self.wrong_set = set(self.wrong_index)
    
    '''
    对列表元素进行softmax计算,转化为0-1范围
    '''
    def softmax(self, l):
        return list(F.softmax(torch.tensor(l, dtype=torch.float)).numpy())
    
    '''
    校验node是否符合要求
    '''
    def evaluate_node(self, node):
        return True
        '''
        e = self.id2entity[node]
        eoid = e.split('_')[0]
        t = e[e.index(eoid) + len(eoid) + 1:]
        if t in self.node_set: 
            return True
        return False
        '''
    
    '''
    给出suggestions
    '''
    def suggestion(self, num=10):
        # return list of prossible symptoms according to the selected_index
        sindexs, scores = self.predict()
        suggestions = [self.index2suggestion_id[_item] for _item in sindexs]
        ts = list(zip(suggestions, scores))
        # filtered
        ts = [line for line in ts if self.evaluate_node(line[0])]
        ts = [line for line in ts if (line[0] not in self.selected_set) and (line[0] not in self.wrong_set)][:num]
        #ts = [(self.idx2sid[tuple[0]], tuple[1]) for tuple in ts if (tuple[0] not in self.selected_index) and (tuple[0] not in self.wrong_index)][:num]
        _sids, _scores = zip(*ts)
        _sids = list(_sids)
        _scores = list(_scores)
        #_scores = self.softmax(_scores)
        
        #todo 并没有对概率进行排序
        #_sids = self.reorder(_sids)
        
        #used for interaction
        self.suggestion_list = _sids.copy()
        return _sids, _scores
    
    '''
    用于判断当前症状是否足够
    '''
    def is_enough(self):
        return self.done
    
    '''
    推断疾病
    '''
    '''
    def inference(self, num=5):
        # return list of possible diseases
        if self.single_symptom and len(self.selected_index) == 1:
            # 单症状疾病
            d = self.single_s[self.idx2sid[self.selected_index[0]]]
            return [d]
        else:
            selected_sid = [self.idx2sid[idx] for idx in self.selected_index]
            dids = self.inference_tool.inference_by_iou(selected_sid, top=num)
            return dids
    '''
    
    '''
    推断疾病并带上概率
    '''    
    def inference_with_probs(self, num=10):
        node_id_list, nodes = self.convert(self.selected_index)
        symptom_entity_list = [self.id2entity[i] for i in node_id_list]
        result_entity_list = self.diagnose_net.predict(symptom_entity_list)
        y = list(zip(*result_entity_list))
        ysort = y[0]
        yprob = y[1]
        yname = [self.diagnose_net.dentity2dname[_item] for _item in ysort]
        return yname[:num], yprob[:num]
    
    '''
    推荐科室
    '''
    def recommend_departments(self):
        recommend_departments = [self.disease2department[d] for d in self.inference_disease_with_probs()[0]]
        recommend_departments = [(k, v) for k, v in dict(Counter(recommend_departments)).items()]
        recommend_departments.sort(key=lambda k: k[1])
        recommend_departments.reverse()
        recommend_departments = list(zip(*recommend_departments))[0]
        return recommend_departments
        
    '''
    根据输入的symptom index tensor,获取预测症状的index的倒序排列,置信度从高到底
    '''
    def predict(self):
        with torch.no_grad():
            node_id_list, nodes = self.convert(self.selected_index)
            x = self.combine(nodes)
            y = self.suggestionNet(x)
            scores, indexs = y[0].sort()
            ii = list(indexs.cpu().numpy())
            ss = list(scores.cpu().numpy())
            return [i for i in reversed(ii)], [s for s in reversed(ss)]
            
            '''
            nodes, relations = self.mutant.convert(self.selected_index)
            x = self.mutant.combine(nodes, relations)
            y = self.suggestionNet(x)
            scores, indexs = y[0].sort()
            ii = list(indexs.cpu().numpy())
            ss = list(scores.cpu().numpy())
            return [i for i in reversed(ii)], [s for s in reversed(ss)]
            '''
        
    '''
    根据输入的index转成pytorch的tensor
    '''
    '''
    def sindex2tensor(self, ids):
        x = np.zeros(self.symptom_size)
        for dimension in ids:
            x[dimension] = 1
        return self.to_tensor([x])
    '''
    
    '''
    根据输入的sid转成长度为input_dimension的pytorch的sparse tensor
    '''
    '''
    def sid2tensor(self, sids):
        x = np.zeros(self.input_dimension)
        for sid in sids:
            x[int(sid)] = 1
        return self.to_tensor([x])
    '''
    
    '''
    根据环境返回的observation(symptom的index),用decisionMakingNet判断症状是否足够
    '''
    def is_done(self, selected_index, threshold=0.5):
        with torch.no_grad():
            node_id_list, nodes = self.convert(selected_index)
            x = self.combine(nodes)
            decision_prob = self.decisionMakingNet(x).squeeze(1)[0].cpu().item() 
            if decision_prob >= threshold:
                return True
            else:
                return False
            '''
            nodes, relations = self.mutant.convert(selected_index)
            x = self.mutant.combine(nodes, relations)
            decision_prob = self.decisionMakingNet(x).squeeze(1)[0].cpu().item() 
            if decision_prob >= threshold:
                return True
            else:
                return False
            '''
            
    '''
    根据symptom列表(observation)转换成nn能接受的输入
    '''
    '''
    def to_observation_tensors(self, observation):
        #把index转成id,因为env返回的observation是index
        symptom_list = [self.idx2sid[index] for index in observation]
        #根据symptom id转换成symptom的tensor
        symptom_tensors = [self.get_entity(int(symptom_id)).view(1, 50) for symptom_id in symptom_list]
        #整理成矩阵方便并行计算
        symptom_tensors = torch.stack(symptom_tensors, dim=0)
        symptom_tensors = symptom_tensors.view(1, -1, 50)
        return symptom_tensors
    '''

    '''
    decisionMakingNet需要用到transE的model,后续可能会考虑重新训练一个decisionMakingNet,不需要用到transE
    '''
    '''
    def get_entity(self, id):
        return self.con.testModel.ent_embeddings(Variable(torch.tensor([id])).cuda())[0]
    '''
        
    '''
    普通数据结构转换成tensor
    '''
    '''
    def to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float).cuda()
    '''
    
    '''
    根据observation转换成网络所输入的tensor
    '''
    '''
    def observation2tensor(self, observation):
        #part 1. patient明确表示postive的症状
        obs = self.sindex2tensor(observation)
        
        #part 2. 候选疾病
        dids = list()
        for index in observation:
            sid = self.idx2sid[index]
            did = self.s2d[sid]
            dids.extend(did)
        dids = set(dids)
        y = np.zeros(self.disease_size)
        for did in dids:
            idx = self.did2idx[did]
            y[idx] = 1
           
        #part 3. 候选症状
        exclusive_index =  set(set(self.selected_index) or set(self.wrong_index))
        sids = list()
        for did in dids:
            sid = self.d2s[did]
            sids.extend(sid)
        sids = set(sids)    
        x = np.zeros(self.symptom_size)    
        for sid in sids:
            idx = self.sid2idx[sid]
            if idx not in exclusive_index:
                x[idx] = 1
                
        st = self.to_tensor([x])
        dt = self.to_tensor([y])
        return torch.cat((obs, st, dt), dim=1)
    '''
    
    '''
    输入症状名
    '''
    '''
    def input_symptom_name(self, name):
        self.input_sid(self.sname2sid[name])
    '''    
    '''
    输入多个症状名
    '''
    '''
    def input_symptom_names(self, names):
        for name in names:
            self.input_sid(self.sname2sid[name])
    '''
    
    '''
    更新否认的症状
    '''
    '''
    def update_wrong_symptom_name(self, name):
        self.update_wrong_sid([self.sname2sid[name]])
    '''    
        
    '''
    更新多个否认的症状
    '''
    '''
    def update_wrong_symptom_names(self, names):
        for name in names:
            self.update_wrong_sid([self.sname2sid[name]])
    '''
        
    '''
    建议症状
    '''
    '''
    def suggestion_symptom_name(self, num=10):
        _sids, _scores = self.suggestion(num)
        return [self.sid2sname[_sid] for _sid in _sids], _scores
    '''
        
    '''
    推断疾病
    '''        
    def inference_disease_with_probs(self, num=10):
        return self.inference_with_probs(num)
        #return [self.id2cname[did] for did in dids], probs

    def get_dname(self, did):
        return self.id2cname[did]
        
    def get_sname(self, sid):
        return self.id2cname[sid]
        
    def get_snames(self, sids):
        return [self.id2cname(sid) for sid in sids]
        
    '''
    #依据与候选疾病交集最多的症状进行排名
    def reorder(self, suggestion):
        _suggestion = [(su, len(self.s2d[su] & self.current_candidate_disease)) for su in suggestion]
        _suggestion.sort(key=lambda a: a[1], reverse=True)
        _suggestion = [su for su, _ in _suggestion]
        return _suggestion
        
    #依据与候选疾病交集最多的症状进行排名
    def reorder_name(self, suggestion, current_candidate_disease):
        suggestion_id = [robot.sname2sid[su] for su in suggestion]
        _suggestion = [(su, len(s2d[su] & current_candidate_disease)) for su in suggestion_id]
        _suggestion.sort(key=lambda a: a[1], reverse=True)
        _suggestion = [robot.sid2sname[su] for su, _ in _suggestion]
        return _suggestion
    '''    
    
    #'''
    #'''
    #def batch_observation2tensor2tensor(self, observations, selected_idxs):
    #    ts = []
    #    for i in range(len(observations)):
    #        observation = observations[i]
    #        selected_idx = selected_idxs[i]
    #        ts.append(self.observation2tensor(observation, selected_idx))
    #    return torch.stack(ts, dim=0)
    
    ########functions############
    def mute_batch(xs, p2=0.1):
        # xs is the batch input
        result = None
    #     nodes_list = [mute(x, p2) for x in xs]
        nodes = [mute(x, p2) for x in xs]
        nodes = list(zip(*nodes))
        node_id_list, nodes_list = nodes[0], nodes[1]
        
        nodes_list = torch.stack(nodes_list)
        batch_size = len(xs)
        n_matrix = model_n_matrix.repeat(batch_size, 1, 1)
        result = nodes_list.bmm(n_matrix).squeeze(1)
        return node_id_list, result

    def mute(x, p2=0.1):
        input_list = generate_muted_input_list(x, p2)
        node_id_list, nodes = convert(input_list)
        return node_id_list, nodes

    def generate_muted_input_list(x, p2=0.1):
        input_list = []
        if len(x) == 1:
            input_list.append(x[0])
        else:
            if train_model:
                n = random.randint(1, len(x))
                random.shuffle(x)
                x = x[0:n]
                input_list = x
            else:
                for item in x:
                    if random.random() >= p2:
                        #drop
                        input_list.append(item)
                if len(input_list) == 0:
                    num = random.randrange(0, len(x))
                    input_list.append(x[num])
        return input_list
        
    def convert(self, input_list):
        node_id_list = input_list

        #nodes [1 * V]
        nodes = self.sindex2tensor(node_id_list)

        return node_id_list, nodes

    def sindex2tensor(self, idxs):
        x = np.zeros(self.input_dimension)
        for idx in idxs:
            x[int(idx)] = 1
        return self.to_tensor([x])

    def to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float).cuda()

    def combine(self, nodes):
        nodes_list = nodes.unsqueeze(0).cuda()
        batch_size = 1
        n_matrix = self.model_n_matrix.repeat(batch_size, 1, 1)
        result = nodes_list.bmm(n_matrix).squeeze(1)
        return result

    def make_data(num):
        samples_diseases = [random.choice(multi_symptom_disease) for i in range(num)]
        samples_symptoms_id = [d2s[did] for did in samples_diseases]
        target_symptoms = []
        for i in range(num):
            target_symptom = random.choice(samples_symptoms_id[i])
            _samples_symptom = samples_symptoms_id[i].copy()
            samples_symptoms_id[i] = list(set(_samples_symptom) - set([target_symptom]))
            #y = np.zeros(output_dimension)
            #y[int(target_symptom)] = 1
            target_symptoms.append(suggestion_id2index[target_symptom])
        target_symptoms = torch.torch.LongTensor(target_symptoms).cuda()
        samples_symptoms_id, samples_symptoms = mute_batch(samples_symptoms_id)
        return samples_symptoms_id, samples_symptoms, target_symptoms
    
if __name__ == '__main__':
    robot = Robot()
    
    
