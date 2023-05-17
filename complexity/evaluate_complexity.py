# from scipy.spatial import KDTree
# from sklearn.cluster import KMeans
# from sklearn.metrics import pairwise_distances_argmin_min
# import numpy as np
# import random
# from sentence_transformers import SentenceTransformer
# import torch

import json
import scipy.stats as stats

# def state_to_NL(slot_value_dict):
#     output = "[CONTEXT] "
#     for k, v in slot_value_dict.items():
#         output += f"{' '.join(k.split('-'))}: {v.split('|')[0]}, "
#     return output

# def input_to_separate_string(context_dict, sys_utt, usr_utt):
#     history = state_to_NL(context_dict)
#     if sys_utt == 'none':
#         sys_utt = ''
#     if usr_utt == 'none':
#         usr_utt = ''
    
#     sys_hist = f"{history} [system] {sys_utt}" if sys_utt != '' else ""
#     usr_hist = f"{history} [user] {usr_utt}" if usr_utt != '' else ""
#     return sys_hist, usr_hist

# def input_to_string(context_dict, sys_utt, usr_utt):
#     history = state_to_NL(context_dict)
#     if sys_utt == 'none':
#         sys_utt = ''
#     if usr_utt == 'none':
#         usr_utt = ''
#     history += f" [SYS] {sys_utt} [USER] {usr_utt}"
#     return history

class ComplexityEvaluator:

    def __init__(self, train_data_path):
        with open(train_data_path,'r') as f:
            self.train_dataset = json.load(f)
    
        self.usr_len = []
        self.sys_len = []
        self.all_len = []

        self.context_count = []
        self.updated_count = []

        for data_item in self.train_dataset:
            usr_utt = data_item['dialog']['usr'][-1]
            sys_utt = data_item['dialog']['sys'][-1]
            
            self.usr_len.append(len(usr_utt))
            if not len(sys_utt) == 0: # 0일때는 지움
                self.sys_len.append(len(sys_utt))
                
            self.all_len.append(len(usr_utt) + len(sys_utt))

            self.context_count.append(len(data_item['last_slot_values']))

            last_slot_values = data_item['last_slot_values']
            slot_values = data_item["slot_values"]
            
            n_update = 0
            for s,v in slot_values.items():
                if s in last_slot_values.keys():
                    if v != last_slot_values[s]:
                        n_update += 1
                else:
                    n_update += 1
            
            self.updated_count.append(n_update)            

    def length_complexity_score(self, data_item, alpha=0.5):
        target_usr_len = len(data_item['dialog']['usr'][-1])
        target_sys_len = len(data_item['dialog']['sys'][-1])

        usr_len_score = stats.percentileofscore(self.usr_len, target_usr_len) / 100
        all_len_score = stats.percentileofscore(self.all_len, target_usr_len + target_sys_len) / 100

        return usr_len_score * alpha + all_len_score * (1 - alpha)

    def context_complexity_score(self, data_item, alpha=0.7):
        target_n_context = len(data_item['last_slot_values'])

        last_slot_values = data_item['last_slot_values']
        slot_values = data_item["slot_values"]
        target_n_update = 0
        for s,v in slot_values.items():
            if s in last_slot_values.keys():
                if v != last_slot_values[s]:
                    target_n_update += 1
            else:
                target_n_update += 1

        context_score = stats.percentileofscore(self.context_count, target_n_context) / 100
        update_score = stats.percentileofscore(self.updated_count, target_n_update) / 100

        return update_score * alpha + context_score * (1 - alpha)

