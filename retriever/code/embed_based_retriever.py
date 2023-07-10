from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import random
from sentence_transformers import SentenceTransformer
import torch

from sklearn.cluster import KMeans, AgglomerativeClustering
import copy 

def state_to_NL(slot_value_dict):
    output = "[CONTEXT] "
    for k, v in slot_value_dict.items():
        output += f"{' '.join(k.split('-'))}: {v.split('|')[0]}, "
    return output

def input_to_string(context_dict, sys_utt, usr_utt):
    history = state_to_NL(context_dict)
    if sys_utt == 'none':
        sys_utt = ''
    if usr_utt == 'none':
        usr_utt = ''
    history += f" [SYS] {sys_utt} [USER] {usr_utt}"
    return history


class Retriever:
    
    def normalize(self, emb):
        return emb/np.linalg.norm(emb, axis=-1,keepdims=True)
    
    def __init__(self, emb_dict):
        
        # to query faster, stack all embeddings and record keys
        self.emb_keys = list(emb_dict.keys())
        emb_dim = emb_dict[self.emb_keys[0]].shape[-1]
    
        self.emb_values = np.zeros((len(self.emb_keys), emb_dim))
        for i, k in enumerate(self.emb_keys):
            self.emb_values[i] = emb_dict[k]
        
        # normalize for cosine distance (kdtree only support euclidean when p=2)
        self.emb_values = self.normalize(self.emb_values)
        self.kdtree = KDTree(self.emb_values)
        
    def topk_nearest_dialogs(self, query_emb, k=5):
        query_emb = self.normalize(query_emb)
        if k == 1:
            return [self.emb_keys[i] for i in self.kdtree.query(query_emb, k=k, p=2)[1]]
        return [self.emb_keys[i] for i in self.kdtree.query(query_emb, k=k,p=2)[1][0]]
    
    def topk_nearest_distinct_dialogs(self, query_emb, k=5):
        return self.topk_nearest_dialogs(query_emb, k=k)
    
    def random_retrieve(self,k=5):
        return random.sample(self.emb_keys,k)

    def topk_nearest_distinct_embeddings(self, query_emb, k=5):
        query_emb = self.normalize(query_emb)
        if k == 1:
            return [[self.emb_keys[i], self.emb_values[i]] for i in self.kdtree.query(query_emb, k=k, p=2)[1]]
        return [[self.emb_keys[i], self.emb_values[i]] for i in self.kdtree.query(query_emb, k=k,p=2)[1][0]]
    

class EmbeddingRetriever:
    
    # sample selection
    def random_sample_selection_by_turn(self, embs, ratio=0.1):
        n_selected = int(ratio*len(embs))
        print(f"randomly select {ratio} of turns, i.e. {n_selected} turns")
        selected_keys = random.sample(list(embs),n_selected)
        return {k:v for k,v in embs.items() if k in selected_keys}
    
    def random_sample_selection_by_dialog(self, embs, ratio=0.1):
        dial_ids = set([turn_label.split('_')[0] for turn_label in embs.keys()])
        n_selected = int(len(dial_ids)*ratio)
        print(f"randomly select {ratio} of dialogs, i.e. {n_selected} dialogs")
        selected_dial_ids = random.sample(dial_ids, n_selected)
        return {k:v for k,v in embs.items() if k.split('_')[0] in selected_dial_ids}

    def pre_assigned_sample_selection(self, embs, examples):
        selected_dial_ids = set([dial['ID'] for dial in examples])
        return {k:v for k,v in embs.items() if k.split('_')[0] in selected_dial_ids}

    
    def __init__(self, datasets, model_path, search_index_filename, sampling_method="none", ratio=1.0, model=None, full_history=False):
        
        # data_items: list of datasets in this notebook. Please include datasets for both search and query
        # embedding_filenames: list of strings. embedding dictionary npy files. Should contain embeddings of the datasets. No need to be same
        # search_index:  string. a single npy filename, the embeddings of search candidates
        # sampling method: "random_by_turn", "random_by_dialog", "kmeans_cosine", "pre_assigned"
        # ratio: how much portion is selected
        
        self.data_items = []
        for dataset in datasets:
            self.data_items += dataset
        
        # save all embeddings and dial_id_turn_id in a dictionary
        self.model = model
        self.full_history = full_history

        if model is None:
            self.model = SentenceTransformer(model_path)
        
        # load the search index embeddings
        self.search_embs = np.load(search_index_filename, allow_pickle=True).item()
        
        # sample selection of search index
        if sampling_method == "none":
            self.retriever = Retriever(self.search_embs)
        elif sampling_method == 'random_by_dialog':
            self.retriever = Retriever(self.random_sample_selection_by_dialog(self.search_embs, ratio=ratio))
        elif sampling_method == 'random_by_turn':
            self.retriever = Retriever(self.random_sample_selection_by_turn(self.search_embs, ratio=ratio))
        elif sampling_method == 'pre_assigned':
            self.retriever = Retriever(self.pre_assigned_sample_selection(self.search_embs, self.data_items))
        else:
            raise ValueError("selection method not supported")

    def data_item_to_embedding(self, data_item):

        def data_item_to_string(data_item):

            # use full history, depend on retriever training (for ablation)
            if self.full_history:
                history = ""
                for sys_utt, usr_utt in zip(data_item['dialog']['sys'], data_item['dialog']['usr']):
                    history += input_to_string({}, sys_utt, usr_utt)
                return history

            # use single turn
            context = data_item['last_slot_values']
            sys_utt = data_item['dialog']['sys'][-1]
            usr_utt = data_item['dialog']['usr'][-1]
            history = input_to_string(context, sys_utt, usr_utt)
            return history

        with torch.no_grad():
            embed = self.model.encode(data_item_to_string(
                data_item), convert_to_numpy=True).reshape(1, -1)
        return embed
    
    def label_to_data_item(self, label):
        ID, _, turn_id = label.split('_')
        turn_id = int(turn_id)
        
        for d in self.data_items:
            if d['ID'] == ID and d['turn_id'] == turn_id:
                return d
        raise ValueError(f"label {label} not found. check data items input")
    
    # examples 생성하는 함수
    def item_to_nearest_examples(self, data_item, k=5):
        # the nearest neighbor is at the end
        return [self.label_to_data_item(l) 
                for l in self.retriever.topk_nearest_distinct_dialogs(
                    self.data_item_to_embedding(data_item), k=k) # data_item_to_embedding은 dialogue history를 입력으로 만든 임베딩
               ][::-1]

    def label_to_nearest_labels(self, label, k=5):
        data_item = self.label_to_data_item(label)
        return [l for l in self.retriever.topk_nearest_distinct_dialogs(
                    self.data_item_to_embedding(data_item), k=k)
                ][::-1]
    
    def random_examples(self, data_item, k=5):
        return [self.label_to_data_item(l)
                for l in self.retriever.random_retrieve(k=k)
               ]
    
    def kmean_cluster(self, example_embeddings, n_clusters):
        clustering_model = KMeans(n_clusters=n_clusters)

        clustering_model.fit(example_embeddings)
        cluster_assignment = clustering_model.labels_

        print(cluster_assignment)

        clustered_embs = [[] for i in range(n_clusters)]
        for emb_idx, cluster in enumerate(cluster_assignment):
            clustered_embs[cluster].append(emb_idx)

        # sort cluster_embs according to the number of elements
        clustered_embs = sorted(clustered_embs, key= lambda x: len(x), reverse=True)

        return clustered_embs

    def agglo_cluster(self, example_embeddings):
        clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5)

        clustering_model.fit(example_embeddings)
        cluster_assignment = clustering_model.labels_

        print(cluster_assignment)

        clustered_embs = [[] for i in range(len(set(cluster_assignment)))]
        for emb_idx, cluster in enumerate(cluster_assignment):
            clustered_embs[cluster].append(emb_idx)

        # sort cluster_embs according to the number of elements
        clustered_embs = sorted(clustered_embs, key= lambda x: len(x), reverse=True)

        return clustered_embs


    def dynamic_cluster_examples(self, data_item, k=10, max_ex=0, min_ex=0):
        # the nearest neighbor is listed last (ascending order)
        example_list = [l for l in self.retriever.topk_nearest_distinct_embeddings(
            self.data_item_to_embedding(data_item), k=k)][::-1]

        example_keys = [example[0] for example in example_list]
        example_embeddings = [example[1] for example in example_list]

        print(example_keys)

        # agglo
        clustered_embs = self.agglo_cluster(example_embeddings)

        # copy
        result_embs = copy.deepcopy(clustered_embs)
        
        # limit the examples according to the number of clusters
        # number of examples = number of clusters + (number of clusters - 1)

        n_clusters = len(result_embs)
        n_dynamic_example = n_clusters + (n_clusters - 1)
        n_dynamic_example = max_ex if n_dynamic_example > max_ex else n_dynamic_example # limit max
        n_dynamic_example = min_ex if n_dynamic_example < min_ex else n_dynamic_example # limit min

        # select examples from the clusters
        selected_list = []
        while n_dynamic_example > 0:
            # print(f"ce: {clustered_embs}")
            for cluster_idx in range(len(clustered_embs)):
                selected_list.append(example_keys[clustered_embs[cluster_idx][-1]])
                n_dynamic_example -= 1

                # remove appended example
                del clustered_embs[cluster_idx][-1]

                if n_dynamic_example == 0:
                    break

            # remove empty cluster
            clustered_embs = [i for i in clustered_embs if i]

            # print(f"ce: {clustered_embs}")
        
        # sort selected_list according to the order of nearest neighbors
        selected_list = sorted(selected_list, key=lambda x: example_keys.index(x))
        print(selected_list)

        return [self.label_to_data_item(label) for label in selected_list], result_embs


    def cluster_examples(self, data_item, n_clusters=3, k=10, n_example=10):
        # the nearest neighbor is listed last (ascending order)
        example_list = [l for l in self.retriever.topk_nearest_distinct_embeddings(
            self.data_item_to_embedding(data_item), k=k)][::-1]

        example_keys = [example[0] for example in example_list]
        example_embeddings = [example[1] for example in example_list]

        print(example_keys)

        # agglo
        clustered_embs = self.agglo_cluster(example_embeddings)

        # copy
        result_embs = copy.deepcopy(clustered_embs)
        
        assert k >= n_example, 'n_example ({0}) is bigger than k ({1})'.format(n_example, k)

        # select examples from the clusters
        selected_list = []
        while n_example > 0:
            # print(f"ce: {clustered_embs}")
            for cluster_idx in range(len(clustered_embs)):
                selected_list.append(example_keys[clustered_embs[cluster_idx][-1]])
                n_example -= 1

                # remove appended example
                del clustered_embs[cluster_idx][-1]

                if n_example == 0:
                    break

            # remove empty cluster
            clustered_embs = [i for i in clustered_embs if i]

            # print(f"ce: {clustered_embs}")
        
        # sort selected_list according to the order of nearest neighbors
        selected_list = sorted(selected_list, key=lambda x: example_keys.index(x))
        print(selected_list)

        return [self.label_to_data_item(label) for label in selected_list], result_embs
            







        

        


        