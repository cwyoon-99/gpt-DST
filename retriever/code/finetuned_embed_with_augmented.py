import numpy as np
import json
import random
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from index_based_retriever import IndexRetriever
from embed_based_retriever import EmbeddingRetriever, input_to_string
from retriever_evaluation import evaluate_retriever_on_dataset, compute_sv_sim
from st_evaluator import RetrievalEvaluator

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_fn', type=str, required=True, help="training data file (few-shot or full shot)")  # e.g. "../../data/mw21_10p_train_v3.json"
parser.add_argument('--load_retriever_dir', type=str, required=True, help="sentence transformer saved path")  # e.g. mw21_10p_v3
parser.add_argument('--pretrained_index_dir', type=str, default="all_mpnet_base_v2", help="directory of pretrained embeddings")  
parser.add_argument('--pretrained_model', type=str, default='sentence-transformers/all-mpnet-base-v2', help="embedding model to finetune with")
parser.add_argument('--epoch', type=int, default=15)
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--toprange', type=int, default=200)
parser.add_argument('--aug_name', type=str, required=True, help="name of augmented dialogue")
parser.add_argument('--emb_save_dir', type=str, default= "../expts/base")

args = parser.parse_args()


TRAIN_FN = args.train_fn

SAVE_NAME = args.load_retriever_dir

PRETRAINED_MODEL = args.pretrained_index_dir
MODEL_NAME = args.pretrained_model

EPOCH = args.epoch
TOPK = args.topk
TOPRANGE = args.toprange


# ------------ CONFIG ends here ------------
SAVE_PATH = f"../expts/{SAVE_NAME}"
PRETRAINED_MODEL_SAVE_PATH = f"../expts/{PRETRAINED_MODEL}"

with open(TRAIN_FN) as f:
    train_set = json.load(f)

# # prepare pretrained retreiver for fine-tuning
# pretrained_train_retriever = IndexRetriever(datasets=[train_set],
#                                             embedding_filenames=[
#     f"{PRETRAINED_MODEL_SAVE_PATH}/mw21_train_{PRETRAINED_MODEL}.npy"],
#     search_index_filename=f"{PRETRAINED_MODEL_SAVE_PATH}/mw21_train_{PRETRAINED_MODEL}.npy",
#     sampling_method="pre_assigned",
# )


# load multiWoZ and calculate all similiarities of dialogue states between turns  
class MWDataset:

    def __init__(self, mw_json,  just_embed_all=False):

        # Only care domain in test
        DOMAINS = ['hotel', 'restaurant', 'attraction', 'taxi', 'train']

        data = mw_json

        
        self.turn_labels = []  # store [SMUL1843.json_turn_1, ]
        self.turn_utts = []  # store corresponding text
        self.turn_states = []  # store corresponding states. [['attraction-type-mueseum',],]


        for turn in data:
            # filter the domains that not belongs to the test domain
            if not set(turn["domains"]).issubset(set(DOMAINS)):
                continue                

            # update dialogue history
            sys_utt = turn["dialog"]['sys'][-1]
            usr_utt = turn["dialog"]['usr'][-1]

            if sys_utt == 'none':
                sys_utt = ''
            if usr_utt == 'none':
                usr_utt = ''

            context = turn["last_slot_values"]

            current_state = turn["turn_slot_values"]

            # if turn is augmented one
            if args.aug_name in turn["ID"]:
                # print(f"original usr utt: {usr_utt}")
                # print(f"original state: {current_state}")
                
                usr_utt = turn['ag_usr_utt']
                current_state = turn['ag_turn_slot_values']

                # print(f"augmented usr utt: {usr_utt}")
                # print(f"augmented state: {current_state}")
                # print()


            # convert to list of strings
            current_state = [self.important_value_to_string(s, v) for s, v in current_state.items()
                                if s.split('-')[0] in DOMAINS]

            history = input_to_string(context, sys_utt, usr_utt)

            self.turn_labels.append(f"{turn['ID']}_turn_{turn['turn_id']}")
            self.turn_utts.append(history)
            self.turn_states.append(current_state)

        self.n_turns = len(self.turn_labels)
        print(f"there are {self.n_turns} turns in this dataset")

        if not just_embed_all:
            # compute all similarity
            self.similarity_matrix = np.zeros((self.n_turns, self.n_turns))
            for i in tqdm(range(self.n_turns)):
                self.similarity_matrix[i, i] = 1
                for j in range(i, self.n_turns):
                    self.similarity_matrix[i, j] = compute_sv_sim(self.turn_states[i],
                                                                  self.turn_states[j])
                    self.similarity_matrix[j, i] = self.similarity_matrix[i, j]

    def important_value_to_string(self, slot, value):
        if value in ["none", "dontcare"]:
            return f"{slot}{value}"  # special slot
        return f"{slot}-{value}"


# class MWContrastiveDataloader:

#     def __init__(self, f1_set, pretrained_retriever):
#         self.f1_set = f1_set
#         self.pretrained_retriever = pretrained_retriever

#     def hard_negative_sampling(self, topk=10, top_range=100):
#         sentences1 = []
#         sentences2 = []
#         scores = []

#         # do hard negative sampling
#         for ind in tqdm(range(self.f1_set.n_turns)):
            
#             # 해당 turn의 
#             # find nearest neighbors given by pre-trained retriever
#             this_label = self.f1_set.turn_labels[ind]
#             nearest_labels = self.pretrained_retriever.label_to_nearest_labels(
#                 this_label, k=top_range+1)[:-1]  # to exclude itself
#             nearest_args = [self.f1_set.turn_labels.index(
#                 l) for l in nearest_labels]

#             # topk and bottomk nearest f1 score examples, as hard examples
#             similarities = self.f1_set.similarity_matrix[ind][nearest_args]
#             sorted_args = similarities.argsort()

#             # sort한 것에서 상위 k개, 하위 k개 뽑음
#             chosen_positive_args = list(sorted_args[-topk:])
#             chosen_negative_args = list(sorted_args[:topk])

#             chosen_positive_args = np.array(nearest_args)[chosen_positive_args]
#             chosen_negative_args = np.array(nearest_args)[chosen_negative_args]

#             for chosen_arg in chosen_positive_args:
#                 sentences1.append(self.f1_set.turn_utts[ind])
#                 sentences2.append(self.f1_set.turn_utts[chosen_arg])
#                 scores.append(1)

#             for chosen_arg in chosen_negative_args:
#                 sentences1.append(self.f1_set.turn_utts[ind])
#                 sentences2.append(self.f1_set.turn_utts[chosen_arg])
#                 scores.append(0)

#         return sentences1, sentences2, scores

#     def generate_easy_hard_examples(self, topk=5):
#         sentences1 = []
#         sentences2 = []
#         scores = []
#         for i in range(self.f1_set.n_turns):
#             sorted_args = self.f1_set.similarity_matrix[i].argsort()
#             chosen_args = list(sorted_args[:topk]) + \
#                 list(sorted_args[-topk-1:])
#             if i in chosen_args:
#                 chosen_args.remove(i)
#             for chosen_arg in chosen_args:
#                 sentences1.append(self.f1_set.turn_utts[i])
#                 sentences2.append(self.f1_set.turn_utts[chosen_arg])
#                 scores.append(self.f1_set.similarity_matrix[i, chosen_arg])
#         return sentences1, sentences2, scores

#     def generate_random_examples(self):
#         indexes = list(range(self.f1_set.n_turns))
#         random.shuffle(indexes)
#         contrastive_utts = [self.f1_set.turn_utts[i] for i in indexes]
#         contrastive_f1s = [self.f1_set.similarity_matrix[original, contrast]
#                            for original, contrast in enumerate(indexes)]
#         return self.f1_set.turn_utts, contrastive_utts, contrastive_f1s

#     def generate_eval_examples(self, topk=5, top_range=100):
#         # add topk closest, furthest, and n_random random indices
#         sentences1, sentences2, scores = self.hard_negative_sampling(
#             topk=topk, top_range=top_range)
#         scores = [float(s) for s in scores]
#         return sentences1, sentences2, scores

#     def generate_train_examples(self, topk=5, top_range=100):
#         sentences1, sentences2, scores = self.generate_eval_examples(
#             topk=topk, top_range=top_range)
#         n_samples = len(sentences1)
#         return [InputExample(texts=[sentences1[i], sentences2[i]], label=scores[i])
#                 for i in range(n_samples)]

# store embedding function
def store_embed(dataset, output_filename):
    # dataset.turn_utts에는 history가 담김 (context + sys_utt + usr_utt)
    embeddings = model.encode(dataset.turn_utts, convert_to_numpy=True)
    output = {}
    for i in tqdm(range(len(embeddings))):
        output[dataset.turn_labels[i]] = embeddings[i:i+1]
    np.save(output_filename, output)
    return

# load best model
model = SentenceTransformer(SAVE_PATH, device="cuda:1")

with open("../../data/mw21_100p_train.json",'r') as f:
    full_data = json.load(f)

with open("../../augments/230517_2222-augment_element_2/augment_log.json", 'r') as f:
    augment_element2_data = json.load(f)

# add augmented dataset
augmented_data = full_data + augment_element2_data

full_train_set = MWDataset(augmented_data, just_embed_all=True)

os.makedirs(args.emb_save_dir, exist_ok=True)
store_embed(full_train_set, f"{args.emb_save_dir}/train_index.npy")
