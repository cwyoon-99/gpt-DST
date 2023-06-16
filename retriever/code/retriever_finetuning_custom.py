import numpy as np
import os
import json
import random
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from index_based_retriever import IndexRetriever
from embed_based_retriever import EmbeddingRetriever, input_to_string
# from retriever_evaluation import evaluate_retriever_on_dataset, compute_sv_sim
from retriever_evaluation_custom import evaluate_retriever_on_dataset, compute_sv_sim, compute_entity_sim, evaluate_retriever_on_dataset_with_ett_score
# from st_evaluator import RetrievalEvaluator
from st_evaluator_custom import RetrievalEvaluator

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_fn', type=str, required=True, help="training data file (few-shot or full shot)")  # e.g. "../../data/mw21_10p_train_v3.json"
parser.add_argument('--save_name', type=str, required=True, help="sentence transformer save path")  # e.g. mw21_10p_v3
parser.add_argument('--pretrained_index_dir', type=str, default="all_mpnet_base_v2", help="directory of pretrained embeddings")  
parser.add_argument('--pretrained_model', type=str, default='sentence-transformers/all-mpnet-base-v2', help="embedding model to finetune with")
parser.add_argument('--epoch', type=int, default=15)
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--toprange', type=int, default=200)

parser.add_argument('--entity_lambda', type=float, default=0)
parser.add_argument('--dev_fn', type=str, default= "../../data/mw24_100p_dev.json")
parser.add_argument('--test_fn', type=str, default= "../../data/mw24_100p_test.json")

args = parser.parse_args()


TRAIN_FN = args.train_fn

SAVE_NAME = args.save_name

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

# prepare pretrained retreiver for fine-tuning
pretrained_train_retriever = IndexRetriever(datasets=[train_set],
                                            embedding_filenames=[
    f"{PRETRAINED_MODEL_SAVE_PATH}/mw21_train_{PRETRAINED_MODEL}.npy"],
    search_index_filename=f"{PRETRAINED_MODEL_SAVE_PATH}/mw21_train_{PRETRAINED_MODEL}.npy",
    sampling_method="pre_assigned",
)


# load multiWoZ and calculate all similiarities of dialogue states between turns  
class MWDataset:

    def __init__(self, mw_json_fn,  just_embed_all=False):

        # Only care domain in test
        DOMAINS = ['hotel', 'restaurant', 'attraction', 'taxi', 'train']

        with open(mw_json_fn, 'r') as f:
            data = json.load(f)


        self.turn_labels = []  # store [SMUL1843.json_turn_1, ]
        self.turn_utts = []  # store corresponding text
        self.turn_states = []  # store corresponding states. [['attraction-type-mueseum',],]

        # self.turn_entities = [] # store corresponing NER result

        self.turn_sys_entities = []
        self.turn_usr_entities = []


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
            history = input_to_string(context, sys_utt, usr_utt)

            # # modify input for retriever of ett replaced dialog
            # history = "[CONTEXT] "
            # for k,v in context.items():
            #     history += f"{' '.join(k.split('-'))}, "

            # domain_set = set()
            # for k, v in context.items():
            #     domain_set.add(k.split('-')[0])
            # history += ' '.join(list(domain_set))
        
            # history += f" [SYS] {sys_utt} [USER] {usr_utt}"

            # print(history)

            current_state = turn["turn_slot_values"]
            # convert to list of strings
            current_state = [self.important_value_to_string(s, v) for s, v in current_state.items()
                                if s.split('-')[0] in DOMAINS]

            self.turn_labels.append(f"{turn['ID']}_turn_{turn['turn_id']}")
            self.turn_utts.append(history)
            self.turn_states.append(current_state)

            self.turn_sys_entities.append(turn['sys_entity'])
            self.turn_usr_entities.append(turn['usr_entity'])


        self.n_turns = len(self.turn_labels)
        print(f"there are {self.n_turns} turns in this dataset")

        if not just_embed_all:
            # compute all similarity
            self.similarity_matrix = np.zeros((self.n_turns, self.n_turns))

            # neg matrix
            self.pos_similarity_matrix = np.zeros((self.n_turns, self.n_turns))
            self.neg_similarity_matrix = np.zeros((self.n_turns, self.n_turns))

            for i in tqdm(range(self.n_turns)):
                self.similarity_matrix[i, i] = 1

                self.pos_similarity_matrix[i, i] = 1
                self.neg_similarity_matrix[i, i] = 1

                for j in range(i, self.n_turns):
                    # # base
                    # self.similarity_matrix[i, j] = compute_sv_sim(self.turn_states[i],
                    #                                             self.turn_states[j])
                    
                    # slot similarities only.
                    value_sim, slot_sim = compute_sv_sim(self.turn_states[i], self.turn_states[j], onescore=False)
                    self.similarity_matrix[i, j] = slot_sim

                    # entity f1 score
                    sv_sim = compute_sv_sim(self.turn_states[i],self.turn_states[j])
                    value_sim, slot_sim = compute_sv_sim(self.turn_states[i], self.turn_states[j], onescore=False)

                    # entity_sim = compute_entity_sim(self.turn_entities[i], self.turn_entities[j])

                    # separate entity F1 score
                    entity_sim = (compute_entity_sim(self.turn_sys_entities[i], 
                                                    self.turn_sys_entities[j]) + compute_entity_sim(self.turn_usr_entities[i], 
                                                                                                    self.turn_usr_entities[j])) / 2
                    
                    # self.similarity_matrix[i, j] = value_sim * ((1 - args.entity_lambda) / 2) + slot_sim * ((1 - args.entity_lambda) / 2) + entity_sim * (args.entity_lambda)

                    self.similarity_matrix[j, i] = self.similarity_matrix[i, j]

                    # compute pos and neg matrix
                    self.pos_similarity_matrix[i, j] = value_sim * ((1 - args.entity_lambda) / 2) + slot_sim * ((1 - args.entity_lambda) / 2) + entity_sim * (args.entity_lambda)
                    # self.pos_similarity_matrix[i, j] = slot_sim * ((1 - args.entity_lambda) / 2) + entity_sim * (args.entity_lambda) # only slot sim with entity_sim

                    self.pos_similarity_matrix[j, i] = self.pos_similarity_matrix[i, j]

                    self.neg_similarity_matrix[i, j] = value_sim * ((1 - args.entity_lambda) / 2) + slot_sim * ((1 - args.entity_lambda) / 2) - entity_sim * (args.entity_lambda)

                    self.neg_similarity_matrix[j, i] = self.neg_similarity_matrix[i, j]

    def important_value_to_string(self, slot, value):
        if value in ["none", "dontcare"]:
            return f"{slot}{value}"  # special slot
        return f"{slot}-{value}"


class MWContrastiveDataloader:

    def __init__(self, f1_set, pretrained_retriever):
        self.f1_set = f1_set
        self.pretrained_retriever = pretrained_retriever

    def hard_negative_sampling(self, topk=10, top_range=100):
        sentences1 = []
        sentences2 = []
        scores = []

        # do hard negative sampling
        for ind in tqdm(range(self.f1_set.n_turns)):
            
            # 해당 turn의 
            # find nearest neighbors given by pre-trained retriever
            this_label = self.f1_set.turn_labels[ind]
            nearest_labels = self.pretrained_retriever.label_to_nearest_labels(
                this_label, k=top_range+1)[:-1]  # to exclude itself
            nearest_args = [self.f1_set.turn_labels.index(
                l) for l in nearest_labels]

            # topk and bottomk nearest f1 score examples, as hard examples
            similarities = self.f1_set.similarity_matrix[ind][nearest_args]
            sorted_args = similarities.argsort()

            # sort한 것에서 상위 k개, 하위 k개 뽑음
            chosen_positive_args = list(sorted_args[-topk:])
            chosen_negative_args = list(sorted_args[:topk])

            chosen_positive_args = np.array(nearest_args)[chosen_positive_args]
            chosen_negative_args = np.array(nearest_args)[chosen_negative_args]

            # Topk nerest F1 score with positive similarity matrix
            pos_similarities = self.f1_set.pos_similarity_matrix[ind][nearest_args]
            pos_sorted_args = pos_similarities.argsort()

            chosen_positive_args = list(pos_sorted_args[-topk:])
            chosen_positive_args = np.array(nearest_args)[chosen_positive_args]

            # # bottomk nearest F1 score with negative similarity matrix
            # neg_similarities = self.f1_set.neg_similarity_matrix[ind][nearest_args]
            # neg_sorted_args = neg_similarities.argsort()

            # chosen_negative_args = list(neg_sorted_args[:topk])
            # chosen_negative_args = np.array(nearest_args)[chosen_negative_args]

            for chosen_arg in chosen_positive_args:
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[chosen_arg])
                scores.append(1)

            for chosen_arg in chosen_negative_args:
                sentences1.append(self.f1_set.turn_utts[ind])
                sentences2.append(self.f1_set.turn_utts[chosen_arg])
                scores.append(0)


            # hard negative sampling test
            topk_domains = set()

            # print("\n\n--------------------------------------------")
            # print(f"tg sentence: {self.f1_set.turn_utts[ind]}") 
            # print(f"tg state change: {self.f1_set.turn_states[ind]}")               
            for idx, chosen_arg in enumerate(chosen_negative_args):
                # print(f"neg sentence {idx} : {self.f1_set.turn_utts[chosen_arg]}")
                # print(f"state change: {self.f1_set.turn_states[chosen_arg]}")
                for state in self.f1_set.turn_states[chosen_arg]:
                    topk_domains.add(state.split('-', 1)[0])
                print()
            
            # tg_states = self.f1_set.turn_states[ind]
            # tg_domains = set()
            # for state in tg_states:
            #     tg_domains.add(state.split('-', 1)[0])

            # topk_check = False
            # for topk_domain in list(topk_domains):
            #     if topk_domain in tg_domains:
            #         topk_check = True

            # if not topk_check:
            #     print("\n\n--------------------------------------------")
            #     print(f"tg sentence: {self.f1_set.turn_utts[ind]}") 
            #     print(f"tg state change: {self.f1_set.turn_states[ind]}")      

            # bottom_args = sorted_args[:int(top_range / 4)]
            # chosen_bottom_args = np.array(nearest_args)[bottom_args]

            # for idx, chosen_arg in enumerate(chosen_bottom_args):
            #     chosen_domains = set()
            #     chosen_states = self.f1_set.turn_states[chosen_arg]
            #     for state in chosen_states:
            #         chosen_domains.add(state.split('-', 1)[0])
                
            #     check = False
            #     for dm in chosen_domains:
            #         if dm in tg_domains:
            #             check = True

            #     if check and not topk_check:
            #         print()
            #         print(f"sentence {idx} : {self.f1_set.turn_utts[chosen_arg]}")
            #         print(f"change: {self.f1_set.turn_states[chosen_arg]}")

        return sentences1, sentences2, scores

    def generate_easy_hard_examples(self, topk=5):
        sentences1 = []
        sentences2 = []
        scores = []
        for i in range(self.f1_set.n_turns):
            sorted_args = self.f1_set.similarity_matrix[i].argsort()
            chosen_args = list(sorted_args[:topk]) + \
                list(sorted_args[-topk-1:])
            if i in chosen_args:
                chosen_args.remove(i)
            for chosen_arg in chosen_args:
                sentences1.append(self.f1_set.turn_utts[i])
                sentences2.append(self.f1_set.turn_utts[chosen_arg])
                scores.append(self.f1_set.similarity_matrix[i, chosen_arg])
        return sentences1, sentences2, scores

    def generate_random_examples(self):
        indexes = list(range(self.f1_set.n_turns))
        random.shuffle(indexes)
        contrastive_utts = [self.f1_set.turn_utts[i] for i in indexes]
        contrastive_f1s = [self.f1_set.similarity_matrix[original, contrast]
                           for original, contrast in enumerate(indexes)]
        return self.f1_set.turn_utts, contrastive_utts, contrastive_f1s

    def generate_eval_examples(self, topk=5, top_range=100):
        # add topk closest, furthest, and n_random random indices
        sentences1, sentences2, scores = self.hard_negative_sampling(
            topk=topk, top_range=top_range)
        scores = [float(s) for s in scores]
        return sentences1, sentences2, scores

    def generate_train_examples(self, topk=5, top_range=100):
        sentences1, sentences2, scores = self.generate_eval_examples(
            topk=topk, top_range=top_range)
        n_samples = len(sentences1)
        return [InputExample(texts=[sentences1[i], sentences2[i]], label=scores[i])
                for i in range(n_samples)]


# Preparing dataset
f1_train_set = MWDataset(TRAIN_FN)

# Dataloader
mw_train_loader = MWContrastiveDataloader(
    f1_train_set, pretrained_train_retriever)


# store embedding function
def store_embed(dataset, output_filename):
    # dataset.turn_utts에는 history가 담김 (context + sys_utt + usr_utt)
    embeddings = model.encode(dataset.turn_utts, convert_to_numpy=True)
    output = {}
    for i in tqdm(range(len(embeddings))):
        output[dataset.turn_labels[i]] = embeddings[i:i+1]
    np.save(output_filename, output)
    return


# prepare the retriever model
word_embedding_model = models.Transformer(MODEL_NAME, max_seq_length=512)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension())

# add special tokens
tokens = ["[USER]", "[SYS]", "[CONTEXT]"]
# entity_tokens = ['TAXI-LEAVE', 'TAXI-CAR', 'TRAIN-PEOPLE', 'HOTEL-STAY', 'RESTAURANT-PEOPLE',
#                 'ATTRACTION-TYPE', 'RESTAURANT-REF', 'HOTEL-TYPE', 'HOTEL-ADDR', 'TRAIN-LEAVE',
#                 'TAXI-DEST', 'BOOKING-PEOPLE', 'TAXI-DEPART', 'BOOKING-TIME', 'HOTEL-REF', 'HOTEL-DAY',
#                 'RESTAURANT-PHONE', 'ATTRACTION-NAME', 'RESTAURANT-POST', 'ATTRACTION-PHONE', 'ATTRACTION-PRICE',
#                 'TAXI-ARRIVE', 'TRAIN-TIME', 'RESTAURANT-CHOICE', 'TRAIN-DEST', 'TAXI-PHONE', 'BOOKING-STAY', 
#                 'RESTAURANT-PRICE', 'TRAIN-DEPART', 'RESTAURANT-NAME', 'RESTAURANT-ADDR', 'BOOKING-REF', 
#                 'HOTEL-PEOPLE', 'RESTAURANT-FOOD', 'HOTEL-PRICE', 'HOTEL-PHONE', 'RESTAURANT-TIME', 'HOTEL-AREA', 
#                 'TRAIN-DAY', 'TRAIN-CHOICE', 'TRAIN-REF', 'TRAIN-ARRIVE', 'RESTAURANT-AREA', 'RESTAURANT-DAY', 
#                 'ATTRACTION-POST', 'HOTEL-POST', 'ATTRACTION-FEE', 'ATTRACTION-ADDR', 'HOTEL-STARS', 'HOTEL-NAME', 
#                 'ATTRACTION-AREA', 'HOTEL-CHOICE', 'BOOKING-NAME', 'TRAIN-ID', 'BOOKING-DAY', 'ATTRACTION-CHOICE', 'TRAIN-TICKET']
# entity_tokens = ['TAXI-LEAVE', 'TRAIN-PEOPLE', 'HOTEL-STAY', 'RESTAURANT-PEOPLE', 'ATTRACTION-TYPE', 'HOTEL-TYPE', 
#                 'TRAIN-LEAVE', 'TAXI-DEST', 'TAXI-DEPART', 'HOTEL-DAY', 'ATTRACTION-NAME', 'TAXI-ARRIVE', 'TRAIN-TIME',
#                 'TRAIN-DEST', 'RESTAURANT-PRICE', 'TRAIN-DEPART', 'RESTAURANT-NAME', 'HOTEL-PEOPLE', 'RESTAURANT-FOOD',
#                 'HOTEL-PRICE', 'RESTAURANT-TIME', 'HOTEL-AREA', 'TRAIN-DAY', 'TRAIN-ARRIVE', 'RESTAURANT-AREA',
#                 'RESTAURANT-DAY', 'HOTEL-STARS', 'HOTEL-NAME', 'ATTRACTION-AREA']
# tokens += entity_tokens

word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(
    len(word_embedding_model.tokenizer))

device = "cuda:0"
model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model], device=device)
print("Finish preparing model!")


# prepare training dataloaders
all_train_samples = mw_train_loader.generate_train_examples(
    topk=TOPK, top_range=TOPRANGE)
train_dataloader = DataLoader(all_train_samples, shuffle=True, batch_size=48)
print(f"number of batches {len(train_dataloader)}")

evaluator = RetrievalEvaluator(TRAIN_FN, f1_train_set, args.dev_fn)

# training
train_loss = losses.OnlineContrastiveLoss(model)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=EPOCH, warmup_steps=100,
        evaluator=evaluator, evaluation_steps=(len(train_dataloader) // 300 + 1)*100,
        output_path=SAVE_PATH)


# load best model
model = SentenceTransformer(SAVE_PATH, device=device)

# full_train_set = MWDataset("../../data/mw21_100p_train.json", just_embed_all=True)
# store_embed(full_train_set, f"{SAVE_PATH}/train_index.npy")

# instead of storing embeddings of full train set, only save few-shot train set embedding
store_embed(f1_train_set, f"{SAVE_PATH}/train_index.npy")

# retriever to evaluation
with open(TRAIN_FN) as f:
    train_set = json.load(f)
# with open("../../data/mw24_100p_dev.json") as f:
with open(args.test_fn) as f:
    test_set_21 = json.load(f)


retriever = EmbeddingRetriever(datasets=[train_set],
                               model_path=SAVE_PATH,
                               search_index_filename=f"{SAVE_PATH}/train_index.npy",
                               sampling_method="pre_assigned")
print("Now evaluating retriever ...")
turn_sv, turn_s, dial_sv, dial_s, turn_ett, turn_sysett, turn_usrett = evaluate_retriever_on_dataset_with_ett_score(
    test_set_21, retriever)
print(turn_sv, turn_s, dial_sv, dial_s, turn_ett, turn_sysett, turn_usrett)

with open(f"{SAVE_PATH}/eval.csv", "w") as f:
    f.write(f"{turn_sv, turn_s, dial_sv, dial_s, turn_ett}")
