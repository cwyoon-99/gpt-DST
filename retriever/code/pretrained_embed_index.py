from transformers import AutoTokenizer, AutoModel
import argparse
import torch
import torch.nn.functional as F
import json, os
import numpy as np
from tqdm import tqdm

# Embedding model

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='sentence-transformers/all-mpnet-base-v2')
parser.add_argument('--save_name', type=str, default='all_mpnet_base_v2')
parser.add_argument('--separate_embed', action="store_true")
args = parser.parse_args()

MODEL_NAME = args.model_name
SAVE_NAME = args.save_name

# ------ Configuration ends here ----------------


# path to save indexes and results
save_path = f"../expts/{SAVE_NAME}"
os.makedirs(save_path, exist_ok = True) 

DEVICE = torch.device("cuda:0")
CLS_Flag = False

def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)


# function for embedding one string
def embed_single_sentence(sentence, cls=CLS_Flag):

    # Sentences we want sentence embeddings for
    sentences = [sentence]

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True,
                          truncation=True, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(DEVICE)
    attention_mask = encoded_input['attention_mask'].to(DEVICE)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(input_ids, attention_mask)

    # Perform pooling
    sentence_embeddings = None

    if cls:
        sentence_embeddings = model_output[0][:,0,:]
    else:
        sentence_embeddings = mean_pooling(model_output, attention_mask)

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def read_MW_dataset(mw_json_fn):

    # only care domain in test
    DOMAINS = ['hotel', 'restaurant', 'attraction', 'taxi', 'train']

    with open(mw_json_fn, 'r') as f:
        data = json.load(f)

    dial_dict = {}
    sys_dict = {}
    usr_dict = {}

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

        history = f"[system] {sys_utt} [user] {usr_utt}"

        # store the history in dictionary
        name = f"{turn['ID']}_turn_{turn['turn_id']}"
        dial_dict[name] = history

        if args.separate_embed:
            # separate embedding
            sys_hist = f"[system] {sys_utt}" if sys_utt != '' else ""
            usr_hist = f"[user] {usr_utt}" if usr_utt != '' else ""

            sys_dict[name] = sys_hist
            usr_dict[name] = usr_hist
        
    return dial_dict, sys_dict, usr_dict

if __name__ == "__main__":

    def store_embed(input_dataset, output_filename, forward_fn):
        outputs = {}
        with torch.no_grad():
            # k: f"{turn['ID']}_turn_{turn['turn_id']}"
            # v: history = f"[system] {sys_utt} [user] {usr_utt}" (current user and system turn)
            for k, v in tqdm(input_dataset.items()):
                outputs[k] = forward_fn(v).detach().cpu().numpy()
        np.save(output_filename, outputs)
        return
    
    if not args.separate_embed:
        mw_train,_,_ = read_MW_dataset("../../data/mw21_100p_train.json")
        mw_dev,_,_ = read_MW_dataset("../../data/mw21_100p_dev.json")
        mw_test,_,_ = read_MW_dataset("../../data/mw21_100p_test.json")
        print("Finish reading data")

        # store the embeddings
        # 각 데이터셋에 해당하는 turn들의 current user and system으로 embedding을 만들어 저장
        store_embed(mw_train, f"{save_path}/mw21_train_{SAVE_NAME}.npy",
                    embed_single_sentence)
        store_embed(mw_dev, f"{save_path}/mw21_dev_{SAVE_NAME}.npy",
                    embed_single_sentence)
        store_embed(mw_test, f"{save_path}/mw21_test_{SAVE_NAME}.npy",
                    embed_single_sentence)
    
    else:
        mw_train, mw_train_sys, mw_train_usr = read_MW_dataset("../../data/mw21_100p_train.json")
        mw_dev, mw_dev_sys, mw_dev_usr = read_MW_dataset("../../data/mw21_100p_dev.json")
        mw_test, mw_test_sys, mw_test_usr = read_MW_dataset("../../data/mw21_100p_test.json")
        print("Finish reading data")


        store_embed(mw_train, f"{save_path}/mw21_train_{SAVE_NAME}.npy",
                    embed_single_sentence)
        store_embed(mw_dev, f"{save_path}/mw21_dev_{SAVE_NAME}.npy",
                    embed_single_sentence)
        store_embed(mw_test, f"{save_path}/mw21_test_{SAVE_NAME}.npy",
                    embed_single_sentence)

        # sys
        store_embed(mw_train_sys, f"{save_path}/mw21_train_sys_{SAVE_NAME}.npy",
                    embed_single_sentence)
        store_embed(mw_dev_sys, f"{save_path}/mw21_dev_sys_{SAVE_NAME}.npy",
                    embed_single_sentence)
        store_embed(mw_test_sys, f"{save_path}/mw21_test_sys_{SAVE_NAME}.npy",
                    embed_single_sentence)

        # usr
        store_embed(mw_train_usr, f"{save_path}/mw21_train_usr_{SAVE_NAME}.npy",
                    embed_single_sentence)
        store_embed(mw_dev_usr, f"{save_path}/mw21_dev_usr_{SAVE_NAME}.npy",
                    embed_single_sentence)
        store_embed(mw_test_usr, f"{save_path}/mw21_test_usr_{SAVE_NAME}.npy",
                    embed_single_sentence)

    print("Finish Embedding data")