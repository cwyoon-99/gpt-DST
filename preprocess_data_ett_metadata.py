import os
import json
import random
import time
import argparse
from tqdm import tqdm
import copy
import re
from config import CONFIG

from prompt.our_prompting import conversion
from api_request.gpt35_turbo_completion import gpt35_turbo_completion
from utils.helper import SpeedLimitTimer

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('--train_fn', type=str, default= "data/mw21_5p_train_v2.json")
parser.add_argument('--dev_fn', type=str)
parser.add_argument('--test_fn', type=str)
parser.add_argument('--output_file_name', type=str, default="debug", help="filename to save running log and configs")
parser.add_argument('--output_dir', type=str, default="./ett_metadata/", help="dir to save running log and configs")
parser.add_argument('--span_info_dir', type=str, help="data.json from MULTIWOZ repo including span info")
parser.add_argument('--mwz_ver', type=str, default="2.4", choices=['2.1', '2.4'], help="version of MultiWOZ")
parser.add_argument('--specific_name', type=str, default="base")

parser.add_argument('--train_ett_replaced_fn', type=str)
parser.add_argument('--dev_ett_replaced_fn', type=str)
parser.add_argument('--test_ett_replaced_fn', type=str)

args = parser.parse_args()

if args.mwz_ver == '2.1':
    ontology_path = CONFIG["ontology_21"]
    args.dev_fn = "./data/mw21_100p_dev.json"
    args.test_fn = "./data/mw21_100p_test.json"
else:
    ontology_path = CONFIG["ontology_24"]
    args.dev_fn = "./data/mw24_100p_dev.json"
    args.test_fn = "./data/mw24_100p_test.json"

cur_time = time.strftime('%y%m%d_%H%M-')

args.output_dir += f"{cur_time}-{args.output_file_name}"
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

ett_list = ['TAXI-LEAVE', 'TAXI-CAR', 'TRAIN-PEOPLE', 'HOTEL-STAY', 'RESTAURANT-PEOPLE', 
                'ATTRACTION-TYPE', 'RESTAURANT-REF', 'HOTEL-TYPE', 'HOTEL-ADDR', 'TRAIN-LEAVE', 
                'TAXI-DEST', 'BOOKING-PEOPLE', 'TAXI-DEPART', 'BOOKING-TIME', 'HOTEL-REF', 'HOTEL-DAY', 
                'RESTAURANT-PHONE', 'ATTRACTION-NAME', 'RESTAURANT-POST', 'ATTRACTION-PHONE', 'ATTRACTION-PRICE', 
                'TAXI-ARRIVE', 'TRAIN-TIME', 'RESTAURANT-CHOICE', 'TRAIN-DEST', 'TAXI-PHONE', 'BOOKING-STAY', 
                'RESTAURANT-PRICE', 'TRAIN-DEPART', 'RESTAURANT-NAME', 'RESTAURANT-ADDR', 'BOOKING-REF', 
                'HOTEL-PEOPLE', 'RESTAURANT-FOOD', 'HOTEL-PRICE', 'HOTEL-PHONE', 'RESTAURANT-TIME', 
                'HOTEL-AREA', 'TRAIN-DAY', 'TRAIN-CHOICE', 'TRAIN-REF', 'TRAIN-ARRIVE', 'RESTAURANT-AREA', 
                'RESTAURANT-DAY', 'ATTRACTION-POST', 'HOTEL-POST', 'ATTRACTION-FEE', 'ATTRACTION-ADDR', 
                'HOTEL-STARS', 'HOTEL-NAME', 'ATTRACTION-AREA', 'HOTEL-CHOICE', 'BOOKING-NAME', 'TRAIN-ID', 
                'BOOKING-DAY', 'ATTRACTION-CHOICE', 'TRAIN-TICKET']


def add_ett_metadata(tg_fn, refer_fn):
    with open(tg_fn,'r') as f:
        tg_dataset = json.load(f)

    with open(refer_fn,'r') as f:
        refer_dataset = json.load(f)

    if len(tg_dataset) != len(refer_dataset):
        raise Exception("tg_dataset doesn't match with refer_dataset")

    all_results = []

    for tg_item, refer_item in zip(tg_dataset, refer_dataset):

        for speaker in ['sys', 'usr']:
            refer_utt = refer_item['dialog'][speaker][-1]
            ett_sublist = []

            for token in refer_utt.split(" "):
                for ett in ett_list:
                    if ett == token:
                        ett_sublist.append(ett)
                        continue

            tg_item[f"{speaker}_entity"] = ett_sublist

        all_results.append(tg_item)
    
    with open(os.path.join(args.output_dir, f"{os.path.basename(tg_fn).split('.json')[0]}_{args.output_file_name}.json"), 'w') as f:
        json.dump(all_results, f, indent=4)


def add_ett_metadata_from_span_info(tg_fn, refer_fn):
    with open(tg_fn,'r') as f:
        tg_dataset = json.load(f)

    with open(refer_fn,'r') as f:
        refer_dataset = json.load(f)

    if len(tg_dataset) != len(refer_dataset):
        raise Exception("tg_dataset doesn't match with refer_dataset")

    all_results = []

    for tg_item, refer_item in zip(tg_dataset, refer_dataset):

        for speaker in ['sys', 'usr']:
            refer_utt = refer_item['span_dialog'][speaker][-1][0]

            refer_span_info = refer_item['span_dialog'][speaker][-1][1]

            ett_sublist = []

            for span_info_dict in refer_span_info:
                ett_sublist.append((f"{span_info_dict['domain']}-{span_info_dict['slot']}").upper())

            # Exception for hotel-parking, hotel-internet, dontcare value
            if speaker == 'usr':
                for k, v in refer_item['turn_slot_values'].items():
                    if k == "hotel-parking" or k == "hotel-internet" or v == "dontcare":
                        ett_sublist.append(k.upper())

            tg_item[f"{speaker}_entity"] = ett_sublist

        all_results.append(tg_item)
    
    with open(os.path.join(args.output_dir, f"{os.path.basename(tg_fn).split('.json')[0]}_{args.output_file_name}.json"), 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    add_ett_metadata(args.train_fn, args.train_ett_replaced_fn)
    add_ett_metadata(args.dev_fn, args.dev_ett_replaced_fn)
    add_ett_metadata(args.test_fn, args.test_ett_replaced_fn)

    # add_ett_metadata_from_span_info(args.train_fn, args.train_ett_replaced_fn)
    # add_ett_metadata_from_span_info(args.dev_fn, args.dev_ett_replaced_fn)
    # add_ett_metadata_from_span_info(args.test_fn, args.test_ett_replaced_fn)