import os
import json
import random
import time
import argparse
from tqdm import tqdm
import copy
import re

from prompt.our_prompting import conversion
from api_request.gpt35_turbo_completion import gpt35_turbo_completion
from utils.helper import SpeedLimitTimer

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('--train_fn', type=str, default= "data/mw21_5p_train_v2.json")
parser.add_argument('--output_file_name', type=str, default="debug", help="filename to save running log and configs")
parser.add_argument('--output_dir', type=str, default="./ett_replace/", help="dir to save running log and configs")
parser.add_argument('--span_info_dir', type=str, help="data.json from MULTIWOZ repo including span info")
parser.add_argument('--mwz_ver', type=str, default="2.4", choices=['2.1', '2.4'], help="version of MultiWOZ")
parser.add_argument('--specific_name', type=str, default="base")
parser.add_argument('--sample_rate', type=float, default=1.0, help='sampling rate of dataset to paraphrase')
args = parser.parse_args()

cur_time = time.strftime('%y%m%d_%H%M-')

args.output_dir = 'ett_replace/' + cur_time + str(args.sample_rate) + "-" + args.output_file_name
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

class EntityDialogue:
    def __init__(self, train_fn, output_dir, specific_name, mwz_ver='2.4'):
        self.data_path = train_fn
        self.output_path = output_dir
        self.specific_name = specific_name
        self.mwz_ver = mwz_ver

        self.ontology_path = "./data/mwz2.1/ontology.json" if self.mwz_ver == '2.1' else "./data/mwz2.4/ontology.json"

        with open(self.data_path,'r') as f:
            self.dataset = json.load(f)

        with open(self.ontology_path,'r') as f:
            self.ontology = json.load(f)

            if 'hospital-department' in self.ontology:
                self.ontology.pop('hospital-department')

        # For span info
        with open(args.span_info_dir,'r') as f:
            self.original_dataset = json.load(f)

    def get_span_info(self):
        span_dataset = []
        for data_idx, data_item in enumerate(tqdm(self.dataset)):
            data_item['span_dialog'] = {}
            data_item['span_dialog']['sys'] = [["",[]]]
            data_item['span_dialog']['usr'] = []

            for dialog_idx, dialog in enumerate(self.original_dataset[data_item['ID']]['log']):
                if data_item['turn_id'] * 2 + 1 <= dialog_idx:
                    break

                # utterance
                span_utt = dialog['text']

                span_list = []
                # states and span info
                if dialog['span_info'] != []:
                    for state in dialog['span_info']:
                        temp = {}
                        temp["domain"] = state[0].split('-')[0]
                        temp["act"] = state[0].split('-')[1]
                        temp["slot"] = state[1]
                        temp["value"] = state[2]
                        temp["span_idx"] = [state[3], state[4]]
                        
                        span_list.append(temp)

                if dialog_idx % 2 == 1:
                    data_item['span_dialog']['sys'].append([span_utt, span_list])
                else:
                    data_item['span_dialog']['usr'].append([span_utt, span_list])

            span_dataset.append(data_item)

            data_item['ett_dialog'] = {}
            data_item['ett_dialog']['sys'] = []
            data_item['ett_dialog']['usr'] = []

            def tokenize_for_span(text):
                text = re.sub("/", " / ", text)
                text = re.sub("\-", " \- ", text)
                # text = re.sub("Im", "I\'m", text)
                # text = re.sub("im", "i\'m", text)
                # text = re.sub("theres", "there's", text)
                # text = re.sub("dont", "don't", text)
                # text = re.sub("whats", "what's", text)
                text = re.sub(r"\bIm\b", "I'm", text)
                text = re.sub(r"\bim\b", "i'm", text)
                text = re.sub(r"\btheres\b", "there's", text)
                text = re.sub(r"\bdont\b", "don't", text)
                text = re.sub(r"\bwhats\b", "what's", text)
            #     text = re.sub("[0-9]:[0-9]+\. ", "[0-9]:[0-9]+ \. ", text)
            #     text = re.sub("[a-z]\.[A-Z]", "[a-z]\. [A-Z]", text)
            #     text = re.sub("\t:[0-9]+", "\t: [0-9]+", text)
                text = re.sub(r"([0-9]:[0-9]+)\. ", r"\1. ", text)
                text = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", text)
                text = re.sub(r"\t:([0-9]+)", r"\t: \1", text)
                tokens = word_tokenize(text)
                return tokens

            # remove one of dialog spans that they have same span idx.
            # because we replace values with special tokens, so remove error cases.
            # (ignore the cases when span_idx of i is subset of span_idx of j ex) i = [1,3] ,j = [1,4])
            for k, v in data_item['span_dialog'].items():
                for dialog, dialog_span in v:
                    sp_list = []
                    for state in dialog_span:
                        sp_list.append(state['span_idx'])
                    
                    index_dict = {}
                    for i,sublist in enumerate(sp_list):
                        value = tuple(sublist)  # Convert sublist to a tuple for hashability
                        if value not in index_dict:
                            index_dict[value] = [i]
                        else:
                            index_dict[value].append(i)

                    for value, indices in index_dict.items():
                        if len(indices) > 1:
                            # for j in indices:
                            #     print(dialog_span[j])
                            for ind in reversed(indices[1:]):
                                del dialog_span[ind]


            # In dialog, change states value into entity ex) 79 -> [ATTRACTION-CHOICE]
            for k, v in data_item['span_dialog'].items():
                for dialog, dialog_span in v:
                    word_index = tokenize_for_span(dialog)
                    pop_list = []
                    for state in dialog_span:
                        # Due to the popping, offset index
                        for p in pop_list:
                            if state['span_idx'][0] > p['start_idx']:
                                state['span_idx'][0] -= p['n_pop']
                                state['span_idx'][1] -= p['n_pop']
                        
                        n_pop = 0
                        for i in reversed(range(state['span_idx'][0]+1,state['span_idx'][1]+1)):
                            word_index.pop(i)
                            n_pop += 1
                        temp = {}
                        temp['start_idx'] = state['span_idx'][0]
                        temp['n_pop'] = n_pop
                        pop_list.append(temp)
                        # print(dialog)
                        # print(state['span_idx'][0])
                        # print(state['span_idx'][1])
                        # print(f"[{state['domain'].upper()}-{state['slot'].upper()}]")
                        word_index[state['span_idx'][0]] = f"[{state['domain'].upper()}-{state['slot'].upper()}]"
                        # print(word_index)
                        # print()
                    data_item['ett_dialog'][k].append(' '.join(word_index))

        with open(os.path.join(args.output_dir, 'train_with_span_info.json'),'w') as f:
            json.dump(span_dataset, f, indent = 4)


    def replace_ett(self):
        timer = SpeedLimitTimer(second_per_step=4)

        all_result = []
        sampled_dataset = random.sample(self.dataset, int(len(self.dataset) * args.sample_rate))
        for data_idx, data_item in enumerate(tqdm(sampled_dataset)):
            prompt_text = "# dialogue\n"

            last_slot_values = {s: v.split(
                '|')[0] for s, v in data_item['last_slot_values'].items()}
            # prompt_text += f"[context] {conversion(', '.join({f'({slot} = {value})' for slot, value in last_slot_values.items()}))}\n"

            sys_exist = True
            if data_item['dialog']['sys'][-1] == "":
                sys_exist = False
            
            last_sys_utt = data_item['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[system] {last_sys_utt}\n"
            prompt_text += f"[user] {data_item['dialog']['usr'][-1]}\n\n"

            # slot_list = []
            # for slot in list(self.ontology.keys()):
            #     if slot.split('-')[0] in data_item['domains']:
            #         slot_list.append(slot)

            # prompt_text += f"dialogue Entity (slot_name): {', '.join(slot_list)}. \n"

            if data_item['turn_slot_values']:
                prompt_text += f"dialogues Entity (slot_name = value): {conversion(', '.join({f'({slot} = {value})' for slot, value in data_item['turn_slot_values'].items()}))}.\n"

            # prompt_text += f", {conversion(', '.join({f'({slot} = {value})' for slot, value in last_slot_values.items()}))}\n"

            # prompt_text += f"Your job is to do Named Entity Recognition (NER) of the dialogue based on the dialogue entity.\n"
            # prompt_text += f"Your job is to find corresponding value of entity based on the dialogue entity.\n"
            prompt_text += f"mask corresponding entity's value with slot_name including bracket '[]'. masked dialogue:"

            # if sys_exist:
            #     prompt_text += f" [system] and"
            # prompt_text += f" [user] prefix." 
            
            # if sys_exist:    
            #     prompt_text += f"(You should generate the [system] first, then [user]. Also, [system] and [user] should be one, respectively.)"
            # else:
            #     prompt_text += f"([user] should be one.)"

            # # rearrange the order of information presented
            # prompt_text += f" In addition, if possible, try to rearrange the order of information in each [system] and [user]. Don't generate the continuing dialogues."

            completion = ""
            complete_flag = False
            while not complete_flag:
                try:
                    completion = gpt35_turbo_completion(prompt_text)
                except Exception as e:
                    print(e)
                    timer.sleep(10)
                else:
                    complete_flag = True
                    timer.step()    

            print(prompt_text)
            print("\n")
            print(completion)
 
            # To filter unnecessary dialogue, extract first two line from completion.
            temp = []
            for line in completion.split("\n"):
                if "]" in line:
                    temp.append(line)

            # completion = '\n'.join(temp[:2]) if sys_exist else '\n'.join(temp[:1])
            completion = '\n'.join(temp[:2])

            sys_utt = completion.split("[user]")[0].replace("[system]","").strip()
            usr_utt = completion.split("[user]")[1].strip()

            # change directory temporarily
            cur_dir = os.getcwd()
            os.chdir('data')
            from data.create_data import normalize
            sys_utt = normalize(sys_utt, clean_value=False)
            usr_utt = normalize(usr_utt, clean_value=False )
            os.chdir(cur_dir)
            print(completion)

            print(f"sys_utt: {sys_utt}")
            print(f"usr_utt: {usr_utt}")

            # data_item['ID'] = f"{data_item['ID'].split('.')[0]}-{args.specific_name}.json"
            # # save original
            # data_item["original_sys"] = data_item['dialog']['sys'][-1]
            # data_item["original_usr"] = data_item['dialog']['usr'][-1]
            # # override augmented
            # data_item['dialog']['sys'][-1] = sys_utt
            # data_item['dialog']['usr'][-1] = usr_utt

            # print()
            # print(f"sys_usr: {data_item['dialog']['sys'][-1]}")
            # print(f"usr_usr: {data_item['dialog']['usr'][-1]}")
            # print("\n\n\n")

            # para_result.append(data_item)

            # if data_idx % 5 == 0:
            #     with open(os.path.join(args.output_dir,f'para_log.json'),'w') as f:
            #         json.dump(para_result, f, indent=4)                  
   

if __name__ == "__main__":

    ett = EntityDialogue(args.train_fn, args.output_dir, args.specific_name)

    # ett.replace_ett()

    ett.get_span_info()

    # p = """# dialogue
    # [system] my favorite it the copper kettle at 4 kings parade city centre cb21sj . it serves british food . does that interest you ?
    # [user] absolutely ! thank you ! i also need information on the attractions that have multiple sports in town , in the same area as the restaurant please .
    
    # dialogues Entity (slot_name = value): (restaurant-name = copper kettle), (attraction-type = multiple sports), (attraction-area = centre), (restaurant-pricerange = moderate), (attraction-type = multiple sports), (restaurant-area = centre), (restaurant-food_type = dontcare)
    # Your job is to do Named Entity Recognition (NER) of the dialogue based on the dialogue entity.

    # After that, replace corresponding entity's value with slot_name with bracket '[]'. (remove value after replacement)"""

    # p1 = """# dialogue
    # [system] the only multiple sports attraction is located in the east of town . would you like more information ?
    # [user] no , i want to be in the centre of town . what about architecture attractions ?
    
    # dialogues Entity (slot_name = value): (attraction-type = architecture), (restaurant-food_type = dontcare), (restaurant-area = centre), (restaurant-pricerange = moderate), (attraction-area = centre), (attraction-type = multiple sports)
    # Your job is to do Named Entity Recognition (NER) of the dialogue based on the dialogue entity.

    # After that, replace corresponding entity's value with slot_name with bracket '[]'. (remove value after replacement)"""

    # p2 = """# dialogue
    # [system] its entrance fee is free .
    # [user] i also need to book a taxi between the restaurant and the church .
    
    # dialogues Entity (slot_name = value): (taxi-destination = all saints church), (taxi-departure = dojo noodle bar), (restaurant-name = dojo noodle bar), (attraction-name = all saints church)
    # Your job is to do Named Entity Recognition (NER) of the dialogue based on the dialogue entity.

    # After that, replace corresponding entity's value with slot_name with bracket '[]'. (remove value after replacement)"""

    # print(gpt35_turbo_completion(p2))