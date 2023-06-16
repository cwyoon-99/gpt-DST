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
parser.add_argument('--output_dir', type=str, default="./ett_replace/", help="dir to save running log and configs")
parser.add_argument('--span_info_dir', type=str, help="data.json from MULTIWOZ repo including span info")
parser.add_argument('--mwz_ver', type=str, default="2.4", choices=['2.1', '2.4'], help="version of MultiWOZ")
parser.add_argument('--specific_name', type=str, default="base")
parser.add_argument('--sample_rate', type=float, default=1.0, help='sampling rate of dataset to paraphrase')
parser.add_argument('--train_ner_fn', type=str, help = 'NER result of train_fn')
parser.add_argument('--dev_ner_fn', type=str, help = 'NER result of dev_fn')
parser.add_argument('--test_ner_fn', type=str, help = 'NER result of test_fn')
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

args.output_dir = 'ett_replace/' + cur_time + str(args.sample_rate) + "-" + args.output_file_name
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "exp_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

class EntityDialogue:
    def __init__(self, train_fn, output_dir, specific_name=None):
        self.data_path = train_fn
        self.output_path = output_dir
        self.specific_name = specific_name

        self.ontology_path = ontology_path

        with open(self.data_path,'r') as f:
            self.dataset = json.load(f)

        with open(self.ontology_path,'r') as f:
            self.ontology = json.load(f)

            if 'hospital-department' in self.ontology:
                self.ontology.pop('hospital-department')

        with open(args.test_fn,'r') as f:
            self.test_dataset = json.load(f)

    def get_span_info(self):

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
            # text = re.sub("[0-9]:[0-9]+\. ", "[0-9]:[0-9]+ \. ", text)
            # text = re.sub("[a-z]\.[A-Z]", "[a-z]\. [A-Z]", text)
            # text = re.sub("\t:[0-9]+", "\t: [0-9]+", text)
            text = re.sub(r"([0-9]:[0-9]+)\. ", r"\1 . ", text)
            text = re.sub(r"([a-z])\.([A-Z])", r"\1. \2", text)
            text = re.sub(r"\t:([0-9]+)", r"\t: \1", text)
            tokens = word_tokenize(text)
            return tokens
        
        def search_nested_dict(data, value):
            for key, val in data.items():
                if isinstance(val, str):
                    # handle exceptional case
                    # lower
                    t_value = value.lower()
                    t_val = val.lower()

                    # remove "'" and " "
                    t_value = t_value.translate({ord(letter): None for letter in "' "})
                    t_val = t_val.translate({ord(letter): None for letter in "' "})

                    # value subset
                    if t_value and t_value == t_val:
                        return True

                if isinstance(val, dict):
                    if search_nested_dict(val, value):
                        return True
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            if search_nested_dict(item, value):
                                return True
            return False

        # For span info
        with open(args.span_info_dir,'r') as f:
            original_dataset = json.load(f)

        span_dataset = []
        count_change = 0
        count_unchange = 0
        for data_idx, data_item in enumerate(tqdm(self.dataset)):
            data_item['span_dialog'] = {}
            data_item['span_dialog']['sys'] = [["",[]]]
            data_item['span_dialog']['usr'] = []

            for dialog_idx, dialog in enumerate(original_dataset[data_item['ID']]['log']):
                # original_dataset has dicts that include metadata of all turns of each dialogue. 
                # Therefore, we need to break the for-loop before we meet out-of-range index.
                if data_item['turn_id'] * 2 + 1 <= dialog_idx:
                    break

                # utterance
                span_utt = dialog['text']

                span_list = []
                # states and span info
                if "span_info" in dialog:
                    for state in dialog['span_info']:
                        temp = {}
                        temp["domain"] = state[0].split('-')[0]
                        temp["act"] = state[0].split('-')[1]
                        temp["slot"] = state[1]
                        temp["value"] = state[2]
                        temp["span_idx"] = [state[3], state[4]]

                        # convert Booking domain into correspoding domains.
                        if temp["domain"] == 'Booking':

                            for domain, data in dialog['metadata'].items():
                                if isinstance(data, dict):
                                    if search_nested_dict(data, temp['value']):
                                        temp['domain'] = domain.capitalize()
                        
                        if temp["domain"] == 'Booking':
                            print(dialog['text'])
                            print(temp)
                            print()
                            count_unchange += 1
                        else:
                            count_change += 1

                        span_list.append(temp)

                if dialog_idx % 2 == 1:
                    data_item['span_dialog']['sys'].append([span_utt, span_list])
                else:
                    data_item['span_dialog']['usr'].append([span_utt, span_list])

            span_dataset.append(data_item)

            data_item['ett_dialog'] = {}
            data_item['ett_dialog']['sys'] = []
            data_item['ett_dialog']['usr'] = []

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

                    rm_idx = []
                    for value, indices in index_dict.items():
                        if len(indices) > 1:
                            for ind in reversed(indices[1:]):
                                rm_idx.append(ind)

                    for i in reversed(sorted(rm_idx)):
                        del dialog_span[i]

            # In dialog, change states value into entity ex) 79 -> [ATTRACTION-CHOICE]
            # unpop version.
            for k, v in data_item['span_dialog'].items():
                for dialog, dialog_span in v:
                    word_index = tokenize_for_span(dialog)
                    for state in dialog_span:                        
                        for i in reversed(range(state['span_idx'][0],state['span_idx'][1]+1)):
                            # print(word_index)
                            # print(state['span_idx'][0])
                            # print(state['span_idx'][1])
                            try:
                                word_index[i] = f"[{state['domain'].upper()}-{state['slot'].upper()}]"
                            except:
                                continue
                    data_item['ett_dialog'][k].append(' '.join(word_index))


        print(f"count change: {count_change}")
        print(f"count unchange: {count_unchange}")

        # make additional path
        os.makedirs(os.path.join(args.output_dir, 'data'), exist_ok=True)

        with open(os.path.join(args.output_dir, self.data_path),'w') as f:
            json.dump(span_dataset, f, indent = 4)

        return span_dataset

    # def replace_dialog(self):
    #     with open(args.train_ner_fn, 'r') as f:
    #         ner_dataset = json.load(f)

    #     all_result = []
    #     # Replaced values with entities
    #     for data_idx, data_item in enumerate(tqdm(self.dataset)):
            
    #         ner_item = copy.deepcopy(data_item)
    #         ner_item['ID'] = f"{data_item['ID']}-{args.specific_name}"
    #         # change all dialogs into modified
    #         # ner_item['origin_dialog'] = data_item['dialog']
    #         # change only current dialogs
    #         ner_item['origin_dialog'] = data_item['dialog']

    #         for speaker in ['sys','usr']:
    #             ner_id = f"{data_item['ID']}-{data_item['turn_id']}-{speaker}"
    #             ett_info = ner_dataset[ner_id]

    #             # apply attention mask to output_tokens
    #             masked_output_tokens = [token if mask == 1 else "O" for token, mask in zip(ett_info['output_token'], ett_info['attention_mask'])]

    #             # remove duplicated entities and replace corresponding values with entities
    #             prev_token = None
    #             ett_replaced_dialog = []
    #             for i, token in enumerate(masked_output_tokens):
    #                 if token != 'O':
    #                     if prev_token == None or token != prev_token:
    #                             ett_replaced_dialog.append(token)
    #                             prev_token = token
    #                     continue

    #                 ett_replaced_dialog.append(ett_info['input'][i])
    #                 prev_token = token

    #             # remove [CLS] and [SEP]
    #             ett_replaced_dialog = ett_replaced_dialog[1:-1]

    #             # save lower case string
    #             ner_item['dialog'][speaker][-1] = ' '.join(ett_replaced_dialog).lower()

    #         all_result.append(ner_item)

    #     # train data를 entity dialogue 로 교체하여 저장. entity retriever finetuning 에 사용됨.
    #     with open(os.path.join(args.output_dir,f'mw21_train_5p_v2_ett.json'),'w') as f:
    #         json.dump(all_result, f, indent=4)


    def replace_dialog_turn_split(self):
        with open(args.train_ner_fn, 'r') as f:
            train_ner_dataset = json.load(f)

        with open(args.dev_ner_fn, 'r') as f:
            dev_ner_dataset = json.load(f)

        with open(args.test_ner_fn, 'r') as f:
            test_ner_dataset = json.load(f)

        all_result = []
        # Replaced values with entities
        for data_idx, data_item in enumerate(tqdm(self.dataset)):
            
            ner_item = copy.deepcopy(data_item)
            ner_item['ID'] = f"{data_item['ID']}-{args.specific_name}"
            # change all dialogs into modified
            # ner_item['origin_dialog'] = data_item['dialog']
            # change only current dialogs
            ner_item['origin_dialog'] = data_item['dialog']

            ner_id = f"{data_item['ID']}-{data_item['turn_id']}"

            if ner_id in train_ner_dataset:
                ner_dataset = train_ner_dataset
            elif ner_id in dev_ner_dataset:
                ner_dataset = dev_ner_dataset
            else:
                ner_dataset = test_ner_dataset

            ett_info = ner_dataset[ner_id]

            # apply attention mask to output_tokens
            masked_output_tokens = [token if mask == 1 else "O" for token, mask in zip(ett_info['output_token'], ett_info['attention_mask'])]

            # remove duplicated entities and replace corresponding values with entities
            prev_token = None
            ett_replaced_dialog = []
            for i, token in enumerate(masked_output_tokens):
                if token != 'O':
                    if prev_token == None or token != prev_token:
                            ett_replaced_dialog.append(token)
                            prev_token = token
                    continue

                ett_replaced_dialog.append(ett_info['input'][i])
                prev_token = token

            # remove [CLS] and [SEP]
            ett_replaced_dialog = ett_replaced_dialog[1:-1]

            sys_s_idx = 0
            for i, token in enumerate(ett_replaced_dialog):
                if token == "system" and ett_replaced_dialog[i+1] == ":":
                        sys_s_idx = i + 2
                        break
            
            usr_s_idx = 0
            for i, token in enumerate(ett_replaced_dialog):
                if token == "user" and ett_replaced_dialog[i+1] == ":":
                        usr_s_idx = i + 2
                        break

            # Extract the system and user strings
            sys_utt = ' '.join(ett_replaced_dialog[sys_s_idx:usr_s_idx - 2])
            usr_utt = ' '.join(ett_replaced_dialog[usr_s_idx:])

            ## Bert adds special '##' to sub tokens. so we have to remove it to make general sentences.
            sys_utt = sys_utt.replace(' ##','')
            usr_utt = usr_utt.replace(' ##','')

            # save
            ner_item['dialog']['sys'][-1] = sys_utt
            ner_item['dialog']['usr'][-1] = usr_utt

            all_result.append(ner_item)

        file_name = os.path.basename(self.data_path).split(".")[0]

        # train data를 entity dialogue 로 교체하여 저장. entity retriever finetuning 에 사용됨.
        with open(os.path.join(args.output_dir,f'{file_name}_{args.output_file_name}.json'),'w') as f:
            json.dump(all_result, f, indent=4)


    # def replace_ett(self):
    #     timer = SpeedLimitTimer(second_per_step=4)

    #     all_result = []
    #     sampled_dataset = random.sample(self.dataset, int(len(self.dataset) * args.sample_rate))
    #     for data_idx, data_item in enumerate(tqdm(sampled_dataset)):
    #         prompt_text = "# dialogue\n"

    #         last_slot_values = {s: v.split(
    #             '|')[0] for s, v in data_item['last_slot_values'].items()}
    #         # prompt_text += f"[context] {conversion(', '.join({f'({slot} = {value})' for slot, value in last_slot_values.items()}))}\n"

    #         sys_exist = True
    #         if data_item['dialog']['sys'][-1] == "":
    #             sys_exist = False
            
    #         last_sys_utt = data_item['dialog']['sys'][-1]
    #         if last_sys_utt == 'none':
    #             last_sys_utt = ''
    #         prompt_text += f"[system] {last_sys_utt}\n"
    #         prompt_text += f"[user] {data_item['dialog']['usr'][-1]}\n\n"

    #         # slot_list = []
    #         # for slot in list(self.ontology.keys()):
    #         #     if slot.split('-')[0] in data_item['domains']:
    #         #         slot_list.append(slot)

    #         # prompt_text += f"dialogue Entity (slot_name): {', '.join(slot_list)}. \n"

    #         if data_item['turn_slot_values']:
    #             prompt_text += f"dialogues Entity (slot_name = value): {conversion(', '.join({f'({slot} = {value})' for slot, value in data_item['turn_slot_values'].items()}))}.\n"

    #         # prompt_text += f", {conversion(', '.join({f'({slot} = {value})' for slot, value in last_slot_values.items()}))}\n"

    #         # prompt_text += f"Your job is to do Named Entity Recognition (NER) of the dialogue based on the dialogue entity.\n"
    #         # prompt_text += f"Your job is to find corresponding value of entity based on the dialogue entity.\n"
    #         prompt_text += f"mask corresponding entity's value with slot_name including bracket '[]'. masked dialogue:"

    #         # if sys_exist:
    #         #     prompt_text += f" [system] and"
    #         # prompt_text += f" [user] prefix." 
            
    #         # if sys_exist:    
    #         #     prompt_text += f"(You should generate the [system] first, then [user]. Also, [system] and [user] should be one, respectively.)"
    #         # else:
    #         #     prompt_text += f"([user] should be one.)"

    #         # # rearrange the order of information presented
    #         # prompt_text += f" In addition, if possible, try to rearrange the order of information in each [system] and [user]. Don't generate the continuing dialogues."

    #         completion = ""
    #         complete_flag = False
    #         while not complete_flag:
    #             try:
    #                 completion = gpt35_turbo_completion(prompt_text)
    #             except Exception as e:
    #                 print(e)
    #                 timer.sleep(10)
    #             else:
    #                 complete_flag = True
    #                 timer.step()    

    #         print(prompt_text)
    #         print("\n")
    #         print(completion)
 
    #         # To filter unnecessary dialogue, extract first two line from completion.
    #         temp = []
    #         for line in completion.split("\n"):
    #             if "]" in line:
    #                 temp.append(line)

    #         # completion = '\n'.join(temp[:2]) if sys_exist else '\n'.join(temp[:1])
    #         completion = '\n'.join(temp[:2])

    #         sys_utt = completion.split("[user]")[0].replace("[system]","").strip()
    #         usr_utt = completion.split("[user]")[1].strip()

    #         # change directory temporarily
    #         cur_dir = os.getcwd()
    #         os.chdir('data')
    #         from data.create_data import normalize
    #         sys_utt = normalize(sys_utt, clean_value=False)
    #         usr_utt = normalize(usr_utt, clean_value=False )
    #         os.chdir(cur_dir)
    #         print(completion)

    #         print(f"sys_utt: {sys_utt}")
    #         print(f"usr_utt: {usr_utt}")

    #         # data_item['ID'] = f"{data_item['ID'].split('.')[0]}-{args.specific_name}.json"
    #         # # save original
    #         # data_item["original_sys"] = data_item['dialog']['sys'][-1]
    #         # data_item["original_usr"] = data_item['dialog']['usr'][-1]
    #         # # override augmented
    #         # data_item['dialog']['sys'][-1] = sys_utt
    #         # data_item['dialog']['usr'][-1] = usr_utt

    #         # print()
    #         # print(f"sys_usr: {data_item['dialog']['sys'][-1]}")
    #         # print(f"usr_usr: {data_item['dialog']['usr'][-1]}")
    #         # print("\n\n\n")

    #         # para_result.append(data_item)

    #         # if data_idx % 5 == 0:
    #         #     with open(os.path.join(args.output_dir,f'para_log.json'),'w') as f:
    #         #         json.dump(para_result, f, indent=4)               
   

if __name__ == "__main__":

    # First Step.
    # prepare data for NER training.
    # ett = EntityDialogue(args.train_fn, args.output_dir)
    # dev_ett = EntityDialogue(args.dev_fn, args.output_dir)
    # test_ett = EntityDialogue(args.test_fn, args.output_dir)

    # ett.get_span_info()
    # dev_ett.get_span_info()
    # test_ett.get_span_info()

    # Second Step.
    # Fine-tune NER model with those data. (we did fine-tuning in colab env)

    # Third Step
    # Train ner
    ett = EntityDialogue(args.train_fn, args.output_dir, args.specific_name)
    ett.replace_dialog_turn_split()

    # dev ner
    ett = EntityDialogue(args.dev_fn, args.output_dir, args.specific_name)
    ett.replace_dialog_turn_split()

    # Test ner
    ett = EntityDialogue(args.test_fn, args.output_dir, args.specific_name)
    ett.replace_dialog_turn_split()