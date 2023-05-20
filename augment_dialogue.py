import os
import json
import itertools
import random
import copy
import time
import argparse
from tqdm import tqdm

from prompt.our_prompting import conversion
from api_request.gpt35_turbo_completion import gpt35_turbo_completion
from utils.helper import SpeedLimitTimer

parser = argparse.ArgumentParser()
parser.add_argument('--train_fn', type=str, default= "data/mw21_5p_train_v2.json")
parser.add_argument('--output_file_name', type=str, default="debug", help="filename to save running log and configs")
parser.add_argument('--output_dir', type=str, default="./augments/", help="dir to save running log and configs")
parser.add_argument('--mwz_ver', type=str, default="2.4", choices=['2.1', '2.4'], help="version of MultiWOZ")
parser.add_argument('--specific_name', type=str, default="base")
args = parser.parse_args()

cur_time = time.strftime('%y%m%d_%H%M-')

args.output_dir = 'augments/' + cur_time + args.output_file_name
os.makedirs(args.output_dir, exist_ok=True)

with open(os.path.join(args.output_dir, "augment_config.json"), 'w') as f:
    json.dump(vars(args), f, indent=4)

class AugmentDialogue:
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

    def augment_dialogue(self):
        slot_list = []
        for data_item in self.dataset:
            slot_list.append(list(data_item['turn_slot_values'].keys()))


        keys_list = list(self.ontology.keys())
        augment_list = []

        # 요소 2개
        sets_list = list(itertools.combinations(keys_list, 2))
        sets_2 = [set(combination) for combination in sets_list]

        count = 0
        for s in sets_2:
            check = False
            for l in slot_list:
                if s == set(l):
                    check = True
            if not check: # train dataset에 속하지 않는 dialogue states 조합
                domain_name = None
                check_same_domain = True
                for element in s:
                    if domain_name is None:
                        domain_name = element.split('-')[0]
                    elif domain_name != element.split('-')[0]:
                        check_same_domain = False
                if check_same_domain: # 같은 도메인에 속하느 경우
                    augment_list.append(list(s))

        # 조합에 해당하는 데이터 저장
        database = {}
        for slots_combo in augment_list:
            for slot in slots_combo:
                if slot not in database:
                    item_list = []
                    for data_item in self.dataset:
                        if set(data_item['turn_slot_values'].keys()) == set([slot]):
                            item_list.append(data_item)

                    database[slot] = item_list

        timer = SpeedLimitTimer(second_per_step=4)

        # augment
        count = 0
        augmented_result = []
        for slots_combo in augment_list:
            for slot in slots_combo:
                ag_slot = next(x for x in slots_combo if x != slot)

                # augmenting every pairs of slot require excessive amount of api request. so we randomly sample database[slot] to reduce its size by half.
                sampled_database = random.sample(database[slot], len(database[slot]) // 2)
                for data_item in tqdm(sampled_database):
                    print(count)
                    prompt_text = ""

                    last_slot_values = {s: v.split(
                        '|')[0] for s, v in data_item['last_slot_values'].items()}
                    # prompt_text += f"[context] {conversion(', '.join({f'({slot} = {value})' for slot, value in last_slot_values.items()}))}\n"
                    
                    last_sys_utt = data_item['dialog']['sys'][-1]
                    if last_sys_utt == 'none':
                        last_sys_utt = ''
                    prompt_text += f"[system] {last_sys_utt}\n"
                    prompt_text += f"[user] {data_item['dialog']['usr'][-1]}\n"
                    prompt_text += f"Answer: {conversion(', '.join({f'({slot} = {value})' for slot, value in data_item['turn_slot_values'].items()}))}\n"
                    prompt_text += f"Augment the user utterance to additionally have "

                    # random sampling of value from ontology
                    sample_value = random.choice(self.ontology[ag_slot])
                    while '|' in sample_value:
                        sample_value = random.choice(self.ontology[ag_slot])

                    prompt_text += f"({conversion(ag_slot)} = {sample_value}) in answer.\n"
                    prompt_text += f"augmented user utterance:"
                    print(prompt_text)

                    save_data_item = copy.deepcopy(data_item)
                    # To distinguish from original, change ID.
                    save_data_item['ID'] = f"{data_item['ID'].split('.')[0]}-{args.specific_name}-{ag_slot}.json"
                    save_data_item['ag_slot'] = ag_slot

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

                    # change directory temporarily
                    cur_dir = os.getcwd()
                    os.chdir('data')
                    from data.create_data import normalize
                    completion = normalize(completion, clean_value=False)
                    os.chdir(cur_dir)

                    # override augmented
                    save_data_item['dialog']['usr'][-1] = completion
                    save_data_item['turn_slot_values'][ag_slot] = sample_value
                    save_data_item["slot_values"][ag_slot] = sample_value

                    print()
                    print(f"{save_data_item['dialog']['usr'][-1]}")
                    print(f"augmented turn slot values: {save_data_item['turn_slot_values']}")
                    print("\n\n")

                    augmented_result.append(save_data_item)
                    count += 1
                    
                with open(os.path.join(args.output_dir,f'augment_log.json'),'w') as f:
                    json.dump(augmented_result, f, indent=4)

        print(count)


    def augment_dialogue_MD(self):
        slot_list = []
        for data_item in self.dataset:
            slot_list.append(list(data_item['turn_slot_values'].keys()))


        keys_list = list(self.ontology.keys())
        augment_list = []

        # 요소 2개
        sets_list = list(itertools.combinations(keys_list, 2))
        sets_2 = [set(combination) for combination in sets_list]

        count = 0
        for s in sets_2:
            check = False
            for l in slot_list:
                if s == set(l):
                    check = True
            if not check: # train dataset에 속하지 않는 dialogue states 조합
                domain_name = None
                check_same_domain = True
                for element in s:
                    if domain_name is None:
                        domain_name = element.split('-')[0]
                    elif domain_name != element.split('-')[0]:
                        check_same_domain = False
                if check_same_domain: # 같은 도메인에 속하느 경우
                    augment_list.append(list(s))

        # # 조합에 해당하는 데이터 저장
        # database = {}
        # for slots_combo in augment_list:
        #     for slot in slots_combo:
        #         if slot not in database:
        #             item_list = []
        #             for data_item in self.dataset:
        #                 if set(data_item['turn_slot_values'].keys()) == set([slot]):
        #                     item_list.append(data_item)

        #             database[slot] = item_list

        # timer = SpeedLimitTimer(second_per_step=4)

        # # augment
        # count = 0
        # augmented_result = []
        # for slots_combo in augment_list:
        #     for slot in slots_combo:
        #         ag_slot = next(x for x in slots_combo if x != slot)

        #         # augmenting every pairs of slot require excessive amount of api request. so we randomly sample database[slot] to reduce its size by half.
        #         sampled_database = random.sample(database[slot], len(database[slot]) // 2)
        #         for data_item in tqdm(sampled_database):
        #             print(count)
        #             prompt_text = ""

        #             last_slot_values = {s: v.split(
        #                 '|')[0] for s, v in data_item['last_slot_values'].items()}
        #             # prompt_text += f"[context] {conversion(', '.join({f'({slot} = {value})' for slot, value in last_slot_values.items()}))}\n"
                    
        #             last_sys_utt = data_item['dialog']['sys'][-1]
        #             if last_sys_utt == 'none':
        #                 last_sys_utt = ''
        #             prompt_text += f"[system] {last_sys_utt}\n"
        #             prompt_text += f"[user] {data_item['dialog']['usr'][-1]}\n"
        #             prompt_text += f"Answer: {conversion(', '.join({f'({slot} = {value})' for slot, value in data_item['turn_slot_values'].items()}))}\n"
        #             prompt_text += f"Augment the user utterance to additionally have "

        #             # random sampling of value from ontology
        #             sample_value = random.choice(self.ontology[ag_slot])
        #             while '|' in sample_value:
        #                 sample_value = random.choice(self.ontology[ag_slot])

        #             prompt_text += f"({conversion(ag_slot)} = {sample_value}) in answer.\n"
        #             prompt_text += f"augmented user utterance:"
        #             print(prompt_text)

        #             save_data_item = copy.deepcopy(data_item)
        #             # To distinguish from original, change ID.
        #             save_data_item['ID'] = f"{data_item['ID'].split('.')[0]}-{args.specific_name}-{ag_slot}.json"
        #             save_data_item['ag_slot'] = ag_slot

        #             completion = ""
        #             complete_flag = False
        #             while not complete_flag:
        #                 try:
        #                     completion = gpt35_turbo_completion(prompt_text)
        #                 except Exception as e:
        #                     print(e)
        #                     timer.sleep(10)
        #                 else:
        #                     complete_flag = True
        #                     timer.step()    

        #             # change directory temporarily
        #             cur_dir = os.getcwd()
        #             os.chdir('data')
        #             from data.create_data import normalize
        #             completion = normalize(completion, clean_value=False)
        #             os.chdir(cur_dir)

        #             # override augmented
        #             save_data_item['dialog']['usr'][-1] = completion
        #             save_data_item['turn_slot_values'][ag_slot] = sample_value
        #             save_data_item["slot_values"][ag_slot] = sample_value

        #             print()
        #             print(f"{save_data_item['dialog']['usr'][-1]}")
        #             print(f"augmented turn slot values: {save_data_item['turn_slot_values']}")
        #             print("\n\n")

        #             augmented_result.append(save_data_item)
        #             count += 1
                    
        #         with open(os.path.join(args.output_dir,f'augment_log.json'),'w') as f:
        #             json.dump(augmented_result, f, indent=4)

        # print(count)

   

if __name__ == "__main__":

    augmenter = AugmentDialogue(args.train_fn, args.output_dir, args.specific_name)

    # augmenter.augment_dialogue()

    augmenter.augment_dialogue_MD()