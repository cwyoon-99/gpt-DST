import os
import json
import random
import time
import argparse
from tqdm import tqdm
import copy

from prompt.our_prompting import conversion
from api_request.gpt35_turbo_completion import gpt35_turbo_completion
from utils.helper import SpeedLimitTimer

train_fn = "data/mw21_0.02p_train_v2.json"
output_file_name = "test"
output_dir = "./para/"
mwz_ver = "2.4"
specific_name = "para"
sample_rate = 1

cur_time = time.strftime('%y%m%d_%H%M-')

output_dir = 'para/' + cur_time + "spr" + str(sample_rate) + "_" + output_file_name
os.makedirs(output_dir, exist_ok=True)

class ParaDialogue:
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

    def paraphrase(self):
        timer = SpeedLimitTimer(second_per_step=4)

        # paraphrase
        para_result = []
        sampled_dataset = random.sample(self.dataset, int(len(self.dataset) * sample_rate))
        for data_idx, data_item in enumerate(tqdm(sampled_dataset)):
        # for data_idx, data_item in enumerate(tqdm(self.dataset)):
            prompt_text = ""

            last_slot_values = {s: v.split(
                '|')[0] for s, v in data_item['last_slot_values'].items()}
            # prompt_text += f"[context] {conversion(', '.join({f'({slot} = {value})' for slot, value in last_slot_values.items()}))}\n"
            
            last_sys_utt = data_item['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[system] {last_sys_utt}\n"
            prompt_text += f"[user] {data_item['dialog']['usr'][-1]}\n\n"
            if data_item['turn_slot_values']:
                prompt_text += f"The dialogue state of the above dialogue is {conversion(', '.join({f'({slot} = {value})' for slot, value in data_item['turn_slot_values'].items()}))}\n"
            prompt_text += f"Paraphrase the dialogue with"

            sys_exist = True
            if data_item['dialog']['sys'][-1] == "":
                sys_exist = False

            if sys_exist:
                prompt_text += f" [system] and"
            prompt_text += f" [user] prefix." 
            
            # if sys_exist:    
            #     prompt_text += f"(You should generate the [system] first, then [user]. Also, [system] and [user] should be one, respectively.)\n"
            # else:
            #     prompt_text += f"([user] should be one.)\n"

            # # rearrange the order of information presented
            # prompt_text += f"In addition, if possible, try to rearrange the order of information in each [system] and [user]. Don't generate the continuing dialogues."

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

            print("="*60)
            print('### prompt_text ###')
            print(prompt_text)
            print("-"*60)
            print('### completion ###')
            print(completion)

            # To filter unnecessary dialogue, extract first two line from completion.
            temp = []
            for line in completion.split("\n"):
                if "]" in line:
                    temp.append(line)

            completion = '\n'.join(temp[:2]) if sys_exist else '\n'.join(temp[:1])

            # change directory temporarily
            cur_dir = os.getcwd()
            os.chdir('data')
            from data.create_data import normalize
            # sys_utt = normalize(sys_utt, clean_value=False)
            # usr_utt = normalize(usr_utt, clean_value=False)
            completion = normalize(completion, clean_value=False)
            os.chdir(cur_dir)
            # print("-"*60)
            # print('### completion (after processing) ###')
            # print(completion)

            sys_utt = completion.split("[user]")[0].replace("[system]","").strip()
            usr_utt = completion.split("[user]")[1].strip()

            data_item['ID'] = f"{data_item['ID'].split('.')[0]}-{specific_name}.json"
            # save original
            data_item["original_sys"] = data_item['dialog']['sys'][-1]
            data_item["original_usr"] = data_item['dialog']['usr'][-1]
            # save change
            data_item["changed_sys"] = sys_utt
            data_item["changed_usr"] = usr_utt
            # override augmented
            data_item['dialog']['sys'][-1] = sys_utt
            data_item['dialog']['usr'][-1] = usr_utt

            print("-"*60)
            print('### original dialogue ###')
            print(f"[system] {data_item['original_sys']}")
            print(f"[user] {data_item['original_usr']}")
            print('### changed dialogue ###')
            print(f"[system] {data_item['changed_sys']}")
            print(f"[user] {data_item['changed_usr']}")
            print("="*60)
            print("\n\n\n")

            para_result.append(data_item)

            if data_idx % 5 == 0:
                with open(os.path.join(output_dir,f'para_log.json'),'w') as f:
                    json.dump(para_result, f, indent=4)

    def cot_paraphrase(self):
        timer = SpeedLimitTimer(second_per_step=4)

        # paraphrase
        para_result = []
        sampled_dataset = random.sample(self.dataset, int(len(self.dataset) * sample_rate))
        for data_idx, data_item in enumerate(tqdm(sampled_dataset)):
            sys_exist = True
            if data_item['dialog']['sys'][-1] == "":
                sys_exist = False
            if data_item['turn_slot_values'] and len(data_item['turn_slot_values']) > 1 and sys_exist:
                print(f"\n***************** {data_idx} *****************")
                step_sys = ""
                step_usr = ""
                step_dialogue = ""
                random_order_slot = list(data_item['turn_slot_values'].keys())
                random.shuffle(random_order_slot)

                for idx, tg_slot in enumerate(random_order_slot):
                    print(f"--------------------step {idx} start----------------------")
                    prompt_text = ""
                    tg_value = data_item['turn_slot_values'][tg_slot]
                    if idx == 0:
                        prompt_text += "# example\n"

                        last_slot_values = {s: v.split(
                            '|')[0] for s, v in data_item['last_slot_values'].items()}
                        # prompt_text += f"[context] {conversion(', '.join({f'({slot} = {value})' for slot, value in last_slot_values.items()}))}\n"
                        
                        last_sys_utt = data_item['dialog']['sys'][-1]
                        if last_sys_utt == 'none':
                            last_sys_utt = ''           

                        prompt_text += f"[system] {last_sys_utt}\n"
                        prompt_text += f"[user] {data_item['dialog']['usr'][-1]}\n"

                        prompt_text += f"The Answer of example is {conversion(', '.join({f'({slot} = {value})' for slot, value in data_item['turn_slot_values'].items()}))}\n\n"


                        # other_turns = copy.deepcopy(data_item['turn_slot_values'])
                        # del other_turns[random_key]
                        exclude_list = []
                        for k, _ in self.ontology.items():
                            if tg_slot.split('-')[0] in k:
                                exclude_list.append(k)

                        turn_slot_list = list(data_item['turn_slot_values'].keys())
                        for i in exclude_list:
                            if i in turn_slot_list:
                                exclude_list.remove(i)

                        # prompt_text += f"Generate single [system] and single [user] including ({conversion(tg_slot)} = {tg_value}) and excluding {', '.join(exclude_list)}. (You should generate the [system] first, then [user].)"
                        # prompt_text += f"Generate single [system] and single [user] to have ({conversion(tg_slot)} = {tg_value}), {', '.join(f'({slot})' for slot in random_order_slot[1:])} in answer. (You should generate the [system] first, then [user].)"
                        prompt_text += f"Generate single [system] and single [user] including ({conversion(tg_slot)} = {tg_value}) in answer. (You should generate the [system] first, then [user].)"

                    else:
                        prompt_text += step_dialogue
                        # prompt_text += f"Augment the dialogue with single [system] and single [user] to additionally have ({conversion(tg_slot)} = {tg_value}) in answer. (You should generate the [system] first, then [user])"
                        prompt_text += f"Change the user utterance with single [user] to additionally have ({conversion(tg_slot)} = {tg_value}) in answer. changed [user]:"

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


                    if idx == 0:

                        # To filter unnecessary dialogue, extract first two line from completion.
                        temp = []
                        for line in completion.split("\n"):
                            if "]" in line:
                                temp.append(line)

                        step_dialogue = '\n'.join(temp[:2]) if sys_exist else '\n'.join(temp[:1])

                        # change directory temporarily
                        cur_dir = os.getcwd()
                        os.chdir('data')
                        from data.create_data import normalize
                        normalized_step_dialogue = normalize(step_dialogue, clean_value=False)
                        os.chdir(cur_dir)

                        step_sys = f'[system] {normalized_step_dialogue.split("[user]")[0].replace("[system]","").strip()}'
                        step_usr = normalized_step_dialogue.split("[user]")[1].strip()

                    else:
                        step_usr = completion

                        step_dialogue = f"{step_sys}\n[user] {step_usr}"

                    temp_s = []
                    for slot in random_order_slot[:idx+1]:
                        temp_s.append(f'({slot} = {data_item["turn_slot_values"][slot]})')

                    step_dialogue += f"\n\nThe Answer of dialogue is {conversion(', '.join(temp_s))}\n"

    def value_paraphrase(self):
        timer = SpeedLimitTimer(second_per_step=4)

        # paraphrase
        para_result = []
        sampled_dataset = random.sample(self.dataset, int(len(self.dataset) * sample_rate))
        for data_idx, data_item in enumerate(tqdm(sampled_dataset)):
        # for data_idx, data_item in enumerate(tqdm(self.dataset)):
            prompt_text = ""

            last_slot_values = {s: v.split(
                '|')[0] for s, v in data_item['last_slot_values'].items()}
            # prompt_text += f"[context] {conversion(', '.join({f'({slot} = {value})' for slot, value in last_slot_values.items()}))}\n"
            
            last_sys_utt = data_item['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[system] {last_sys_utt}\n"
            prompt_text += f"[user] {data_item['dialog']['usr'][-1]}\n\n"
            if data_item['turn_slot_values']:
                prompt_text += f"The dialogue state of the above dialogue is {conversion(', '.join({f'({slot} = {value})' for slot, value in data_item['turn_slot_values'].items()}))}\n"
            prompt_text += f"Paraphrase the dialogue with"

            sys_exist = True
            if data_item['dialog']['sys'][-1] == "":
                sys_exist = False

            if sys_exist:
                prompt_text += f" [system] and"
            prompt_text += f" [user] prefix." 
            
            # if sys_exist:    
            #     prompt_text += f"(You should generate the [system] first, then [user]. Also, [system] and [user] should be one, respectively.)\n"
            # else:
            #     prompt_text += f"([user] should be one.)\n"

            # # rearrange the order of information presented
            # prompt_text += f"In addition, if possible, try to rearrange the order of information in each [system] and [user]. Don't generate the continuing dialogues."

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

            print("="*60)
            print('### prompt_text ###')
            print(prompt_text)
            print("-"*60)
            print('### completion ###')
            print(completion)

            # To filter unnecessary dialogue, extract first two line from completion.
            temp = []
            for line in completion.split("\n"):
                if "]" in line:
                    temp.append(line)

            completion = '\n'.join(temp[:2]) if sys_exist else '\n'.join(temp[:1])

            # change directory temporarily
            cur_dir = os.getcwd()
            os.chdir('data')
            from data.create_data import normalize
            # sys_utt = normalize(sys_utt, clean_value=False)
            # usr_utt = normalize(usr_utt, clean_value=False)
            completion = normalize(completion, clean_value=False)
            os.chdir(cur_dir)
            # print("-"*60)
            # print('### completion (after processing) ###')
            # print(completion)

            sys_utt = completion.split("[user]")[0].replace("[system]","").strip()
            usr_utt = completion.split("[user]")[1].strip()

            data_item['ID'] = f"{data_item['ID'].split('.')[0]}-{specific_name}.json"
            # save original
            data_item["original_sys"] = data_item['dialog']['sys'][-1]
            data_item["original_usr"] = data_item['dialog']['usr'][-1]
            # save change
            data_item["changed_sys"] = sys_utt
            data_item["changed_usr"] = usr_utt
            # override augmented
            data_item['dialog']['sys'][-1] = sys_utt
            data_item['dialog']['usr'][-1] = usr_utt

            print("-"*60)
            print('### original dialogue ###')
            print(f"[system] {data_item['original_sys']}")
            print(f"[user] {data_item['original_usr']}")
            print('### changed dialogue ###')
            print(f"[system] {data_item['changed_sys']}")
            print(f"[user] {data_item['changed_usr']}")
            print("="*60)
            print("\n\n\n")

            para_result.append(data_item)

            if data_idx % 5 == 0:
                with open(os.path.join(output_dir,f'para_log.json'),'w') as f:
                    json.dump(para_result, f, indent=4)


if __name__ == "__main__":

    para = ParaDialogue(train_fn, output_dir, specific_name)

    para.value_paraphrase()

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