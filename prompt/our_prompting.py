custom_prompt = """Your task is to find the changed domain-slots based on the context and the dialogue between user and system, and find the corresponding value.
The following lists are domain-slots and their possible values.
you don't have to find other changed domain-slots if they are not in the list.

hotel-name: a and b guest house, ashley hotel, el shaddia guest house, etc.
hotel-pricerange: dontcare, cheap, moderate, expensive
hotel-type: hotel, guest house
hotel-parking: dontcare, yes, no
hotel-book_stay: 1, 2, 3, etc.
hotel-book_day: monday, tuesday, etc.
hotel-book_people: 1, 2, 3, etc.
hotel-area: dontcare, centre, east, north, south, west
hotel-stars: dontcare, 0, 1, 2, 3, 4, 5
hotel-internet: dontcare, yes, no

train-destination: london kings cross, cambridge, peterborough, etc.
train-departure: cambridge, stansted airport, etc.
train-day: monday, saturday, etc.
train-book_people: 1, 2, 3, etc.
train-leaveat: 20:24, 12:06, etc.
train-arriveby: 05:51, 20:52, etc.

attraction-name: abbey pool and astroturf pitch, adc theatre, all saints church, castle galleries, etc.
attraction-area: dontcare, centre, east, north, south, west
attraction-type: architecture, boat, church, cinema, college, concert hall, entertainment, hotspot, multiple sports, museum, nightclub, park, special, swimming pool, theatre

restaurant-name: pizza hut city centre, the missing sock, golden wok, cambridge chop house, darrys cookhouse and wine shop, etc.
restaurant-food: italian, international, chinese, dontcare, modern european, etc.
restaurant-pricerange: dontcare, cheap, moderate, expensive
restaurant-area: centre, east, north, south, west
restaurant-book_time: 13:30, 17:11, etc.
restaurant-book_day: wednesday, friday, etc.
restaurant-book_people: 1, 2, 3, etc.

taxi-destination: copper kettle, magdalene college, lovell lodge
taxi-departure: royal spice, university arms hotel, da vinci pizzeria
taxi-leaveat: 14:45, 11:15, etc.
taxi-arriveby: 15:30, 12:45, etc.
"""

def conversion(prompt, reverse=False):
    conversion_dict = {"leaveat": "depart_time", "arriveby": "arrive_by_time",
                       "book_stay": "book_number_of_days",
                       "food": "food_type"}
    reverse_conversion_dict = {v: k for k, v in conversion_dict.items()}
    used_dict = reverse_conversion_dict if reverse else conversion_dict

    for k, v in used_dict.items():
        prompt = prompt.replace(k, v)
    return prompt

def get_our_prompt(data_item, examples, given_context=None, n_examples=None):
    
    question_item = data_item

    prompt_text = f"{conversion(custom_prompt)}\n"

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:]):
            prompt_text += f"Example #{example_id + 1}\n"

            # remove multiple choice in last slot values
            last_slot_values = {s: v.split(
                '|')[0] for s, v in example['last_slot_values'].items()}
            
            prompt_text += f"[context] {conversion(', '.join({f'{slot} = {value}' for slot, value in last_slot_values.items()}))}\n"

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[system] {last_sys_utt}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"
            prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"

            prompt_text += f"A: ({conversion(', '.join({f'{slot} = {value}' for slot, value in example['turn_slot_values'].items()}))})\n"
            prompt_text += "\n\n"

    prompt_text += f"Example #{max_n_examples + 1}\n"
    if given_context is None:
        # remove mulitple choice
        last_slot_values = {s: v.split(
            '|')[0] for s, v in question_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    prompt_text += f"[context] {conversion(', '.join({f'{slot} = {value}' for slot, value in last_slot_values.items()}))}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    
    prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"
    prompt_text += "A: "

    return prompt_text

def get_prompt_with_bracket(data_item, examples, given_context=None, n_examples=None):
    
    question_item = data_item

    prompt_text = f"{conversion(custom_prompt)}\n"

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:]):
            prompt_text += f"Example #{example_id + 1}\n"

            # remove multiple choice in last slot values
            last_slot_values = {s: v.split(
                '|')[0] for s, v in example['last_slot_values'].items()}
            
            prompt_text += f"[context] {conversion(', '.join({f'({slot} = {value})' for slot, value in last_slot_values.items()}))}\n"

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[system] {last_sys_utt}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"
            prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"

            prompt_text += f"A: {conversion(', '.join({f'({slot} = {value})' for slot, value in example['turn_slot_values'].items()}))}\n"
            prompt_text += "\n\n"

    prompt_text += f"Example #{max_n_examples + 1}\n"
    if given_context is None:
        # remove mulitple choice
        last_slot_values = {s: v.split(
            '|')[0] for s, v in question_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    prompt_text += f"[context] {conversion(', '.join({f'{slot} = {value}' for slot, value in last_slot_values.items()}))}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    
    prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"
    prompt_text += "A: "

    return prompt_text

def get_two_stage_1_prompt(data_item, examples, given_context=None, n_examples=None):
    # test instance에 slot을 먼저 예측.
    question_item = data_item

    prompt_text = f"{conversion(custom_prompt)}\n"

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:]):
            prompt_text += f"Example #{example_id + 1}\n"

            # remove multiple choice in last slot values
            last_slot_values = {s: v.split(
                '|')[0] for s, v in example['last_slot_values'].items()}
            
            prompt_text += f"[context] {conversion(', '.join({f'{slot} = {value}' for slot, value in last_slot_values.items()}))}\n"

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[system] {last_sys_utt}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"
            prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), which slots are changed?\n"

            prompt_text += f"A: ({conversion(', '.join({f'{slot}' for slot, value in example['turn_slot_values'].items()}))})\n"
            prompt_text += "\n\n"

    prompt_text += f"Example #{max_n_examples + 1}\n"
    if given_context is None:
        # remove mulitple choice
        last_slot_values = {s: v.split(
            '|')[0] for s, v in question_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    prompt_text += f"[context] {conversion(', '.join({f'{slot} = {value}' for slot, value in last_slot_values.items()}))}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), which slots are changed?\n"

    prompt_text += "A: "

    return prompt_text

def get_two_stage_2_prompt(data_item, examples, given_context=None, n_examples=None):
    # test instance에 slot을 먼저 예측.
    question_item = data_item

    prompt_text = f"{conversion(custom_prompt)}\n"

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:]):
            prompt_text += f"Example #{example_id + 1}\n"

            # remove multiple choice in last slot values
            last_slot_values = {s: v.split(
                '|')[0] for s, v in example['last_slot_values'].items()}
            
            prompt_text += f"[context] {conversion(', '.join({f'{slot} = {value}' for slot, value in last_slot_values.items()}))}\n"

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[system] {last_sys_utt}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"
            prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), which slots are changed?\n"
            prompt_text += f"A: the changed slot is ({conversion(', '.join({f'{slot}' for slot, value in example['turn_slot_values'].items()}))}).\n"
            prompt_text += f"Q: Then, I want you to return a set of changed slots and values.\n"
            prompt_text += f"A: ({conversion(', '.join({f'{slot} = {value}' for slot, value in example['turn_slot_values'].items()}))})\n"
            prompt_text += "\n\n"

    prompt_text += f"Example #{max_n_examples + 1}\n"
    if given_context is None:
        # remove mulitple choice
        last_slot_values = {s: v.split(
            '|')[0] for s, v in question_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    prompt_text += f"[context] {conversion(', '.join({f'{slot} = {value}' for slot, value in last_slot_values.items()}))}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), which slots are changed?\n"
    prompt_text += f"A: the changed slot is {question_item['slot_completion']}\n"
    prompt_text += f"Q: Then, I want you to return a set of changed slots and values.\n"
    prompt_text += f"A:"

    return prompt_text

# For each example, we request API to generate Chain-of-thought prompt
def get_each_cot_prompt(example):
    prompt_text = f"{conversion(custom_prompt)}\n"

    prompt_text += f"Example \n"

    
    # remove multiple choice in last slot values
    last_slot_values = {s: v.split(
        '|')[0] for s, v in example['last_slot_values'].items()}
        
    prompt_text += f"[context] {conversion(', '.join({f'{slot} = {value}' for slot, value in last_slot_values.items()}))}\n"

    last_sys_utt = example['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"

    prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), which slots are changed and what values do the slots have?\n"
    prompt_text += f"A: the answer (slot_name = value) is ({conversion(', '.join({f'{slot} = {value}' for slot, value in example['turn_slot_values'].items()}))}). "
    prompt_text += f"Let's think why the answer is correct in step by step within 150 tokens."

    return prompt_text

def get_cot_augmented_prompt(data_item, examples, given_context=None, n_examples=None):

    question_item = data_item

    prompt_text = f"{conversion(custom_prompt)}\n"

    max_n_examples = len(examples)
    if n_examples is not None:
        max_n_examples = n_examples

    # in case for zero-shot learning
    if max_n_examples > 0:
        for example_id, example in enumerate(examples[-max_n_examples:]):
            prompt_text += f"Example #{example_id + 1}\n"

            # remove multiple choice in last slot values
            last_slot_values = {s: v.split(
                '|')[0] for s, v in example['last_slot_values'].items()}
            
            prompt_text += f"[context] {conversion(', '.join({f'{slot} = {value}' for slot, value in last_slot_values.items()}))}\n"

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[system] {last_sys_utt}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"

            prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), which slots are changed and what values do the slots have?\n"
            
            # base CoT
            # prompt_text += f"A: {example['cot_completion']} "

            # slot provided version
            prompt_text += f"A: the changed slot is ({conversion(', '.join({f'{slot}' for slot, value in example['turn_slot_values'].items()}))}). "
            prompt_text += f"{example['cot_completion']} "

            prompt_text += f"Therefore, the changed slot and value (slot_name = value) is "
            prompt_text += f"({conversion(', '.join({f'{slot} = {value}' for slot, value in example['turn_slot_values'].items()}))})\n"
            prompt_text += "\n\n"

    prompt_text += f"Example #{max_n_examples + 1}\n"
    if given_context is None:
        # remove mulitple choice
        last_slot_values = {s: v.split(
            '|')[0] for s, v in question_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    prompt_text += f"[context] {conversion(', '.join({f'{slot} = {value}' for slot, value in last_slot_values.items()}))}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"

    prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), which slots are changed and what values do the slots have?\n"
    
    # base CoT
    # prompt_text += "A:"

    # slot provided version
    prompt_text += "A: "
    prompt_text += f"the changed slot is ({conversion(', '.join({f'{slot}' for slot, value in question_item['turn_slot_values'].items()}))})."


    return prompt_text
