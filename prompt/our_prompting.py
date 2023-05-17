from utils.slot_idx import slot_to_idx, idx_to_slot

custom_prompt = """Your task is to find the changed domain-slots based on the context and the dialogue between user and system, and find the corresponding value.
The following lists are domain-slots and their possible values.
you don't have to find other changed domain-slots if they are not in the list.

hotel-name: a and b guest house, ashley hotel, el shaddia guest house, etc.
hotel-pricerange: dontcare, cheap, moderate, expensive
hotel-type: hotel, guest house
hotel-parking: dontcare, yes, no
hotel-book stay: 1, 2, 3, etc.
hotel-book day: monday, tuesday, etc.
hotel-book people: 1, 2, 3, etc.
hotel-area: dontcare, centre, east, north, south, west
hotel-stars: dontcare, 0, 1, 2, 3, 4, 5
hotel-internet: dontcare, yes, no

train-destination: london kings cross, cambridge, peterborough, etc.
train-departure: cambridge, stansted airport, etc.
train-day: monday, saturday, etc.
train-book people: 1, 2, 3, etc.
train-leaveat: 20:24, 12:06, etc.
train-arriveby: 05:51, 20:52, etc.

attraction-name: abbey pool and astroturf pitch, adc theatre, all saints church, castle galleries, etc.
attraction-area: dontcare, centre, east, north, south, west
attraction-type: architecture, boat, church, cinema, college, concert hall, entertainment, hotspot, multiple sports, museum, nightclub, park, special, swimming pool, theatre

restaurant-name: pizza hut city centre, the missing sock, golden wok, cambridge chop house, darrys cookhouse and wine shop, etc.
restaurant-food: italian, international, chinese, dontcare, modern european, etc.
restaurant-pricerange: dontcare, cheap, moderate, expensive
restaurant-area: centre, east, north, south, west
restaurant-book time: 13:30, 17:11, etc.
restaurant-book day: wednesday, friday, etc.
restaurant-book people: 1, 2, 3, etc.

taxi-destination: copper kettle, magdalene college, lovell lodge
taxi-departure: royal spice, university arms hotel, da vinci pizzeria
taxi-leaveat: 14:45, 11:15, etc.
taxi-arriveby: 15:30, 12:45, etc.
"""

slot_classify_prompt = """Your task is to find the changed domain-slots based on the context and the dialogue between user and system, and find the corresponding value.
The following lists are domain-slots and their possible values.
All domain-slots have their own number that represent themselves in examples.

domain-slots (number): possible values
hotel-name (0): a and b guest house, ashley hotel, el shaddia guest house, etc.
hotel-pricerange (1): dontcare, cheap, moderate, expensive
hotel-type (2): hotel, guest house
hotel-parking (3): dontcare, yes, no
hotel-book stay (4): 1, 2, 3, etc.
hotel-book day (5): monday, tuesday, etc.
hotel-book people (6): 1, 2, 3, etc.
hotel-area (7): dontcare, centre, east, north, south, west
hotel-stars (8): dontcare, 0, 1, 2, 3, 4, 5
hotel-internet (9): dontcare, yes, no
train-destination (10): london kings cross, cambridge, peterborough, etc.
train-departure (11): cambridge, stansted airport, etc.
train-day (12): monday, saturday, etc.
train-book people (13): 1, 2, 3, etc.
train-leaveat (14): 20:24, 12:06, etc.
train-arriveby (15): 05:51, 20:52, etc.
attraction-name (16): abbey pool and astroturf pitch, adc theatre, all saints church, castle galleries, etc.
attraction-area (17): dontcare, centre, east, north, south, west
attraction-type (18): architecture, boat, church, cinema, college, concert hall, entertainment, hotspot, multiple sports, museum, nightclub, park, special, swimming pool, theatre
restaurant-name (19): pizza hut city centre, the missing sock, golden wok, cambridge chop house, darrys cookhouse and wine shop, etc.
restaurant-food (20): italian, international, chinese, dontcare, modern european, etc.
restaurant-pricerange (21): dontcare, cheap, moderate, expensive
restaurant-area (22): centre, east, north, south, west
restaurant-book time (23): 13:30, 17:11, etc.
restaurant-book day (24): wednesday, friday, etc.
restaurant-book people (25): 1, 2, 3, etc.
taxi-destination (26): copper kettle, magdalene college, lovell lodge
taxi-departure (27): royal spice, university arms hotel, da vinci pizzeria
taxi-leaveat (28): 14:45, 11:15, etc.
taxi-arriveby (29): 15:30, 12:45, etc.
"""

slot_description_prompt = """Your task is to find the changed domain-slots based on the context and the dialogue between user and system, and find the corresponding value.
The following lists are domain-slots and their possible values.
All domain-slots have their own number that represent themselves in examples.

(number) domain-slots (description): possible values
(0) hotel-name (name of the hotel): a and b guest house, ashley hotel, el shaddia guest house, etc.
(1) hotel-pricerange (price budget of the hotel): dontcare, cheap, moderate, expensive
(2) hotel-type (what is the type of the hotel): hotel, guest house
(3) hotel-parking (parking facility at the hotel): dontcare, yes, no
(4) hotel-book stay (length of stay at the hotel): 1, 2, 3, etc.
(5) hotel-book day (day of the hotel booking): monday, tuesday, etc.
(6) hotel-book people (number of people for the hotel booking): 1, 2, 3, etc.
(7) hotel-area (area or place of the hotel): dontcare, centre, east, north, south, west
(8) hotel-stars (star rating of the hotel): dontcare, 0, 1, 2, 3, 4, 5
(9) hotel-internet (internet option at the hotel): dontcare, yes, no
(10) train-destination (destination of the train): london kings cross, cambridge, peterborough, etc.
(11) train-departure (departure location of the train): cambridge, stansted airport, etc.
(12) train-day (day of the train): monday, saturday, etc.
(13) train-book people (number of people booking for train): 1, 2, 3, etc.
(14) train-leaveat (leaving time for the train): 20:24, 12:06, etc.
(15) train-arriveby (arrival time of the train): 05:51, 20:52, etc.
(16) attraction-name (name of the attraction): abbey pool and astroturf pitch, adc theatre, all saints church, castle galleries, etc.
(17) attraction-area (area or place of the attraction): dontcare, centre, east, north, south, west
(18) attraction-type (type of the attraction): architecture, boat, church, cinema, college, concert hall, entertainment, hotspot, multiple sports, museum, nightclub, park, special, swimming pool, theatre
(19) restaurant-name (name of the restaurant): pizza hut city centre, the missing sock, golden wok, cambridge chop house, darrys cookhouse and wine shop, etc.
(20) restaurant-food (food type for the restaurant): italian, international, chinese, dontcare, modern european, etc.
(21) restaurant-pricerange (price budget for the restaurant): dontcare, cheap, moderate, expensive
(22) restaurant-area (area or place of the restaurant): centre, east, north, south, west
(23) restaurant-book time (time of the restaurant booking): 13:30, 17:11, etc.
(24) restaurant-book day (day of the restaurant booking): wednesday, friday, etc.
(25) restaurant-book people (number of people booking the restaurant): 1, 2, 3, etc.
(26) taxi-destination (destination of taxi): copper kettle, magdalene college, lovell lodge
(27) taxi-departure (departure location of taxi): royal spice, university arms hotel, da vinci pizzeria
(28) taxi-leaveat (leaving time of taxi): 14:45, 11:15, etc.
(29) taxi-arriveby (arrival time of taxi): 15:30, 12:45, etc.
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
            
            # if not example['turn_slot_values']:
            #     prompt_text += f"A: ()\n"
            # else:
            prompt_text += f"A: {conversion(', '.join({f'({slot} = {value})' for slot, value in example['turn_slot_values'].items()}))}\n"
            prompt_text += "\n\n"

    prompt_text += f"Example #{max_n_examples + 1}\n"
    if given_context is None:
        # remove mulitple choice
        last_slot_values = {s: v.split(
            '|')[0] for s, v in question_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    prompt_text += f"[context] {conversion(', '.join({f'({slot} = {value})' for slot, value in last_slot_values.items()}))}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    
    prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"
    prompt_text += "A:"

    return prompt_text

def get_slot_classify_prompt(data_item, examples, given_context=None, n_examples=None):
    
    question_item = data_item

    prompt_text = f"{conversion(slot_description_prompt)}\n"

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
            
            prompt_text += f"[context] {conversion(', '.join({f'({slot_to_idx(slot)} = {value})' for slot, value in last_slot_values.items()}))}\n"

            last_sys_utt = example['dialog']['sys'][-1]
            if last_sys_utt == 'none':
                last_sys_utt = ''
            prompt_text += f"[system] {last_sys_utt}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}\n"
            prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"

            prompt_text += f"A: {conversion(', '.join({f'({slot_to_idx(slot)} = {value})' for slot, value in example['turn_slot_values'].items()}))}\n"
            prompt_text += "\n\n"

    prompt_text += f"Example #{max_n_examples + 1}\n"
    if given_context is None:
        # remove mulitple choice
        last_slot_values = {s: v.split(
            '|')[0] for s, v in question_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    prompt_text += f"[context] {conversion(', '.join({f'({slot_to_idx(slot)} = {value})' for slot, value in last_slot_values.items()}))}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    
    prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"
    prompt_text += "A: "

    return prompt_text

def get_intent_prompt(data_item, given_context=None):
    
    question_item = data_item

    prompt_text = ""

    prompt_text += f"Example\n"
    if given_context is None:
        # remove mulitple choice
        last_slot_values = {s: v.split(
            '|')[0] for s, v in question_item['last_slot_values'].items()}
    else:
        last_slot_values = given_context
    prompt_text += f"[context] {conversion(', '.join({f'{(slot)} = {value}' for slot, value in last_slot_values.items()}))}\n"

    last_sys_utt = question_item['dialog']['sys'][-1]
    if last_sys_utt == 'none':
        last_sys_utt = ''
    prompt_text += f"[system] {last_sys_utt}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
    
    prompt_text += f"\nIn the example above, [context] is dialogue states that represent previous state of dialogue turns.\n[system] and [user] are a current system utterance and user utterance respectively.\nQ: what is the intent of user?\n"
    prompt_text += f"A:"

    return prompt_text

def get_implicit_info_prompt(data_item, previous_prompt=None, intent_completion=None):
    prompt_text = previous_prompt

    prompt_text += f" {intent_completion}\n"

    prompt_text += f"Q: then, Is there any implicit information between [user] and [system] that we need to consider when answering the question?\n"

    prompt_text += f"A:"

    return prompt_text

def get_follow_up_cot_prompt(data_item, examples=None, given_context=None, n_examples=None):
    
    question_item = data_item

    prompt_text = f"{conversion(custom_prompt)}\n"

    # with examples
    if examples is not None:
        max_n_examples = len(examples)
        if n_examples is not None:
            max_n_examples = n_examples

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
        prompt_text += f"[context] {conversion(', '.join({f'({(slot)} = {value})' for slot, value in last_slot_values.items()}))}\n"

        last_sys_utt = question_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        prompt_text += f"[system] {last_sys_utt}\n"
        prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"

        prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"

        # prompt_text += f"(Just so you know, {data_item['intent_completion']} Also, {data_item['implicit_completion']})\n"

        # prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"
        # prompt_text += f"A:"

        # return prompt_text

        if not data_item['predicted_slot_values']:
            prompt_text += f" the changed domain-slots and values is (), ()."
        elif len(data_item['predicted_slot_values']) == 1:
            prompt_text += f" the changed domain-slots and values is {', '.join({f'({(slot)} = {value})' for slot, value in data_item['predicted_slot_values'].items()})}, ()."
        else:
            prompt_text += f"FYI: the changed domain-slots and values is {', '.join({f'({(slot)} = {value})' for slot, value in data_item['predicted_slot_values'].items()})}."
        prompt_text += f" Just so you know, {data_item['intent_completion']} Also, {data_item['implicit_completion']}\n"
        prompt_text += f"A:"

        return prompt_text

    else:
        # without examples
        prompt_text += f"Example\n"

        if given_context is None:
            # remove mulitple choice
            last_slot_values = {s: v.split(
                '|')[0] for s, v in question_item['last_slot_values'].items()}
        else:
            last_slot_values = given_context
        prompt_text += f"[context] {conversion(', '.join({f'({(slot)} = {value})' for slot, value in last_slot_values.items()}))}\n"

        last_sys_utt = question_item['dialog']['sys'][-1]
        if last_sys_utt == 'none':
            last_sys_utt = ''
        prompt_text += f"[system] {last_sys_utt}\n"
        prompt_text += f"[user] {question_item['dialog']['usr'][-1]}\n"
        
        prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"
        prompt_text += f"A:"
        # prompt_text += f" considering the question simply,"

        if not data_item['predicted_slot_values']:
            prompt_text += f" the changed domain-slots and values is ()."
        elif len(data_item['predicted_slot_values']) == 1:
            prompt_text += f" the changed domain-slots and values is {', '.join({f'({(slot)} = {value})' for slot, value in data_item['predicted_slot_values'].items()})}, ()."
        else:
            prompt_text += f" the changed domain-slots and values is {', '.join({f'({(slot)} = {value})' for slot, value in data_item['predicted_slot_values'].items()})}."

        # prompt_text += f" Besides, {data_item['intent_completion']}"
        prompt_text += f" Just so you know, {data_item['intent_completion']}"
        prompt_text += f" Also, {data_item['implicit_completion']}"
        # prompt_text += f" Consequently, the changed domain-slots and values is"
        prompt_text += f" Therefore, the changed domain-slots and values is"

        return prompt_text