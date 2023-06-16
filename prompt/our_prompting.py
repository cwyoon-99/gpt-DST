from utils.slot_idx import slot_to_idx, idx_to_slot

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

slot_description_prompt = """hotel-name : name of the hotel
hotel-pricerange : price budget of the hotel
hotel-type : type of the hotel
hotel-parking : parking facility at the hotel 
hotel-book_stay : length of stay at the hotel
hotel-book_day : day of the hotel booking
hotel-book_people : number of people for the hotel booking
hotel-area : area or place of the hotel
hotel-stars : star rating of the hotel
hotel-internet : internet option at the hotel
train-destination : destination of the train
train-departure : departure location of the train
train-day : day of the train
train-book_people : number of people booking for train
train-leaveat : leaving time for the train
train-arriveby : arrival time of the train
attraction-name : name of the attraction
attraction-area : area or place of the attraction
attraction-type : type of the attraction
restaurant-name : name of the restaurant
restaurant-food : food type for the restaurant 
restaurant-pricerange : price budget for the restaurant 
restaurant-area : area or place of the restaurant 
restaurant-book_time : time of the restaurant booking 
restaurant-book_day : day of the restaurant booking 
restaurant-book_people : number of people booking the restaurant 
taxi-destination : destination of taxi 
taxi-departure : departure location of taxi
taxi-leaveat : leaving time of taxi 
taxi-arriveby : arrival time of taxi

-- answer the following multi-turn conversational questions for the ontology provided above.
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
            # prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"
            
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
    
    # prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"
    prompt_text += "A:"

    return prompt_text

def ett_conversion(prompt, reverse=False):
    # booking time, booking-day, 
    conversion_dict = {"leave": "leaveat", "arrive": "arriveby", "people": "book_people",
                       "stay": "book_stay", "dest": "destination", "depart": "departure",
                       "day": "book_day",
                       "addr": "address",
                       "ref": "reference", 
                       "restaurant-time": "restaurant-book_time",
                       "hotel-day" : "hotel-book_day",
                       "restaurant-day": "restaurant_book_day",
                       "restaurant-price": "restaurant-pricerange",
                       "hotel-price":"hotel-pricerange" }
    reverse_conversion_dict = {v: k for k, v in conversion_dict.items()}
    used_dict = reverse_conversion_dict if reverse else conversion_dict

    for k, v in used_dict.items():
        prompt = prompt.replace(k, v)
    return prompt

def get_prompt_for_ett(data_item, examples, given_context=None, n_examples=None):
    
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
            prompt_text += f"[system] {last_sys_utt}"
            prompt_text += f" {', '.join({f'({conversion(ett_conversion(entity.lower()))})' for entity in example['sys_entity']})}\n"
            prompt_text += f"[user] {example['dialog']['usr'][-1]}"
            prompt_text += f" {', '.join({f'({conversion(ett_conversion(entity.lower()))})' for entity in example['usr_entity']})}\n"

            # prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"
            
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
    prompt_text += f"[system] {last_sys_utt}"
    prompt_text += f" {', '.join({f'({conversion(ett_conversion(entity.lower()))})' for entity in question_item['sys_entity']})}\n"
    prompt_text += f"[user] {question_item['dialog']['usr'][-1]}"
    prompt_text += f" {', '.join({f'({conversion(ett_conversion(entity.lower()))})' for entity in question_item['usr_entity']})}\n"

    
    # prompt_text += f"Q: Based on current dialogue states ([context]), system utterance ([system]), and user utterance ([user]), what domain-slots have been changed and what are their values?\n"
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