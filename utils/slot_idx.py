slot_idx_dict = {'hotel-name': 0,
 'hotel-pricerange': 1,
 'hotel-type': 2,
 'hotel-parking': 3,
 'hotel-book stay': 4,
 'hotel-book day': 5,
 'hotel-book people': 6,
 'hotel-area': 7,
 'hotel-stars': 8,
 'hotel-internet': 9,
 'train-destination': 10,
 'train-departure': 11,
 'train-day': 12,
 'train-book people': 13,
 'train-leaveat': 14,
 'train-arriveby': 15,
 'attraction-name': 16,
 'attraction-area': 17,
 'attraction-type': 18,
 'restaurant-name': 19,
 'restaurant-food': 20,
 'restaurant-pricerange': 21,
 'restaurant-area': 22,
 'restaurant-book time': 23,
 'restaurant-book day': 24,
 'restaurant-book people': 25,
 'taxi-destination': 26,
 'taxi-departure': 27,
 'taxi-leaveat': 28,
 'taxi-arriveby': 29}

def slot_to_idx(str):
    # str = str.replace(" ","_")
    if str in slot_idx_dict.keys():
        return slot_idx_dict[str]
    else:
        return -1

def idx_to_slot(idx):
    if idx in slot_idx_dict.values():
        return {v: k for k, v in slot_idx_dict.items()}[idx]
    else:
        return "invalid_slot_idx"