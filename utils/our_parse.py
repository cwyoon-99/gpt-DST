import re

from collections import OrderedDict
from utils.slot_idx import slot_to_idx, idx_to_slot

def our_pred_parse(pred):

    # the output format is "(slot_name = value)"
    start_pos = pred.rfind("(") + 1
    end_pos = pred.rfind(")")
    pred = pred[start_pos:end_pos]

    # fix for no states
    if pred == "":
        return {}
    
    pred_slot_values = {}

    slot_value = pred.split(",")

    value_assigner = "="
    for i in slot_value:
      if value_assigner not in i:
        continue
      else:
        pred_slot_values[i.split(value_assigner)[0].strip()] = i.split(value_assigner)[1].strip()

    return pred_slot_values

def our_pred_parse_with_bracket(pred):
    if pred == "":
       return {}
    
    pred_slot_values = {}

    value_assigner = "="
    slot_value = pred.split(',')
    for i in slot_value:
        bracket_flag = re.search('\(', i)
        if bracket_flag is not None:
            i = i.replace("(","").replace(")","")
            if value_assigner not in i:
               continue
            else:
               pred_slot_values[i.split(value_assigner)[0].strip()] = i.split(value_assigner)[1].strip()

    return pred_slot_values

def pred_parse_with_bracket_matching(pred):

    # find all values where they are in the brackets

    # fix for no states
    if pred == "":
        return {}

    pred_slot_values = {}

    # slot_value = pred.split(",")
    slot_value = re.findall(r'\((.*?)\)', pred)

    value_assigner = "="
    for i in slot_value:
      i = i.replace("(","").replace(")","")
      if value_assigner not in i:
        continue
      else:
        pred_slot_values[i.split(value_assigner)[0].strip()] = i.split(value_assigner)[1].strip()

    return pred_slot_values
    
def slot_classify_parse(pred):

    # # the output format is "(slot_name = value)"
    # start_pos = pred.rfind("(") + 1
    # end_pos = pred.rfind(")")
    # pred = pred[start_pos:end_pos]

    # fix for no states
    if pred == "":
        return {}

    pred_slot_values = {}

    slot_value = pred.split(",")

    value_assigner = "="
    for i in slot_value:
      i = i.replace("(","").replace(")","")
      if value_assigner not in i:
        continue
      else:
        pred_slot = idx_to_slot(int(i.split(value_assigner)[0].strip()))
        pred_value = i.split(value_assigner)[1].strip()
        pred_slot_values[pred_slot] = pred_value

    return pred_slot_values

def sv_dict_to_string(svs, sep=' ', sort=True):
    result_list = [f"{s.replace('-', sep)}{sep}{v}" for s, v in svs.items()]
    if sort:
        result_list = sorted(result_list)
    return ', '.join(result_list)
