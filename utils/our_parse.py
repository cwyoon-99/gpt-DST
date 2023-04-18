from collections import OrderedDict

def our_pred_parse(pred):
    if pred == "":
       return {}
    
    pred_slot_values = {}

    value_assigner = "="
    slot_value = pred.split(', ')
    for i in slot_value:
        i = i.strip('()')
        if value_assigner not in i:
           continue
        else:
           pred_slot_values[i.split(value_assigner)[0].strip()] = i.split(value_assigner)[1].strip()

    return pred_slot_values

def sv_dict_to_string(svs, sep=' ', sort=True):
    result_list = [f"{s.replace('-', sep)}{sep}{v}" for s, v in svs.items()]
    if sort:
        result_list = sorted(result_list)
    return ', '.join(result_list)
