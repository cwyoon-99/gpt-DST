import os
import json
import math

def loadJson(data_file):
    if os.path.isfile(data_file):
        with open(data_file, 'r') as read_file:
            data = json.load(read_file)
            return data
    
def getBeliefSet(ds):
    bs = set()

    # error handling for the cases that model did not follow the output format.
    for key, value in ds.items():
        t = key+"-"+value
        bs.add(t)
    
#     new_ds = {}
#     for key, value in ds.items():
# #         domain = key.split('-')[0]
# #         slot = key.split('-')[1]
#         domain, slot = key.split('-')
#         if domain not in new_ds:
#             new_ds[domain] = {}
#         new_ds[domain][slot] = value
      
#     for dom in new_ds:
#         for slot in new_ds[dom]:
#             t = dom+"-"+slot+"-"+new_ds[dom][slot]
#             bs.add(t)
            
    return bs

# Flexible Goal Accuracy
def getFGA(gt_list, pr_list, turn_diff, L):
    gt = gt_list[-1]
    pr = pr_list[-1]
    diff1 = gt.symmetric_difference(pr)
    if len(diff1)==0: #Exact match
        return 1
    else:
        if len(gt_list)==1: 
            #Type 1 error
            #First turn is wrong
            return 0
        else:
            diff2 = gt_list[-2].symmetric_difference(pr_list[-2])
            if len(diff2)==0: 
                #Type 1 error
                #Last turn was correct i.e the error in current turn
                return 0
            else:
                tgt = gt.difference(gt_list[-2])
                tpr = pr.difference(pr_list[-2])
                if(not tgt.issubset(pr) or not tpr.issubset(gt)): 
                    #Type 1 error
                    #There exists some undetected/false positive intent in the current prediction
                    return 0
                else:
                    #Type 2 error
                    #Current turn is correct but source of the error is some previous turn
                    return (1-math.exp(-L*turn_diff))


# Run
def FGA(dir_path):
    dst_res_path = os.path.join(dir_path, 'running_log.json')
    dst_res = loadJson(dst_res_path)
    
    fga = [0, 0, 0, 0]
    turn_acc = 0
    total = 0
    lst_lambda = [0.25, 0.5, 0.75, 1.0]

    res = []
    for t in dst_res:
      if t not in res:
        res.append(t)

    for turn in res:
        if turn['turn_id'] == 0:
            gt_list = []
            pr_list = []
            error_turn = -1

        total+=1
        
#         print(turn['pred'])
        gt = getBeliefSet(turn['slot_values'])
        pr = getBeliefSet(turn['pred'])
        gt_list.append(gt)
        pr_list.append(pr)

        m = 0
        for l in range(len(lst_lambda)):
            m = getFGA(gt_list, pr_list, turn['turn_id']-error_turn, lst_lambda[l])
            fga[l]+=m
        if(m==0):
            error_turn = turn['turn_id']
        else:
            turn_acc+=1
    
    result = []
    for l in range(len(lst_lambda)):
        fga_acc = round(fga[l]*100.0/total, 2)
        result.append(f"FGA with L={lst_lambda[l]} : {fga_acc}")
    
    return result