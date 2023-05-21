import json

dir_name = '230520_1527-gpt35_turbo_5p_v2_excluded_history_0to10'
log_path = 'expts/'+ dir_name +'/running_log.json'
with open(log_path) as f:
    data = json.load(f)

previous_id = ''
turn_num = 0
for turn in data[8:9]:
    current_id = turn['ID']
    if current_id != previous_id:
        print()
        print(turn_num)
        print('Dialogue ID: ' + current_id)
    # print('[system] ' + turn['dialog']['sys'][-1])
    # print('[user] ' + turn['dialog']['usr'][-1])
    print('')
    print('prompt: ' + turn['prompt'])
    previous_id = current_id
    turn_num += 1
print()