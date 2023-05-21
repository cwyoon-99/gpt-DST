import json

dir_name = '230521_1654-gpt35_mw24_1p_v2_excluded_history_737'
log_path = 'expts/'+ dir_name +'/running_log.json'
with open(log_path) as f:
    data = json.load(f)

previous_id = ''
turn_num = 0
for turn in data[:5]:
    current_id = turn['ID']
    if current_id != previous_id:
        print('\n\n\n\n\n')
        print(f'### Start Dialogue Turn Number: {turn_num} ###')
        print(f'### Dialogue ID: {current_id} ###')
    print('[system] ' + turn['dialog']['sys'][-1])
    print('[user] ' + turn['dialog']['usr'][-1])
    print('\n### prompt ###\n' + turn['prompt'])
    previous_id = current_id
    turn_num += 1
print()