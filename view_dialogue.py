import json

log_path = 'expts/230509_0236-gpt35_turbo_5p_v2_baseline_0to1842/running_log.json'
with open(log_path) as f:
    data = json.load(f)

previous_id = ''
turn_num = 0
for turn in data[:50]:
    current_id = turn['ID']
    if current_id != previous_id:
        print()
        print(turn_num)
        print('Dialogue ID: ' + current_id)
    print('system: ' + turn['dialog']['sys'][-1])
    print('user: ' + turn['dialog']['usr'][-1])
    previous_id = current_id
    turn_num += 1
print()